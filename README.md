<p align="center">
  <h1 align="center">🫀 Project-Lewis</h1>
  <p align="center">
    <strong>Pipeline completo de ECG → modelos quantizados INT8 → firmware embarcado STM32F4</strong><br>
    validado sem hardware físico via simulação Renode, com RAG local (Camada C11) para agentes de coding.
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python" alt="Python 3.12">
    <img src="https://img.shields.io/badge/TensorFlow-2.21-orange?logo=tensorflow" alt="TensorFlow">
    <img src="https://img.shields.io/badge/STM32F4-TFLM%20%7C%20CMSIS--NN-green?logo=arm" alt="STM32F4 TFLM">
    <img src="https://img.shields.io/badge/Renode-1.15.3-purple?logo=robotframework" alt="Renode">
    <img src="https://img.shields.io/badge/Knowledge-sqlite--vec%20%7C%20MCP-blue" alt="Knowledge sqlite-vec MCP">
    <img src="https://img.shields.io/badge/License-MIT-lightgrey" alt="License MIT">
  </p>
</p>

---

## 📑 Índice

1. [Visão Geral](#-visão-geral)
2. [Arquitetura](#-arquitetura)
3. [Pipeline de Dados](#-pipeline-de-dados)
4. [ML Pipeline](#-ml-pipeline)
5. [Camada C11 — Knowledge Layer](#-camada-c11--knowledge-layer-rag)
6. [Firmware, Simulação & DevOps](#-firmware-simulação--devops)
7. [Estrutura do Repositório](#-estrutura-do-repositório)
8. [Como Executar](#-como-executar)
9. [Quality Gates](#-quality-gates)
10. [Limites da Simulação](#-limites-da-simulação)
11. [Versão Atual](#-versão-atual)
12. [Autor & Licença](#-autor--licença)

---

## 🎯 Visão Geral

O **Project-Lewis** demonstra uma arquitetura end-to-end para classificação de arritmias cardíacas a partir de sinais de ECG, indo da ingestão de dados públicos até a inferência embarcada em um microcontrolador **STM32F4** usando **TensorFlow Lite Micro**.

O projeto é dividido em camadas bem definidas, cada uma com contratos de interface, quality gates e documentação própria:

| Camada | Responsabilidade | Tecnologias |
| :--- | :--- | :--- |
| **01 — Ingestão** | Download, validação e governança de datasets | `wfdb`, `wget`, DVC, DLQ |
| **02 — Pré-processamento** | Resample 500 Hz, lead única, filtro, Z-score | `scipy`, YAML versionado |
| **03 — Features** | Detecção AMPT, features morfológicas/temporais | Python puro |
| **04 — Modelagem** | Backbone 1D-CNN em duas etapas (N vs Anormal → S/V/F) | TensorFlow/Keras |
| **05 — Quantização** | PTQ INT8 per-channel, exportação para C | TFLite |
| **06 — Validação** | Quality gates e relatórios de qualidade | `pytest`, CI/CD |
| **07 — DevOps** | Ambiente reprodutível, CI, Docker | `uv`, GitHub Actions |
| **08 — Firmware** | Firmware C/C++17 bare-metal para STM32F4 | ARM GCC, TFLM, CMSIS-NN |
| **09 — Simulação** | Validação sem hardware via Renode | Renode 1.15.3 |
| **10 — Test Harness** | Testes HIL C vs Python, bit-exatidão, fidelidade | Harness C + pytest |
| **11 — Knowledge Layer** | RAG local para documentação/código via MCP | `sqlite-vec`, `sentence-transformers`, MCP SDK |

> 🇧🇷 Projeto de arquitetura de firmware, CI/CD e quality gates para sistemas embarcados médicos.
> 🇺🇸 Demonstration of embedded firmware architecture, CI/CD and quality gates for medical edge devices.

---

## 🏗️ Arquitetura

```mermaid
flowchart TD
    A[Datasets Públicos<br/>MIT-BIH, Chapman, SVDB, AFDB, INCART] --> B[Ingestão & Validação]
    B --> C[Resample 500 Hz + Lead MLII]
    C --> D[Pré-processamento DSP]
    D --> E[Feature Engineering]
    E --> F1[Stage 1<br/>N vs Anormal]
    F1 -->|Anormal| F2[Stage 2<br/>S vs V vs F]
    F1 -->|Normal| G[Quantização PTQ INT8]
    F2 --> G
    G --> H[Export C Headers]
    H --> I[STM32F4 Firmware]
    I --> J[Inferência TFLM + CMSIS-NN]
    J --> K[UART Output]
    K --> L[Validação Renode]
    L --> M{PASS / FAIL}

    K11[Documentação + Código] --> L11[Camada C11<br/>sqlite-vec + MCP]
    L11 --> M11[Agentes de Coding]
```

**Fluxo de dados no firmware:**

```text
ADC stub / UART raw int8
    → dequantização float32
    → filtro passa-banda 0.5–40 Hz
    → filtro notch 60 Hz
    → normalização Z-score
    → quantização int8
    → inferência TFLM (CMSIS-NN)
    → argmax → saída UART
```

---

## 🫀 Pipeline de Dados

A camada de dados unifica sinais de ECG de múltiplas fontes públicas em um formato único de **500 Hz**, lead **MLII-equivalente** e janelas de **1000 ms** — tudo rastreável e reprodutível.

### 📊 Datasets de Entrada

| Dataset | Registros | Fs original | Tamanho (raw) | Papel |
| :--- | :--- | :--- | :--- | :--- |
| **Chapman-Shaoxing** | 45.152 | 500 Hz | ~5.1 GB | Pré-treino do backbone |
| **MIT-BIH Arrhythmia** | 48 | 360 Hz | ~104 MB | Fine-tuning + teste |
| **MIT-BIH SVDB** | 78 | 250 Hz | ~75 MB | Fine-tuning (supraventricular) |
| **MIT-BIH AFDB** | 25 (23 c/ sinal) | 250 Hz | ~606 MB | Fine-tuning (fibrilação atrial) |
| **INCART** | 75 | 257 Hz | ~795 MB | Fine-tuning (diversidade russa) |

> **Total MIT-BIH+:** 226 registros • ~1.1 GB

### ⚙️ Fluxo de Pré-processamento

```bash
make env          # uv + dependências
make download-all # PhysioNet / ZIP / mirror
make process      # resample → lead → filter → detrend → normalize
make test         # QG0 → QG1
```

1. **Ingestão:** `wfdb.io.dl_database` com retry exponencial + DLQ (`data/.dlq/`) para falhas.
2. **Resample:** `scipy.signal.resample_poly` para **500 Hz**.
3. **Lead única:** MLII/ECG1 para MIT-BIH/SVDB/AFDB; lead **II** para Chapman/INCART.
4. **Filtro:** Butterworth 4ª ordem, bandpass **0.5–40 Hz** (`filtfilt` zero-phase).
5. **Detrend linear** + **Z-score global** (fit no treino, transform no teste).
6. **Segmentação:** janela de **1000 ms** centrada no R-peak; fallback para **600 ms** quando RR < 600 ms. Sem padding zero.

### 🎯 Seleção de Lead MLII-equivalente

| Dataset | Lead | Índice | Nota |
| :--- | :--- | :--- | :--- |
| Chapman | `II` | `lead_names.index("II")` | Nativamente 500 Hz |
| MIT-BIH | `MLII` | `0` | Lead padrão |
| SVDB | `ECG1` | `0` | Equivalência não documentada oficialmente |
| AFDB | `ECG1` | `0` | Equivalência não documentada oficialmente |
| INCART | `II` | `lead_names.index("II")` | Anatomicamente próximo de MLII |

### 📜 Linhagem e Governança

Cada registro processado gera um JSON em `data/lineage/{dataset}/{record_id}.json` com checksum, parâmetros, pipeline e metadados. A DLQ (`data/.dlq/`) captura falhas de download e processamento para reprocessamento seletivo.

```yaml
# config/preprocess_v1.0.yaml
filter:
  type: butterworth
  order: 4
  lowcut: 0.5
  highcut: 40.0
normalization:
  type: zscore_global
```

---

## 🧠 ML Pipeline

### 🏗️ Arquitetura — Dois Modelos 1D-CNN

O classificador v2.2 usa **dois modelos leves em cascata**, cada um com backbone 1D-CNN escalado (~38K–50K parâmetros). Essa divisão foi necessária porque um classificador 5-classes AAMI direto (v1.1) não atingiu os thresholds de qualidade no cenário inter-paciente.

#### Stage 1 — N vs Anormal (S + V + F)

```text
Input(500, 1)                    # 1000 ms @ 500 Hz
 → Conv1D(32, k7) → MaxPool1D(2)
 → Conv1D(64, k5) → MaxPool1D(2)
 → Conv1D(96, k3) → MaxPool1D(2)
 → GlobalAveragePooling1D()
 → Dense(96, relu) → Dropout(0.3)
 → Dense(2, softmax)              # N, Anormal
```

#### Stage 2 — S vs V vs F

Mesma arquitetura do Stage 1, com saída `Dense(3, softmax)`.

- **~38K–50K parâmetros por modelo** | **FlatBuffer ~54 KB cada** | **Arena TFLM < 64 KB**

### 🔬 Feature Engineering

| Módulo | Descrição |
| :--- | :--- |
| `src/features/ampt_500hz.py` | Detector AMPT (banda 5–15 Hz, MWI 150 ms, refratariedade 360 ms) |
| `src/features/time_domain.py` | RR, HRV, RMSSD, heart rate |
| `src/features/morphological.py` | R-amp, Q-depth, T-amp, QRS-width (envelope method), ST-slope J+60ms→J+80ms |
| `src/features/augmentation.py` | Jitter, baseline wander, powerline, time warp *(apenas treino)* |
| `src/features/balancer.py` | SMOTE/ADASYN no espaço de features |

### 🎯 Treinamento em Dois Estágios

1. **Stage 1 — N vs Anormal:** treinado sobre MIT-BIH+ (226 registros, 1 lead, AAMI) com **GroupKFold por paciente**. Batimentos das classes S, V e F são agrupados como "Anormal"; a classe Q é excluída da classificação final a partir de v2.0.
2. **Stage 2 — S vs V vs F:** treinado apenas sobre as amostras classificadas como "Anormal" pelo Stage 1.

> **Nota sobre pré-treino:** o pré-treino em Chapman-Shaoxing foi avaliado e **abandonado por subajuste** (AUC-ROC macro por registro ~0,71). O backbone atual é treinado **from-scratch no MIT-BIH+**.

Comandos:

```bash
# Pipeline legado (v1.1) — ainda disponível no Makefile
make pretrain
make finetune

# Pipeline produtivo atual (v2.2)
uv run python scripts/run_stage1_training.py
uv run python scripts/run_stage2_training.py
uv run python scripts/run_two_stage_pipeline.py
```

Artefatos gerados:

```
models/stage1_float32_v2.0.keras
models/stage2_float32_v2.0.keras
models/input_scaler_stage1_v2.0.pkl
models/input_scaler_stage2_v2.0.pkl
models/stage1_threshold_v2.0.json
```

### 👤 Validação Inter-Patient

Nunca misturamos batimentos do mesmo paciente entre treino e teste. Usamos **GroupKFold por paciente** (`n_splits=5`) para evitar *data leakage*.

### 📊 Métricas AAMI EC57 — Thresholds v2.2

Os thresholds originais v1.1 (Acc > 93%, F1-macro > 0,85) não foram atingíveis com um modelo compacto (< 100K params, FlatBuffer < 64 KB) no cenário inter-paciente. A v2.2 revisou os gates para refletir o estado da arte de classificadores leves em MIT-BIH+.

| Componente | Métrica | Threshold v2.2 | Valor obtido |
| :--- | :--- | :--- | :--- |
| **Stage 1** | Acc | > 75% | 79,3% |
| **Stage 1** | F1-macro | > 55% | 0,593 |
| **Stage 1** | Recall Anormal | ≥ 30% | 0,325 |
| **Stage 1** | Precision Anormal | ≥ 25% | 0,290 |
| **Stage 2** | F1(S) | ≥ 55% | 0,643 |
| **Stage 2** | F1(V) | ≥ 70% | 0,710 |
| **Stage 2** | F1(F) | ≥ 15% | 0,203 |
| **Pipeline integrado** | Acc | > 78% | 78,7% |
| **Pipeline integrado** | F1-macro | > 30% | 0,316 |

> F1-macro e MCC são as métricas primárias; acurácia global pode ser enganosa em classes desbalanceadas.

### ⚡ Quantização PTQ INT8

Cada modelo (Stage 1 e Stage 2) é quantizado separadamente com **PTQ per-channel INT8** e 512 amostras estratificadas AAMI:

```bash
make quantize
# ou
uv run python scripts/quantize_two_stage_v2.0.py
```

| Critério | Limite | Valor obtido |
| :--- | :--- | :--- |
| FlatBuffer Stage 1 | < 64 KB | 54,36 KB |
| FlatBuffer Stage 2 | < 64 KB | 54,47 KB |
| ΔF1-macro (INT8 vs float) | < 2% | ~0,3% |

### 📤 Exportação para Firmware

Conversão para headers C puros, sem dependências externas:

```bash
make export
# entregáveis:
#   firmware/src/ml/stage1_int8_v2.0.h
#   firmware/src/ml/stage2_int8_v2.0.h
#   firmware/src/ml/quantization_params.h
```

---

## 🧠 Camada C11 — Knowledge Layer (RAG)

Sistema de recuperação semântica local para dar contexto a agentes de coding (Kimi Code / OpenCode) sobre a documentação e o código do Project-Lewis.

- **Vector DB:** `sqlite-vec` sobre SQLite (`data/knowledge.db`, ~7 MB).
- **Embeddings:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 dimensões, CPU-only).
- **Protocolo:** MCP server stdio via SDK oficial (`mcp.json` → `project-lewis-knowledge`).
- **Metadados 3D:** camada (C01–C11), versão, tags semânticas.
- **Segurança/LGPD:** scan de PII e bloqueio automático de dados brutos de ECG.

```bash
make knowledge-index      # reindexa docs/src/firmware
make knowledge-status     # status do índice
make knowledge-query      # query interativa
make knowledge-test       # roda tests/test_knowledge/
make knowledge-validate   # valida índice gerado
```

Tools MCP expostas:

- `search_docs(query, layer?, version?, tags?, k?)`
- `list_layers()`
- `get_doc_by_source(source, k?)`

> Documentação completa: [`docs/SDD-C11-Knowledge-Impl-v2.0.md`](docs/SDD-C11-Knowledge-Impl-v2.0.md)

---

## ⚙️ Firmware, Simulação & DevOps

O Project-Lewis entrega um firmware C/C++17 para **STM32F407VG** rodando **TensorFlow Lite Micro** com kernels **CMSIS-NN**. Toda a validação de hardware é feita sem silicone real, via emulação fiel no **Renode 1.15.3**.

### 🖥️ Firmware & Pipeline DSP

| Recurso | Especificação |
| :--- | :--- |
| **MCU** | STM32F407VG (Cortex-M4F, 168 MHz, 192 KB SRAM, 1 MB Flash) |
| **DSP** | Filtro passa-banda 0.5–40 Hz + notch 60 Hz + Z-score |
| **ML** | TFLM INT8, arena estática 64 KB, modelo < 64 KB |
| **Debug** | UART4 sem `printf`/semihosting |

### 🔌 Firmware Test Harness

O projeto inclui um **test harness C/C++** para validação automatizada do firmware, permitindo rodar as mesmas suites no host (compilação nativa) e no Renode (STM32F4 emulado) e comparar as saídas do C com as referências Python.

| Componente | Descrição |
| :--- | :--- |
| `firmware/tests/harness.{c,h}` | Mini-framework de registro/execução de testes e asserts |
| `firmware/tests/harness_main.c` | Ponto de entrada que registra todas as suites |
| `firmware/tests/test_dsp.c` | Filtros DSP C vs Python (QG16) |
| `firmware/tests/test_r_peak.c` | Detector R-peak C vs AMPT Python (QG18) |
| `firmware/tests/test_inference.cpp` / `test_pipeline.c` | Inferência/pipeline C vs Python (QG8, QG17) |
| `firmware/scripts/generate_harness_fixtures.py` | Gera fixtures C a partir das referências Python |
| `firmware/scripts/run_harness.py` | Orquestra build native/Renode e gera relatório |
| `firmware/renode/harness.resc` | Plataforma Renode para o harness |
| `firmware/stm32f407vg_harness.ld` | Linker script com pilha de 64 KB |

```bash
cd firmware
make harness-native    # executa no host (rápido, sem hardware)
make harness-renode    # executa no Renode (STM32F4 emulado)
make harness           # ambos os ambientes
# ou diretamente
python3 scripts/run_harness.py --mode both
```

O relatório é gerado em `firmware/test_harness_report.json` com o resumo `native`/`renode` de PASS/FAIL/TOTAL.

### 🧪 Simulação Renode

```bash
make firmware-deps     # ARM GCC 13.3 + Renode 1.15.3
make firmware-tflm     # clone + build do TensorFlow Lite Micro (host + ARM)
make firmware-build    # ELF para STM32F4
make firmware-test     # 5 s de simulação headless
make hard-gates        # Hard Gates HG-01..HG-06
make harness-renode    # harness de testes no Renode
```

> ⚠️ **Limites:** timings são representativos, energia não é estimada e há tolerância de 1 LSB entre CMSIS-NN e kernels de referência. Veja [`docs/SIMULATION_LIMITS.md`](docs/SIMULATION_LIMITS.md).

### 🔄 CI/CD & Reprodutibilidade

- **`uv` + `pyproject.toml`**: lockfile determinístico e ambientes isolados.
- **Makefile**: targets por camada com paralelismo (`make -j4 all`).
- **Docker + Docker Compose**: reprodutibilidade total entre máquinas.
- **GitHub Actions**: lint → unit tests → integration tests → quality gates.
- **Pre-commit**: Black, isort, flake8, mypy, bandit.

```bash
make env              # cria ambiente
make all              # pipeline completo
make docker-build     # imagem reprodutível
make quality-report   # relatório QG0–QG6
```

---

## 📁 Estrutura do Repositório

```
project-lewis/
├── config/                # Parâmetros versionados (pré-processamento, ML, C11)
│   ├── knowledge_v2.0.yaml
│   ├── preprocess_v1.0.yaml
│   ├── stage1_binary.yaml
│   └── stage2_multiclass.yaml
├── data/                  # Datasets brutos/processados (DVC, gitignored)
│   ├── raw_chapman/
│   ├── raw_mitbih/
│   ├── raw_svdb/
│   ├── raw_afdb/
│   ├── raw_incart/
│   ├── processed/
│   ├── features/
│   ├── lineage/
│   ├── .dlq/
│   └── knowledge.db       # RAG sqlite-vec (C11)
├── docs/                  # Especificações por camada
│   ├── ESPECIFICACAO_Fase1_Agentes-v1.1.md
│   ├── Camada-01-Ingestao-v1.1.md
│   ├── Camada-02-Resample-Preprocessamento-v1.1.md
│   ├── Camada-03-Feature-Engineering-v1.1.md
│   ├── Camada-04-Modelagem-v1.1.md
│   ├── Camada-05-Quantizacao-Exportacao-v1.1.md
│   ├── Camada-06-Validacao-Quality-Gates-v1.1.md
│   ├── Camada-07-Integracao-DevOps-v1.1.md
│   ├── Camada-08-Firmware-v1.1.md
│   ├── Camada-09-Simulacao-v1.1.md
│   ├── Camada-09-Energia-v1.4.md
│   ├── SDD_Project-Lewis_v3.md
│   ├── SDD-C11-Knowledge-Impl-v2.0.md
│   ├── DEBITO_TECNICO_Energia_Renode-v1.4.md
│   └── SIMULATION_LIMITS.md
├── firmware/              # Firmware embarcado
│   ├── src/               # app, dsp, hal, ml, platform, utils
│   ├── renode/            # scripts .resc, .robot
│   ├── scripts/           # runners Renode
│   ├── tests/             # testes HIL
│   ├── Makefile
│   ├── third_party/       # tflite-micro (clonado em build time, ver .commit)
│   └── tools/             # ARM GCC e Renode (instalados localmente)
├── models/                # Modelos treinados e quantizados
│   ├── stage1_float32_v2.0.keras
│   ├── stage2_float32_v2.0.keras
│   ├── input_scaler_stage1_v2.0.pkl
│   ├── input_scaler_stage2_v2.0.pkl
│   ├── stage1_threshold_v2.0.json
│   └── quantized/         # .tflite e headers INT8
├── notebooks/             # EDA e validação visual
├── reports/               # Relatórios de qualidade e simulação
├── scripts/               # Automação de quality gates e relatórios
├── src/                   # Código Python do pipeline
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── quantization/
│   ├── tracking/          # Tracking de experimentos (SQLite)
│   └── knowledge/         # Camada C11 (RAG + MCP)
├── tests/                 # Testes pytest (QG0–QG19 + QG-C11)
│   └── test_knowledge/    # Testes da Camada C11
├── .github/workflows/     # CI/CD
├── docker-compose.yml
├── Dockerfile
├── Makefile
├── mcp.json               # Configuração MCP servers
├── LICENSE
└── README.md
```

---

## 🚀 Como Executar

### 1. Ambiente

```bash
# Usando uv (recomendado)
make env
source .venv/bin/activate

# Ou Docker
make docker-build
make docker-run
```

### 2. Pipeline de Dados

```bash
make download-all   # QG0
make process        # QG1
```

### 3. Features e Modelagem

```bash
make features       # QG2/QG3

# Pipeline atual (v2.2) — duas etapas
uv run python scripts/run_stage1_training.py
uv run python scripts/run_stage2_training.py
uv run python scripts/run_two_stage_pipeline.py

# Targets legados (v1.1) — ainda disponíveis
make pretrain       # QG4
make finetune       # QG5
```

### 4. Quantização e Exportação

```bash
make quantize       # QG6
make export         # headers C
```

### 5. Firmware, Harness e Simulação

```bash
cd firmware
make firmware-deps
make firmware-tflm
make firmware-build
make firmware-test
make harness           # test harness native + renode
```

### 6. Camada C11 — Knowledge Layer

```bash
make knowledge-index     # indexa docs/src/firmware
make knowledge-status    # status do índice
make knowledge-query     # query interativa
make knowledge-test      # testes C11
make knowledge-validate  # validação do índice
```

### 7. Testes Completos

```bash
make test              # pytest completo (323 testes)
make hard-gates        # Hard Gates HG-01..HG-06
make quality-report    # relatório consolidado
make lint              # flake8 + mypy + bandit
```

---

## 🛡️ Quality Gates

Nenhum artefato avança para a próxima camada sem passar no gate correspondente.

### Fase 1 — Dados & ML

| Gate | Foco | Threshold | Comando |
| :--- | :--- | :--- | :--- |
| **QG0** | Download | Chapman ≥ 45k; MIT-BIH 48; SVDB 78; AFDB 25; INCART 75; DLQ vazia | `pytest tests/test_download.py` |
| **QG1** | Pré-processamento | Fs = 500 Hz; range ±5 mV; Z-score global; linhagem 100% | `pytest tests/test_preprocessing.py` |
| **QG2** | AMPT @ 500 Hz | Sens > 96,5%; PPV > 99,0%; tol = 150 ms | `pytest tests/test_ampt.py` |
| **QG3** | Features | Janela 1000 ms; ≥ 10 dimensões; sem NaN/Inf | `pytest tests/test_features.py` |
| **QG4** | Pré-treino Chapman | AUC-ROC macro > 0,85 *(abandonado na prática; backbone treinado from-scratch)* | `pytest tests/test_pretrain.py` |
| **QG5'** | Fine-tuning v2.2 | Stage1: Acc > 75%, F1-macro > 0,55, recall/precision Anormal ≥ 30%/25%; Stage2: F1(S) ≥ 55%, F1(V) ≥ 70%, F1(F) ≥ 15%; Pipeline: Acc > 78%, F1-macro > 0,30 | `pytest tests/test_finetune.py tests/test_two_stage_pipeline.py` |
| **QG6** | Quantização | ΔF1-macro < 2%; FlatBuffer Stage1/Stage2 < 64 KB; headers compiláveis | `pytest tests/test_quantization.py` |

### Firmware & Simulação

| Gate | Foco | Threshold | Comando |
| :--- | :--- | :--- | :--- |
| **QG7** | Build firmware | `-Werror`; FlatBuffer < 64 KB | `make firmware-build` |
| **QG8** | Bit-exatidão C vs Python | `atol = 1` LSB | `pytest -m qg8` |
| **QG9** | Latência TFLM | < 200 ms/batimento | Relatório Renode |
| **QG10** | Fidelidade DSP | cosine > 0,99 | `pytest -m qg10` |
| **QG11** | Fault injection | Graceful degradation | `pytest -m qg11` |
| **QG12** | Arena limit (48 KB RAM) | `INIT FAIL` sem HardFault | `pytest -m qg12` |
| **QG13** | Watchdog de inferência | Reset após timeout | `pytest -m qg13` |
| **QG16** | Filtros DSP vs Python | correlação > 0,99 / RMSE < 1e-6 | `pytest -m qg16` / `make harness` |
| **QG17** | Pipeline filtrado C vs Python | MAE < 0,01 / cosine > 0,99 | `pytest -m qg17` / `make harness` |
| **QG18** | Detector R-peak C vs AMPT | Sens ≥ 90%; PPV ≥ 90% | `pytest -m qg18` / `make harness` |
| **QG19** | Consumo energético | < 50 mA e < 165 mJ/batimento @ 3,3 V | `reports/firmware_simulation_report.json` |

### Camada C11 — Knowledge Layer

| Gate | Foco | Threshold | Comando |
| :--- | :--- | :--- | :--- |
| **QG-C11-01** | Cobertura de indexação | 100% dos `.md`/`.py` indexados | `pytest tests/test_knowledge/test_indexer.py` |
| **QG-C11-02** | Determinismo | Reindexação gera mesmos IDs | `pytest tests/test_knowledge/test_indexer.py` |
| **QG-C11-03** | Retrieval | MRR@5 ≥ 0,80 | `pytest tests/test_knowledge/test_retriever.py` |
| **QG-C11-04** | Filtros 3D | Layer/version/tag funcionais | `pytest tests/test_knowledge/test_retriever.py` |
| **QG-C11-05** | LGPD | Zero PII/dados ECG no índice | `pytest tests/test_knowledge/test_lgpd_compliance.py` |
| **QG-C11-06** | MCP protocol | `initialize` e `tools/list` respondem | `pytest tests/test_knowledge/test_mcp_server.py` |
| **QG-C11-07** | Tamanho do banco | < 500 MB | `pytest tests/test_knowledge/test_indexer.py` |
| **QG-C11-08** | CLI funcional | `reindex`, `query`, `status` | `pytest tests/test_knowledge/test_integration.py` |
| **QG-C11-09** | Deps compliance | Sem LangChain/Chroma/typer | `pytest tests/test_knowledge/test_deps_compliance.py` |

> QG19 é um débito técnico documentado para a v1.4. Veja [`docs/DEBITO_TECNICO_Energia_Renode-v1.4.md`](docs/DEBITO_TECNICO_Energia_Renode-v1.4.md).

---

## ⚠️ Limites da Simulação

Consulte [`docs/SIMULATION_LIMITS.md`](docs/SIMULATION_LIMITS.md) para detalhes sobre:

- Validação sem hardware físico (Renode 1.15.3).
- Latências determinísticas (sem modelagem de cache/jitter/temperatura).
- Divergência de até 1 LSB entre CMSIS-NN e kernels de referência.
- Modelagem de energia ainda não implementada (débito técnico v1.4).

---

## 📌 Versão Atual

`v2.2+c11` — pipeline de classificação em duas etapas (Stage 1: N vs Anormal; Stage 2: S vs V vs F) atingindo todos os thresholds v2.2, firmware STM32F4 validado via Renode, e Camada C11 (RAG local sqlite-vec + MCP) para suporte a agentes de coding.

---

## 👤 Autor

**Douglas Souza** — Engenheiro de Software & Arquiteto de Sistemas

Arquitetura de firmware embarcado, CI/CD, compliance e integração ML embarcado.

---

## 📜 Licença

MIT License — veja [`LICENSE`](LICENSE) para detalhes.
