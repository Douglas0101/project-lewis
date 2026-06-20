# Especificação Arquitetural — Fase 1: Pipeline de Dados + Ciência de Dados
## Para Agentes de Coding (OpenCode / Kimi Code / MCP)

**Versão:** 1.1 | **Data:** 2026-06-09 | **Arquiteto:** Douglas Souza
**Hardware Dev:** Lenovo IdeaPad 3 15ITL6 | Zorin OS 18.1 (Ubuntu 24.04.3 LTS)
**Python:** 3.12.x (system Python do SO — não usar 3.13+)
**Escopo:** Do MIT-BIH bruto até `model_data.h` pronto para TFLM.

---

## 1. Diretrizes para Agentes de Coding

### 1.1 Regra de Ouro
**Não inventar stacks. Não usar hype.** Cada dependência deve ser justificável com: (a) versão LTS/estável, (b) compatibilidade com Ubuntu 24.04, (c) necessidade para o pipeline médico.

### 1.2 Isolamento de Ambiente
- **Obrigatório:** `uv` (Astral) como gerenciador de dependências — nunca instalar no system Python.
- **Lockfile:** `uv.lock` nativo + `pyproject.toml` (PEP 621). Fallback: `pip-tools` com hash pinning.
- **Container:** `Dockerfile` + `docker-compose.yml` para reprodutibilidade total.
- **CI:** GitHub Actions com `ubuntu-24.04` runner, cache externo S3/GCS para datasets > 1GB.
- **Pre-commit:** Black, isort, flake8, mypy, bandit obrigatórios antes de cada commit.

### 1.3 Versionamento de Dados (DVC)
- **Obrigatório:** DVC (Data Version Control) para rastrear datasets grandes.
- `.dvc` files versionados no Git; binários em remote S3/GCS.
- `dvc pull` garante reprodutibilidade em qualquer máquina.
- `data/raw_*/`, `data/processed/`, `data/mirrors/` no `.gitignore`.

---

## 2. Estratégia de Dados em Escala — GB para Precisão de Produção

**Alerta:** O MIT-BIH Arrhythmia Database tem 48 registros (~109.000 batimentos anotados, ~73.5 MB zip / ~104.3 MB raw). Isso é insuficiente para treinar modelos de deep learning com precisão de produção. A literatura de ECG foundation models (2024–2026) demonstra que dados na escala de GB são necessários para convergência robusta.

### 2.1 Datasets Complementares Públicos (Escala GB) — CORRIGIDO

| Dataset | Registros | Pacientes | Tamanho (Zip) | Tamanho (Raw) | Leads | Fs | Duração | Acesso | Anotações | Uso no Projeto |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Chapman-Shaoxing** | 45.152 | 34.905 | ~2.3 GB | ~5.1 GB | 12 | 500 Hz | 10 s | Público | 5 superclasses SCP-ECG | **Pré-treino backbone** |
| **MIT-BIH Arrhythmia** | 48 | 47 | ~73.5 MB | ~104.3 MB | 2 | 360 Hz | 30 min | Público | Beat-level AAMI | **Fine-tuning + teste** |
| **MIT-BIH SVDB** | 78 | 78 | ~52.0 MB | ~75 MB | 2 | **250 Hz** | 30 min | Público | Beat-level AAMI | **Fine-tuning** (supraventricular) |
| **MIT-BIH AFDB** | 25 (23 c/ sinais) | 25 | ~439.7 MB | ~605.9 MB | 2 | 250 Hz | 10–24 h | Público | Rhythm + beat | **Fine-tuning** (fibrilação atrial) |
| **INCART** | 75 | 32 | ~563.5 MB | ~794.5 MB | 12 | 257 Hz | 30 min | Público | Beat-level | **Fine-tuning** (diversidade russa) |
| **PTB-XL** | 21.837 | 18.869 | ~1.5 GB | ~3.0 GB | 12 | 500 Hz | 10 s | Público | 5 superclasses SCP-ECG | **Pré-treino** (backbone alternativo) |
| **AHA Database** | 154 | — | ~200 MB | ~400 MB | 2 | 250 Hz | 2–24 h | Restrito* | Beat-level | **Treino** (volume massivo, opcional) |

*AHA requer solicitação de acesso; não bloquear pipeline esperando AHA.

> **Correções do v1.0:** Tamanhos do MIT-BIH, SVDB, AFDB e INCART estavam drasticamente subestimados. Os valores acima são oficiais da PhysioNet v1.0.0. SVDB é **250 Hz** (frequência corrigida em relação à v1.0). AFDB tem 23 registros com sinais `.dat` (2 são anotações-only: 00735 e 03665).

### 2.2 Estratégia Híbrida Recomendada — 3 Estágios

```
Stage 1: Pré-treino (Supervised Multi-Label)
├── Datasets: Chapman-Shaoxing (5.1 GB raw) + PTB-XL (3.0 GB raw, fallback)
├── Formato: 12-lead, 10s, 500 Hz, 5 superclasses SCP-ECG (NORM, CD, MI, HYP, STTC)
├── Objetivo: Aprender representações de ECG robustas (backbone 1D-CNN)
├── Output: Pesos pré-treinados (backbone_1d_pretrained_v1.0.keras)
└── Nota: Pular se tempo de TCC não permitir. Opcional mas recomendado.

Stage 2: Fine-tuning Supervised (Arritmias Beat-Level)
├── Datasets: MIT-BIH (48) + SVDB (78) + AFDB (25) + INCART (75) = 226 registros
├── Formato: 1-lead MLII-equivalente, 500 Hz (resample), beat-level AAMI
├── Objetivo: Classificação N/V/S/F/Q por batimento
├── Técnica: Transfer learning — congelar backbone, retreinar classifier
├── Split: GroupKFold por paciente (n_splits=5, inter-patient)
├── Data Augmentation: Jitter + baseline wander + powerline noise (apenas treino)
├── Balanceamento: SMOTE no espaço de features (nunca no sinal bruto)
└── Output: Modelo float32 treinado (finetuned_float32_v1.0.keras)

Stage 3: Quantização e Exportação
├── Dataset representativo: 512–1024 amostras estratificadas do Stage 2
├── Técnica: PTQ INT8 per-channel (TFLite)
├── Output: model_data.h + quantization_params.h para TFLM
└── Validação: ΔF1-macro < 2% vs float32, FlatBuffer < 64KB
```

### 2.3 Data Augmentation Obrigatória (MIT-BIH é pequeno)

Como o conjunto de arritmias beat-level é pequeno (~109k–500k batimentos), augmentation é **não opcional**:

| Técnica | Implementação | Parâmetros | Classes Beneficiadas | Risco |
| :--- | :--- | :--- | :--- | :--- |
| **Amplitude Jitter** | `x += np.random.normal(0, 0.01*std, size)` | σ = 1% do sinal | Todas | Baixo |
| **Baseline Wander** | Adicionar senoide 0.05–0.5 Hz | amp < 0.2 mV | Todas | Baixo (simula respiração) |
| **Powerline Noise** | Adicionar 50/60 Hz + harmônicos | amp < 0.05 mV | Todas | Baixo (simula rede) |
| **Time Warping** | `scipy.interpolate` stretch 0.95–1.05× | ±5% | Todas | Médio (distorce QRS) |
| **SMOTE** | `imbalanced-learn.SMOTE` | k_neighbors=5, no **feature space** | V, S, F, Q (minoritárias) | Médio (sintético) |
| **Mixup** | `lambda*x1 + (1-lambda)*x2` | lambda ~ Beta(0.2, 0.2) | Todas | Médio |

> **Recomendação para TCC:** Implementar jitter + baseline wander + powerline + SMOTE no feature space. GANs/Diffusion são overkill e difíceis de validar em TCC.

> **Regra:** Augmentation aplicável **APENAS** no fine-tuning (MIT-BIH+). **NUNCA** no pré-treino (Chapman) ou teste.

### 2.4 Decisão Arquitetural — Escopo de Dados (CONFIRMADO: OPÇÃO E)

**Decisão do arquiteto (2026-06-09):** Opção E — Pré-treino em Chapman-Shaoxing (~5.1 GB raw) + fine-tuning em agregado MIT-BIH+ (226 registros, ~1.1 GB raw total).

**Motivo:** Chapman oferece diversidade demográfica (população asiática), volume massivo (45k registros), e frequência nativa 500 Hz (alinha com decisão de resample). MIT-BIH+ fornece ground truth beat-level AAMI para fine-tuning especializado em arritmias.

**Pipeline de dados confirmado:**
```
Chapman-Shaoxing (5.1 GB raw, 12-lead, 500Hz, 10s, 5 SCP-ECG superclasses)
    → Pré-treino backbone 1D-CNN (supervised multi-label, sigmoid)
    → Pesos pré-treinados (backbone_pretrained_v1.0.keras)

MIT-BIH (48) + SVDB (78) + AFDB (25) + INCART (75)
    → Resample 360/250/257 Hz → 500 Hz (scipy.signal.resample_poly, padtype="line")
    → Seleção lead II (Chapman/INCART) ou MLII/ECG1 (MIT-BIH/SVDB/AFDB)
    → Segmentação 1000ms = 500 amostras @ 500Hz (fallback 600ms para RR < 600ms)
    → Data augmentation (jitter, baseline wander, powerline — apenas treino)
    → Fine-tuning do backbone (congelar convs, retreinar classifier)
    → GroupKFold por paciente (n_splits=5, inter-patient)
    → Modelo float32 validado (F1-macro > 85%, MCC > 0.80, inter-patient)
    → PTQ INT8 per-channel com 512 amostras estratificadas
    → Exportação model_data.h (< 64KB FlatBuffer) + quantization_params.h
```

**Estrutura de diretórios atualizada para múltiplos datasets:**
```
data/
├── raw_chapman/          # 45k registros, ~5.1 GB — NÃO versionar (DVC)
├── raw_mitbih/           # 48 registros, ~104.3 MB
├── raw_svdb/             # 78 registros, ~75 MB
├── raw_afdb/             # 25 registros, ~605.9 MB
├── raw_incart/           # 75 registros, ~794.5 MB
├── processed/            # Resampled 500Hz, lead unificado, mV — NÃO versionar
│   ├── chapman/          # Pré-treino
│   ├── mitbih_500hz/     # Fine-tuning
│   ├── svdb_500hz/
│   ├── afdb_500hz/
│   └── incart_500hz/
├── features/             # NÃO versionar
│   ├── chapman_scp/      # Labels SCP-ECG para pré-treino
│   └── mitbih_plus_aami/ # Labels AAMI para fine-tuning
├── lineage/              # JSON de rastreabilidade por registro/batimento
├── catalog/              # Metadados extraídos dos .hea
├── .dlq/                 # Dead Letter Queue (falhas de download/processamento)
└── mirrors/              # Tarballs de backup — NÃO versionar
    ├── chapman_mirror.tar.gz
    ├── mitbih_family_mirror.tar.gz
    └── ...
```

---

## 3. Estrutura de Diretórios (Contrato)

Os agentes devem criar EXATAMENTE esta estrutura. Não flat, não misturar.

```
Project-Lewis/
├── data/                   # Dados (DVC, não versionar no Git)
│   ├── raw_chapman/
│   ├── raw_mitbih/
│   ├── raw_svdb/
│   ├── raw_afdb/
│   ├── raw_incart/
│   ├── processed/
│   ├── features/
│   ├── lineage/
│   ├── catalog/
│   ├── .dlq/
│   └── mirrors/
├── notebooks/              # EDA + validação visual (opcional, não bloqueante)
├── src/
│   ├── data/
│   │   ├── download_chapman.py    # Kaggle API / Figshare / wget
│   │   ├── download_mitbih.py     # wfdb.io.dl_database + mirror
│   │   ├── loader.py              # MITBIHLoader: ganho/baseline do .hea
│   │   ├── resampler.py           # scipy.signal.resample_poly para 500 Hz
│   │   ├── lead_selector.py       # Seleção de lead MLII-equivalente
│   │   ├── preprocessor.py        # Butterworth filtfilt 0.5–40Hz, z-score global
│   │   └── segmenter.py           # Janelas 1000ms (fallback 600ms), sem padding
│   ├── features/
│   │   ├── ampt_500hz.py          # AMPT detector — referência Python
│   │   ├── time_domain.py         # RR, HRV, RMSSD, heart_rate
│   │   ├── morphological.py       # R-amp, Q-depth, T-amp, QRS-width, ST-slope
│   │   ├── augmentation.py        # Jitter, baseline wander, powerline, time warp
│   │   ├── balancer.py            # SMOTE / ADASYN no feature space
│   │   └── aami_mapper.py         # Mapeamento WFDB → AAMI EC57
│   ├── models/
│   │   ├── backbone_1d.py         # Arquitetura 1D-CNN (input 500 amostras)
│   │   ├── pretrain_chapman.py    # Pré-treino multi-label SCP-ECG
│   │   ├── finetune_mitbih.py     # Fine-tuning com backbone congelado
│   │   ├── train.py               # GroupKFold por paciente
│   │   └── evaluate.py            # Métricas AAMI EC57 (Se, PPV, FPR, F1, MCC)
│   └── quantization/
│       ├── ptq.py                 # PTQ INT8 per-channel
│       ├── extract_params.py      # Extração de scale/zero_point para firmware
│       └── export_tflite.py       # Exportação para header C (Python nativo)
├── tests/
│   ├── conftest.py              # Fixtures compartilhadas (pytest)
│   ├── test_download.py         # QG0: contagem, checksums, DLQ vazia
│   ├── test_loader.py           # QG1: ganho/baseline do .hea, range ±5mV
│   ├── test_resampler.py        # QG1: RMSE < 1e-6, preservação de frequência
│   ├── test_preprocessing.py    # QG1: zero-phase, z-score, linhagem
│   ├── test_ampt.py             # QG2: Sens > 96.5%, PPV > 99.0%, tol=150ms
│   ├── test_segmenter.py        # QG3: janela 1000ms, sem padding zero
│   ├── test_features.py         # QG3: sem NaN/Inf, ranges fisiológicos
│   ├── test_integration.py        # Pipeline end-to-end (3 registros)
│   ├── test_pretrain.py         # QG4: AUC-ROC macro > 0.85
│   ├── test_finetune.py         # QG5: F1-macro > 85%, MCC > 0.80
│   ├── test_quantization.py     # QG6: ΔF1-macro < 2%, FlatBuffer < 64KB
│   └── test_tflm_integration.py # QG6: header compilável, CMSIS-NN
├── scripts/
│   ├── generate_quality_report.py       # Relatório markdown consolidado
│   └── validate_firmware_deliverables.py # Validação de headers C
├── config/
│   ├── preprocess_v1.0.yaml     # Parâmetros de pré-processamento versionados
│   └── pretrain_v1.0.yaml       # Hiperparâmetros de pré-treino
├── firmware/ (placeholder Fase 2)
│   ├── src/ml/
│   ├── src/dsp/
│   ├── src/features/
│   └── tests/hil/
├── pyproject.toml              # Dependências (uv / PEP 621)
├── uv.lock                     # Lockfile determinístico
├── Makefile                    # Orquestração com paralelismo
├── Dockerfile                  # Container reprodutível
├── docker-compose.yml          # Compose para dev
├── .pre-commit-config.yaml     # Hooks obrigatórios
├── .gitignore                  # DVC + dados + ambiente
├── .dvcignore                  # Ignorar processados no DVC
└── README.md
```

---

## 4. Contratos de Interface (APIs Internas)

### 4.1 src.data.loader.MITBIHLoader

```python
class MITBIHLoader:
    """Loader unificado para MIT-BIH family + Chapman.
    NÃO assume ganho/baseline fixos. Sempre lê do .hea via wfdb.rdheader().
    """
    FS_TARGET = 500.0  # Hz PADRÃO APÓS RESAMPLE

    def load_signal(record_path: Path, channel: int = 0, units: str = "physical") -> np.ndarray:
        # Usar wfdb.rdrecord(..., channels=[channel], units=units)
        # Se units="physical", wfdb já converte para mV usando adc_gain e baseline do .hea
        # Validar: se physical, assert range [-5, +5] mV
        # Emitir WARN se adc_gain ou baseline divergirem drasticamente dos padrões MIT-BIH

    def load_annotations(record_path: Path) -> Tuple[np.ndarray, np.ndarray, dict]:
        # Retorna (r_samples, aami_labels, metadata)
        # Mapear símbolos WFDB → AAMI EC57 test labels (N, V, F, Q, S)
        # Filtrar apenas batimentos (beat annotations)
        # Remover: rhythm changes ('+'), signal quality ('~'), non-beat markers
        # metadata: total_beats, noise_segments, paced_ratio

    def get_record_names(raw_dir: Path) -> List[str]:
        # Listar stems de .hea. Esperado >= 226 registros (MIT-BIH family)
```

> **Correção v1.0:** O loader v1.0 hardcodava `GAIN = 200.0` e `BASE = 1024.0`. Isso quebra para INCART e AFDB. O loader v1.1 lê `adc_gain` e `baseline` do `.hea` via `wfdb.rdheader()`.

### 4.2 src.data.preprocessor.ECGPreprocessor

```python
class ECGPreprocessor:
    def __init__(self, config_path: Path):
        self.cfg = load_yaml(config_path)  # config/preprocess_v1.0.yaml
        self.fs = self.cfg["filter"]["target_fs"]  # 500.0
        self.lowcut = self.cfg["filter"]["lowcut"]   # 0.5
        self.highcut = self.cfg["filter"]["highcut"] # 40.0
        self.order = self.cfg["filter"]["order"]     # 4
        # Pré-computar coeficientes Butterworth para evitar recálculo por registro
        nyq = self.fs / 2.0
        self.b_band, self.a_band = scipy.signal.butter(
            self.order, [self.lowcut/nyq, self.highcut/nyq], btype="band"
        )

    def filter(self, x: np.ndarray) -> np.ndarray:
        # scipy.signal.filtfilt(self.b_band, self.a_band, x) — zero-phase

    def detrend(self, x: np.ndarray) -> np.ndarray:
        # scipy.signal.detrend(x, type="linear")

    def normalize(self, x: np.ndarray) -> np.ndarray:
        # Z-score global: (x - mean) / (std + eps)

    def process(self, x: np.ndarray, record_id: str) -> Tuple[np.ndarray, dict]:
        # 1. Resample (se fs != 500)
        # 2. filter → detrend → normalize
        # 3. Gerar metadata: input_range_mV, output_range_mV, mean, std, duration_sec
        # 4. Persistir linhagem em data/lineage/preprocess/{record_id}.json
        # 5. Retornar (x_processed, metadata)
```

> **Correção v1.0:** O v1.0 usava normalização min-max por segmento. Isso amplifica ruído em batimentos de baixa amplitude. O v1.1 usa **Z-score global** (fit no treino, transform em val/teste).

### 4.3 src.data.segmenter.ECGSegmenter

```python
class ECGSegmenter:
    def __init__(self, fs: float = 500.0, window_ms: float = 1000.0, min_window_ms: float = 600.0):
        self.fs = fs
        self.half_len = int((window_ms * fs) / 2000)  # 250 amostras para 1000ms
        self.min_half_len = int((min_window_ms * fs) / 2000)  # 150 amostras para 600ms
        self.window_len = 2 * self.half_len + 1  # 501 amostras (ímpar, R no centro)

    def segment_with_labels(self, sig, r_peaks, labels, rr_intervals_ms) -> Tuple[np.ndarray, np.ndarray]:
        # 1. Para cada batimento i:
        #    a. Calcular RR_interval atual (ms)
        #    b. Se RR_interval < 600ms: usar min_window (600ms) para evitar overlap
        #    c. Senão: usar window padrão (1000ms)
        #    d. Verificar se há amostras suficientes antes e depois do R-peak
        #    e. Se bordas insuficientes: descartar (NUNCA padding com zeros)
        # 2. Retornar X shape (n_segments, window_len) float32, y shape (n_segments,)
        # 3. Logar: n_descartados_bordas, n_usados_600ms, n_usados_1000ms
```

> **Correção v1.0:** O v1.0 fixava janela em 600ms (300 amostras). Isso descarta 40% do contexto temporal. O v1.1 usa **1000ms padrão** com **fallback 600ms** para taquicardia (RR < 600ms).

> **Regra de ouro:** Padding com zeros é proibido. Zeros introduzem descontinuidades artificiais que confundem filtros e CNNs.

### 4.4 src.features.ampt_500hz.AMPTDetector

```python
class AMPTDetector:
    """AccYouRate Modified Pan-Tompkins (Neri et al., 2023).
    Referência: https://github.com/Accyourate-Group-S-p-A/acy_ampt
    """
    def __init__(self, fs: float = 500.0):
        self.fs = fs
        # Bandpass: 5-15 Hz (MESMO do Pan-Tompkins original)
        self.b_band, self.a_band = scipy.signal.butter(
            2, [5.0/(fs/2), 15.0/(fs/2)], btype="band"
        )
        # Derivative: 5-point (Pan-Tompkins standard)
        # Squaring: point-by-point
        # Moving Window Integration: 150ms = 75 amostras @ 500Hz
        self.mwi_window = int(0.150 * fs)
        # Refratariedade / T-wave discrimination: 360ms = 180 amostras @ 500Hz
        self.refractory = int(0.360 * fs)

    def detect(self, sig: np.ndarray) -> np.ndarray:
        # 1. Bandpass filter (filtfilt para zero-phase)
        # 2. 5-point derivative
        # 3. Squaring
        # 4. Moving average integration (75 amostras, rectangular window)
        # 5. Adaptive thresholding (SPKF, NPKF, THRESHOLDF1, THRESHOLDF2)
        # 6. T-wave discrimination (slope check within 360ms window)
        # 7. Search-back for missed QRS (1.66 * RR_average1)

    def evaluate(self, sig, r_true, tol_ms: float = 150.0) -> dict:
        # TP, FN, FP, Sens, PPV, F1
```

> **Correção v1.0:** O v1.0 afirmava "banda 5-25 Hz (AMPT estendido)". **Isso está incorreto.** O AMPT usa os **mesmos filtros 5–15 Hz** do Pan-Tompkins original; simplifica apenas a fase de decisão. A banda 5–18 Hz é do Pan-Tompkins++ (arXiv 2024), não do AMPT.

> **Correção v1.0:** Tolerância de 3 amostras (6ms) é para localização de pico, não detecção de batimento. A tolerância padrão AAMI/PhysioNet para detecção é **150ms** (75 amostras @ 500Hz).

### 4.5 src.features.morphological.MorphologicalFeatures

```python
class MorphologicalFeatures:
    def extract(self, segments: np.ndarray, fs: float = 500.0, r_idx: int = None) -> list:
        # Para cada segmento:
        #   1. r_idx = np.argmax(np.abs(seg))
        #   2. r_amplitude = seg[r_idx]
        #   3. QRS onset (busca 300ms antes do R):
        #      - Envelope do sinal filtrado bandpass [5, 30] Hz
        #      - Threshold = 50% de |r_amplitude|
        #      - Onset = último ponto antes de r_idx onde envelope < threshold
        #   4. QRS offset (busca 150ms após o R):
        #      - Mesmo threshold
        #      - Offset = primeiro ponto após r_idx onde envelope < threshold
        #   5. qrs_width_ms = (offset - onset) / fs * 1000.0
        #   6. q_depth = np.min(seg[max(0, r_idx-50):r_idx])
        #   7. t_amplitude = np.max(seg[r_idx:min(len(seg), r_idx+150)])
        #   8. J-point = offset (fim do QRS)
        #   9. st_start = J-point + int(0.060 * fs)  # 60ms após J
        #   10. st_end = J-point + int(0.080 * fs)   # 80ms após J
        #   11. st_slope = np.polyfit(range(st_start, st_end), seg[st_start:st_end], 1)[0]
        #   12. qrs_area = np.trapezoid(np.abs(seg[onset:offset]), dx=1.0/fs)
```

> **Correção v1.0:** QRS width era definido vagamente como "largura a 50% amplitude". O v1.1 usa o **envelope method** com janelas de busca clínicas (onset 300ms antes, offset 150ms depois, threshold 50% de |R|). ST slope é medido a **J+60ms → J+80ms** (padrão ACC/AHA/HRS), não a r_idx + 0.08–0.12s.

### 4.6 src.features.aami_mapper.AAMIMapper

```python
AAMI_MAP = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "V": "V", "E": "V",
    "A": "S", "a": "S", "J": "S", "S": "S",
    "F": "F",
    "/": "Q", "f": "Q", "Q": "Q", "|": "Q",
}

AAMI_CLASSES = ["N", "S", "V", "F", "Q"]

def map_annotations(symbols: List[str]) -> Tuple[List[str], Dict]:
    # Mapear símbolos WFDB para classes AAMI EC57
    # Retorna: (labels_aami, stats)
    # stats: {n_total, n_unmapped, n_by_class, n_by_symbol}
```

> **Nota:** AAMI EC57:1998 define as métricas de avaliação (Se, PPV, FPR, F1, MCC). A implementação de referência é o `bxb` da PhysioNet.

---

## 5. Modelo — Especificação para Agentes

### 5.1 Arquitetura Backbone 1D-CNN

**Input shape:** `(500, 1)` — 1000ms @ 500Hz, 1 canal (MLII-equivalente)  
**Classes pré-treino:** 5 superclasses SCP-ECG (NORM, CD, MI, HYP, STTC) — sigmoid multi-label  
**Classes fine-tuning:** 5 (AAMI: N, V, S, F, Q) — softmax

```python
def build_backbone_1d(input_len: int = 500, num_classes: int = 5, embedding_dim: int = 64):
    """1D-CNN enxuto para STM32F4 (arena ~64KB, FlatBuffer < 64KB).
    Arquitetura baseada em literatura TinyML para ECG (≈13K–20K params):
    Input(500, 1)
    → Conv1D(16, kernel=7, activation="relu", padding="same") → MaxPool1D(2)   # 250
    → Conv1D(32, kernel=5, activation="relu", padding="same") → MaxPool1D(2)   # 125
    → Conv1D(64, kernel=3, activation="relu", padding="same") → MaxPool1D(2)   # 62
    → GlobalAveragePooling1D()                                                # 64
    → Dense(embedding_dim, activation="relu")                                   # 64
    → Dropout(0.3)                                                            # 64
    → Dense(num_classes, activation="softmax", name="output")                 # 5

    RESTRIÇÕES TFLM:
    - NÃO usar LSTM/GRU/RNN (suporte limitado em TFLM; alto consumo de SRAM)
    - NÃO usar BatchNormalization (pode ser folded, mas em PTQ full-integer
      requer cuidado com zero-point; preferir omitir para simplicidade)
    - NÃO usar SeparableConv1D (TFLM decompõe em DepthwiseConv + PointwiseConv;
      nem sempre otimizado via CMSIS-NN; preferir Conv1D padrão)
    - NÃO usar attention mechanisms (overhead de memória em TFLM)
    - NÃO usar GroupNorm/LayerNorm (suporte parcial em TFLM; evitar)
    """
```

**Estimativa de tamanho:**
- **Total params:** ~13.3K
- **Pesos INT8:** ~13KB
- **Bias INT32:** ~2KB
- **Scale/Zero-Point per-channel:** ~1KB
- **Tensor arena (ativações):** ~15–25KB
- **Scratch buffer (CMSIS-NN):** ~4–8KB
- **Interpreter overhead:** ~2KB
- **Total Arena:** ~40–50KB (< 64KB limite STM32F4)
- **FlatBuffer TFLM:** ~20–25KB (< 64KB)

> **Referência:** Estudo TinyML em Arduino UNO (32KB Flash, 2KB SRAM) usou 1D-CNN com ~18.5K params, 21.8KB Flash, 1.7KB SRAM, inferência 200ms/beat. O STM32F4 (192KB SRAM, 1MB Flash) tem margem 10× superior.

### 5.2 Treinamento — GroupKFold por Paciente (Inter-Patient)

**Regra absoluta:** NUNCA misturar batimentos do mesmo paciente em treino e teste. Data leakage em sinais fisiológicos invalida métricas. A literatura distingue intra-patient (~99% Acc, artificial) vs inter-patient (~88–96% Acc, realista).

```python
def train_group_kfold(X, y, groups, n_splits: int = 5, random_state: int = 42):
    """GroupKFold por paciente — NUNCA misturar batimentos do mesmo paciente.
    1. groups = array com ID do paciente (ex: "100", "101", ..., "I75")
    2. gkf = GroupKFold(n_splits=5)
    3. Para cada fold:
       a. Separar X_train, X_test, y_train, y_test por grupos
       b. Normalização global: fit scaler em X_train, transform em X_test
       c. Carregar backbone pré-treinado
       d. Congelar camadas convolucionais (trainable=False)
       e. Treinar classifier em X_train, y_train com class_weight="balanced"
       f. Avaliar em X_test, y_test (pacientes NUNCA vistos)
       g. Registrar métricas AAMI EC57 por fold
    4. Retornar: métricas médias + desvio padrão entre folds
    5. Persistir: melhor fold (maior F1-macro) como modelo final
    """
```

**Normalização de input:**
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
X_val_norm = scaler.transform(X_val.reshape(-1, 1)).reshape(X_val.shape)
joblib.dump(scaler, "models/input_scaler_v1.0.pkl")
# No firmware: aplicar mesma média/std (valores fixos compilados em C)
```

> **Regra:** NUNCA fazer fit do scaler no conjunto de teste. NUNCA normalizar por segmento individualmente.

### 5.3 Métricas de Avaliação — AAMI EC57

| Métrica | Fórmula | Threshold Mínimo (Inter-Patient) |
| :--- | :--- | :--- |
| **Acc global** | TP_total / N_total | > 93% |
| **F1-macro** | mean(F1 por classe) | > 85% |
| **MCC** | Matthews Correlation Coefficient | > 0.80 |
| **Sens N** | TP_N / (TP_N + FN_N) | > 96% |
| **Sens V** | TP_V / (TP_V + FN_V) | > 90% |
| **Sens S** | TP_S / (TP_S + FN_S) | > 75% |
| **Sens F** | TP_F / (TP_F + FN_F) | > 60% |
| **Sens Q** | TP_Q / (TP_Q + FN_Q) | > 70% |
| **FPR global** | FP_total / (FP_total + TN_total) | < 5% |
| **GroupKFold std F1-macro** | std entre 5 folds | < 3% |

> **Caveat:** Acurácia global (Acc) é enganosa em datasets desbalanceados. Um modelo que classifica tudo como N já atinge ~75% de Acc. **F1-macro e MCC são as métricas primárias** para comparação entre modelos.

### 5.4 Quantização PTQ — Per-Channel INT8

```python
def quantize_ptq(model, representative_data: np.ndarray, 
                 num_calibration_samples: int = 512) -> Tuple[bytes, dict]:
    """Quantização pós-treino para INT8 com per-channel quantization.
    1. Selecionar representative dataset estratificado:
       a. Amostrar de cada classe AAMI proporcionalmente
       b. Incluir batimentos de baixa/high amplitude
       c. Total: num_calibration_samples (default 512)
    2. Configurar converter:
       converter = tf.lite.TFLiteConverter.from_keras_model(model)
       converter.optimizations = [tf.lite.Optimize.DEFAULT]
       converter.representative_dataset = representative_dataset
       converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
       converter.inference_input_type = tf.int8
       converter.inference_output_type = tf.int8
    3. Extrair parâmetros de quantização:
       input_scale, input_zero_point = interpreter.get_input_details()[0]['quantization']
       output_scale, output_zero_point = interpreter.get_output_details()[0]['quantization']
    4. Validar degradação:
       - ΔAcc global < 1%, ΔF1-macro < 2%
       - ΔSens N < 0.5%, ΔSens V/S/F/Q < 3%
       - Se ΔF1-macro > 2%: aumentar calibration samples ou usar QAT
    """
```

> **Cuidado:** Em modelos multi-task complexos (12-lead, 75 classes), PTQ INT8 pode degradar drasticamente (AUC de 0.893 → 0.513). Para 1D-CNN simples (single-lead, 5 classes), a degradação típica é 0–1% se calibrado corretamente.

### 5.5 Exportação para Header C

```python
def tflite_to_header(tflite_bytes: bytes, output_path: Path, 
                     var_name: str = "g_ecg_model_data",
                     alignment: int = 16) -> Path:
    """Exportar .tflite para header C puro, sem dependências externas (sem xxd).
    1. Gerar array C a partir dos bytes do FlatBuffer
    2. Aplicar alinhamento: alignas(16) para ARM Cortex-M4
    3. Incluir metadata em comentários: SHA256, tamanho, data, versão
    4. Salvar: output_path
    5. Validar: compilação com arm-none-eabi-gcc -c -Werror
    """
```

---

## 6. Quality Gates (Gates de Liberação)

Nenhum código, dataset ou modelo avança para a próxima camada sem passar no gate.

| Gate | Critério | Dataset | Valor Mínimo | Como Validar | Bloqueia |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **QG0** | Download | Todos | Chapman ≥ 45.000; MIT-BIH 48; SVDB 78; AFDB 25; INCART 75; checksums SHA256 válidos; DLQ vazia | `pytest tests/test_download.py -v` | Pré-treino |
| **QG1** | Resample + Pré-processamento | MIT-BIH+ | Fs = 500 Hz; ganho/baseline lidos do .hea; range ±5 mV; zero-phase filter; Z-score global; linhagem 100% | `pytest tests/test_loader.py` + `test_resampler.py` + `test_preprocessing.py` | Segmentação |
| **QG2** | AMPT @ 500Hz | MIT-BIH | Sens > 96.5%, PPV > 99.0%, F1 > 97.5%, FP < 1%; banda 5–15 Hz; tol = 150 ms | `pytest tests/test_ampt.py -v` | Feature Engineering |
| **QG3** | Features | MIT-BIH+ | Janela 1000ms (600ms fallback); ≥ 10 dimensões; sem NaN/Inf; QRS width via envelope method; ST slope J+60ms→J+80ms; SMOTE apenas em feature space; augmentation apenas treino fine-tuning | `pytest tests/test_features.py` + `test_segmenter.py` | Modelagem |
| **QG4** | Pré-treino | Chapman | AUC-ROC macro > 0.85; loss < 0.15; 5 superclasses SCP-ECG; seed 42; determinístico | `pytest tests/test_pretrain.py -v` | Fine-tuning |
| **QG5** | Fine-tuning | MIT-BIH+ (inter-patient) | Acc > 93%; F1-macro > 85%; MCC > 0.80; Sens N > 96%, V > 90%, S > 75%, F > 60%, Q > 70%; FPR < 5%; GroupKFold std F1-macro < 3%; DLQ vazia; linhagem 100% | `pytest tests/test_finetune.py -v` | Quantização |
| **QG6** | Quantização + Exportação | MIT-BIH+ subset | Per-channel INT8; ΔAcc < 1%; ΔF1-macro < 2%; ΔSens N < 0.5%, V/S/F/Q < 3%; FlatBuffer < 64KB; arena TFLM < 64KB; CMSIS-NN ativado; header C compilável (-Werror); parâmetros de quantização extraídos; alinhamento 16 bytes | `pytest tests/test_quantization.py` + `test_tflm_integration.py` | Firmware (Fase 2) |

> **Nota sobre métricas inter-patient:** O padrão AAMI EC57:1998 exige avaliação "hands-off" e reprodutível. Métricas intra-patient (shuffle aleatório) são inválidas para validação final, pois produzem Acc ~99% artificialmente.

---

## 7. Prompts para Agentes de Coding (Atualizados)

### Agente A — Engenharia de Dados
```
Você é um engenheiro de dados especializado em sinais biomédicos.
Implemente em Python 3.13 (código limpo, type hints obrigatórios) os módulos:
- src/data/download_chapman.py (Kaggle API / Figshare / wget, com fallback mirror)
- src/data/download_mitbih.py (wfdb.io.dl_database, idempotente, com mirror fallback)
- src/data/loader.py (MITBIHLoader: ganho/baseline lidos do .hea, não hardcoded)
- src/data/resampler.py (scipy.signal.resample_poly para 500 Hz, padtype="line")
- src/data/lead_selector.py (MLII/II/ECG1 conforme dataset)
- src/data/preprocessor.py (Butterworth filtfilt 0.5-40Hz, detrend, z-score global)
- src/data/segmenter.py (janelas 1000ms centradas no R, fallback 600ms, sem padding zero)

Regras:
1. Usar apenas scipy, numpy, wfdb, pandas. Não inventar dependências.
2. Todos os datasets são resampleados para 500 Hz. SVDB é 250 Hz (não o valor obsoleto documentado na v1.0).
3. loader.py deve ler adc_gain/baseline do .hea via wfdb.rdheader().
4. segmenter.py deve descartar bordas (nunca padding com zeros).
5. Cada registro processado gera linhagem JSON em data/lineage/.
6. Falhas vão para data/.dlq/ com traceback e config.
7. Escrever tests/test_download.py, test_loader.py, test_resampler.py, test_preprocessing.py, test_segmenter.py.

Entregável: código + testes passando em `make test`.
```

### Agente B — Feature Engineering
```
Você é um cientista de dados especializado em features de ECG.
Implemente em Python 3.13:
- src/features/ampt_500hz.py (AMPT detector completo, banda 5-15 Hz, MWI 150ms, refratariedade 360ms)
- src/features/time_domain.py (RR, HRV, RMSSD, heart_rate)
- src/features/morphological.py (R-amp, Q-depth, T-amp, QRS-width via envelope method, ST-slope J+60ms→J+80ms)
- src/features/augmentation.py (jitter, baseline wander, powerline, time warp — apenas treino fine-tuning)
- src/features/balancer.py (SMOTE/ADASYN no feature space, nunca no sinal bruto)
- src/features/aami_mapper.py (WFDB → AAMI EC57, 5 classes)

Regras:
1. AMPT deve ser fiel ao paper Neri et al. 2023: banda 5-15 Hz (não 5-25 Hz), tol=150ms.
2. NÃO usar biosppy, heartpy, neurokit2 — implementar do zero para auditabilidade.
3. evaluate() deve retornar TP, FN, FP, Sens, PPV, F1 para validação contra .atr.
4. QRS width: envelope method, onset 300ms antes, offset 150ms depois, threshold 50% de |R|.
5. ST slope: J-point + 60ms → 80ms (padrão ACC/AHA/HRS).
6. Augmentation apenas no treino do fine-tuning; NUNCA no pré-treino ou teste.
7. SMOTE apenas no espaço de features para classificadores tradicionais.

Entregável: código + testes/test_features.py + test_ampt.py passando.
```

### Agente C — Modelagem + Quantização
```
Você é um engenheiro de ML especializado em TinyML médico.
Implemente em Python 3.13 + TensorFlow 2.21:
- src/models/backbone_1d.py (arquitetura 1D-CNN, input 500 amostras, ~13K params)
- src/models/pretrain_chapman.py (pré-treino multi-label SCP-ECG, 5 superclasses, sigmoid)
- src/models/finetune_mitbih.py (fine-tuning com backbone congelado, class_weight="balanced")
- src/models/train.py (GroupKFold por paciente, 5 folds, inter-patient, seed 42)
- src/models/evaluate.py (métricas AAMI EC57: Se, PPV, FPR, F1, MCC por classe)
- src/quantization/ptq.py (PTQ INT8 per-channel, 512 amostras estratificadas)
- src/quantization/extract_params.py (extração de scale/zero_point para firmware)
- src/quantization/export_tflite.py (exportação para header C, Python nativo, sem xxd)

Regras:
1. GroupKFold por paciente é obrigatório — nunca shuffle aleatório.
2. Normalização global (StandardScaler fit no treino, transform em teste). Nunca por segmento.
3. Métricas primárias: F1-macro e MCC (não Acc global, que é enganosa).
4. PTQ: per-channel INT8, input/output int8, representative dataset estratificado.
5. Degradação: ΔF1-macro < 2%, ΔSens N < 0.5%, ΔSens V/S/F/Q < 3%.
6. model_data.h: alignas(16), const unsigned char[], metadata em comentários.
7. quantization_params.h: struct tipado com scale e zero_point para input/output.
8. Seed 42, determinístico. Persistir config e linhagem em experiments/.

Entregável: modelo treinado, métricas AAMI, model_data.h + quantization_params.h prontos.
```

### Agente D — DevOps/Integração
```
Você é um engenheiro de infraestrutura.
Configure:
- pyproject.toml (PEP 621, uv, dependências prod + dev)
- uv.lock (lockfile determinístico)
- Makefile (targets: env, download-all, process, pretrain, finetune, quantize, export, test, quality-report, docker-build, clean)
- Dockerfile (python:3.12-slim-bookworm, uv, reprodutível)
- docker-compose.yml (volumes para data/models/reports)
- .pre-commit-config.yaml (black, isort, flake8, mypy, bandit, check-large-files)
- .github/workflows/ci.yml (lint → unit-tests → integration-tests → quality-gates, DVC pull, artifact upload)
- DVC setup (.dvcignore, remote S3)
- scripts/validate_firmware_deliverables.py (compilação arm-none-eabi-gcc -Werror)
- scripts/generate_quality_report.py (markdown consolidado QG0–QG6)

Regras:
1. Não usar requirements.txt congelado — usar uv + pyproject.toml.
2. CI deve usar cache externo S3 para datasets > 1GB (Chapman ~5GB).
3. Pre-commit obrigatório — nunca bypassar com --no-verify.
4. Makefile deve suportar paralelismo: make -j4 download-all.
5. Docker deve reproduzir ambiente identicamente em qualquer máquina.
6. Artifact retention: modelos 30 dias, reports 90 dias, coverage 7 dias.
```

---

## 8. Handoffs para Fase 2 (Firmware)

| Entregável | Origem | Destino | Formato | Validação | Nota |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `model_data.h` | Agente C | `firmware/src/ml/` | `alignas(16) const unsigned char[]` + `len` | `arm-none-eabi-gcc -c -Werror` | FlatBuffer INT8 < 64KB |
| `quantization_params.h` | Agente C | `firmware/src/ml/` | `struct { float scale; int32_t zero_point; }` | `sizeof()` == 8 bytes | Input/output dequantização |
| `filter_coeffs_q31.h` | Agente B (SciPy) | `firmware/src/dsp/` | Coeficientes IIR/FIR Q31 para CMSIS-DSP | RMSE < 1e-6 vs Python | Butterworth bandpass 0.5–40 Hz |
| `features_config.h` | Agente B | `firmware/src/features/` | `struct` com offsets, escalas, janelas | `sizeof()` alinhado a 4 bytes | AMPT params @ 500Hz |
| `normalization_params.h` | Agente C | `firmware/src/ml/` | `float mean, std;` | Consistência treino vs inference | Z-score global |
| `mitbih_100.h` | Agente A | `firmware/tests/hil/` | Array de amostras em mV (500Hz, 30min) | Playback determinístico | Registro 100 para HIL test |

---

## 9. Decisões do Arquiteto — CONFIRMADAS

### Decisão 1: Escopo de Dados — OPÇÃO E (Chapman-Shaoxing + MIT-BIH+)
**Status:** APROVADA em 2026-06-09  
**Justificativa:** Volume ~5.1 GB raw (Chapman) com diversidade demográfica + ground truth AAMI beat-level (MIT-BIH+). Melhor custo-benefício entre precisão e viabilidade.

### Decisão 2: Frequência de Amostragem — OPÇÃO B (Resample 500 Hz)
**Status:** APROVADA em 2026-06-09  
**Justificativa:** Chapman e PTB-XL são nativamente 500 Hz. Padronizar todo o pipeline em 500 Hz elimina dual-rate. MIT-BIH (360Hz), SVDB (250Hz), AFDB (250Hz), INCART (257Hz) são resampleados via `scipy.signal.resample_poly` com filtro FIR interno.  
**Impacto no modelo:** Input shape `(500, 1)` para janela de 1000ms.  
**Impacto no firmware:** ADS1292R configurado para 500 SPS.

### Decisão 3: Display Simulador — 320×320 (Dispositivo Médico Portátil)
**Status:** APROVADA em 2026-06-09  
**Justificativa:** Área suficiente para traçado ECG + métricas de arritmia.  
**Impacto no LVGL:** `LV_HOR_RES = 320`, `LV_VER_RES = 320`.

### Decisão 4: Target Primário — STM32F4 (Cortex-M4F, 168MHz, 192KB SRAM)
**Status:** APROVADA em 2026-06-09  
**Justificativa:** Padrão FDA em dispositivos médicos. 192KB SRAM comporta arenas: DSP 32KB + TFLM 64KB + SQLCipher 96KB = 192KB (limite apertado, exige otimização).  
**Fallback:** STM32F7 (320KB SRAM) ou STM32H7 (1MB SRAM) se modelo exceder 64KB arena.

### Decisão 5: Canais AFE — 1 Canal (MLII-equivalente)
**Status:** APROVADA em 2026-06-09  
**Justificativa:** MIT-BIH usa MLII como lead primário. Chapman/INCART usam lead II (anatomicamente idêntico a MLII). Simplifica DSP e reduz arena TFLM.  
**Impacto no modelo:** Input shape `(500, 1)` — canal único.

---

## 10. Checklist de Aceite da Fase 1 (Atualizado)

- [ ] `make env` instala dependências via uv em < 1 minuto.
- [ ] `make download-all` baixa todos os datasets em < 10 minutos (com mirror).
- [ ] `make test` passa QG0, QG1, QG2, QG3 em < 2 minutos (unit tests).
- [ ] `make process` gera `data/lineage/` com 100% de cobertura.
- [ ] `make pretrain` converge em 30 épocas com AUC-ROC macro > 0.85.
- [ ] `make finetune` atinge F1-macro > 85%, MCC > 0.80 (inter-patient).
- [ ] `make quantize` gera `.tflite` < 64KB com ΔF1-macro < 2%.
- [ ] `make export` gera `model_data.h` + `quantization_params.h` compiláveis.
- [ ] `make all` executa pipeline completo de ponta a ponta.
- [ ] `make docker-build` + `make docker-run` reproduzem ambiente identicamente.
- [ ] Pre-commit passa em todos os arquivos (black, isort, flake8, mypy, bandit).
- [ ] CI no GitHub Actions passa em ubuntu-24.04 (lint → unit → integration → QG4–QG6).
- [ ] DVC push sincroniza datasets brutos para remote S3.
- [ ] Relatório de qualidade `reports/quality_report.md` gerado e validado.
- [ ] DLQ vazia: `data/.dlq/` não contém falhas pendentes.
