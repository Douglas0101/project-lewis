# AGENTS.md — Project-Lewis: Contexto de Projeto SDD

## Escopo
Sistema de classificação de arritmias ECG em edge (STM32F4).
Pipeline: ingestão → resample → pré-processamento → features → modelagem → quantização → firmware → simulação Renode.

## Stack Aprovada
| Camada | Tecnologia | Observação |
|--------|-----------|------------|
| Python | 3.12.x | System Python Zorin OS; veto 3.13+ |
| Gerenciador | uv (Astral) | Lockfile nativo; nunca requirements.txt cru |
| Dados | numpy, scipy, pandas, wfdb | Base numérica + sinais |
| ML | TensorFlow 2.21, scikit-learn, imbalanced-learn | Treinamento + utilidades |
| Tracking | SQLite, SQLAlchemy 2.0 | Banco local de experimentos, métricas e alertas |
| Testes | pytest (>=8.0), pytest-cov, pytest-xdist | Pirâmide 70/20/10 |
| Qualidade | black, isort, flake8, mypy, bandit, pre-commit | Hooks obrigatórios |
| Container | Docker, docker-compose | Reprodutibilidade |
| Dados | DVC (remote local `~/.cache/project-lewis-dvc`; S3/GCS opcional) | Versionamento de datasets |
| Firmware | C/C++ bare-metal, arm-none-eabi-gcc 13.3.rel1 | Bare-metal |
| ML Embarcado | TFLM, CMSIS-DSP, CMSIS-NN | Aceleração Cortex-M4F; TFLM clonado em `firmware/third_party/tflite-micro/` e pinado por `firmware/third_party/tflite-micro.commit` |
| Knowledge (C11) | sqlite-vec, sentence-transformers, MCP Python SDK | RAG local offline; sem LangChain/Chroma/typer |
| Memory | `src/memory/` + tabela `Artifact` do tracking | Checksums SHA-256 e registro de artefatos por run |
| Simulação | Renode 1.15.3 | Emulação fiel STM32F4 |
| Hardware | STM32F407VG, ADS1292R | 168 MHz, 192KB SRAM, 1MB Flash |
| Compliance | LGPD Lei 13.709/18 | Por design |

## Datasets Versionados (C01)

| Dataset | Registros | Uso no pipeline | Estado |
|---------|-----------|-----------------|--------|
| Chapman-Shaoxing | 45.152 | Pré-treino backbone (superclasses SCP-ECG) | ✅ presente + mirror + DVC |
| MIT-BIH Arrhythmia | 48 | Fine-tuning / teste (AAMI beat-level) | ✅ presente + ZIP cache + DVC |
| MIT-BIH SVDB | 78 | Fine-tuning (supraventricular) | ✅ presente + ZIP cache + DVC |
| MIT-BIH AFDB | 25 | Fine-tuning (fibrilação atrial) | ✅ presente + ZIP cache + DVC |
| INCART | 75 | Fine-tuning (diversidade russa) | ✅ presente + ZIP cache + DVC |
| PTB-XL | 43.598 | **Fallback adicional** para pré-treino / backbone alternativo | ✅ presente + DVC |

> **Nota:** PTB-XL não consta da `ESPECIFICACAO_Fase1_Agentes-v1.1.md` original, mas foi adicionado como fallback de pré-treino por também conter 12-lead ECG a 500 Hz com superclasses SCP-ECG.

## Camadas SDD (Project-Lewis)
1. **C01 — Ingestão** — `docs/Camada-01-Ingestao-v1.1.md`
2. **C02 — Resample/Pré-processamento** — `docs/Camada-02-Resample-Preprocessamento-v1.1.md`
3. **C03 — Feature Engineering** — `docs/Camada-03-Feature-Engineering-v1.1.md`
4. **C04 — Modelagem** — `docs/Camada-04-Modelagem-v1.1.md`
5. **C05 — Quantização/Exportação** — `docs/Camada-05-Quantizacao-Exportacao-v1.1.md`
6. **C06 — Validação/QG** — `docs/Camada-06-Validacao-Quality-Gates-v1.1.md`
7. **C07 — DevOps/Integração** — `docs/Camada-07-Integracao-DevOps-v1.1.md`
8. **C08 — Firmware** — `docs/Camada-08-Firmware-v1.1.md`
9. **C09 — Simulação/Energia** — `docs/Camada-09-Simulacao-v1.1.md` / `docs/Camada-09-Energia-v1.4.md`
10. **C10 — Test Harness** — `docs/SDD_Project-Lewis_v3.md` (seção 3.10)
11. **C11 — Knowledge Layer (RAG)** — `docs/SDD-C11-Knowledge-Impl-v2.0.md`

## Quality Gates (QG0–QG19)
| QG | Camada | Critério | Threshold |
|----|--------|----------|-----------|
| QG0 | C01 | Download completo + checksums | Chapman ≥ 45k, MIT-BIH 48, SVDB 78, AFDB 25, INCART 75; PTB-XL como fallback adicional para pré-treino |
| QG1 | C02 | Resample + pré-processamento | Fs=500Hz, range ±5mV, Z-score, linhagem 100% |
| QG2 | C03 | AMPT @ 500Hz | Sens > 96.5%, PPV > 99.0%, F1 > 97.5% |
| QG3 | C03 | Features | Janela 1000ms, ≥10 dimensões, sem NaN, SMOTE em feature space |
| QG4 | C04 | Pré-treino Chapman | AUC-ROC macro > 0.85, loss < 0.15 |
| QG5 | C04 | Fine-tuning MIT-BIH+ (v2.2) | Pipeline duas etapas (N vs Anormal → S/V/F); Acc > 78%, F1-macro > 30%; QG5' Estágio 1: recall Anormal ≥ 30%, precision Anormal ≥ 25%, F1-macro ≥ 55%; QG5' Estágio 2: F1(S) ≥ 55%, F1(V) ≥ 70%, F1(F) ≥ 15%, F1-macro ≥ 45% |
| QG6 | C05 | Quantização + Exportação | ΔF1-macro < 2%, FlatBuffer < 64KB, header compilável |
| QG7 | C08 | Build firmware | Sem warnings (-Werror), FlatBuffer < 64KB |
| QG8 | C08/C10 | Bit-exatidão | int8 vs Python BUILTIN_REF |
| QG9 | C08/C09 | Latência + Memória | < 200ms/batimento, RAM < 64KB, Flash < 512KB |
| QG10 | C09/C10 | Fidelidade numérica | cosine > 0.99 vs ground-truth |
| QG11 | C08/C09 | Fault injection SPI/UART | Sistema recupera ou reporta erro sem travar |
| QG12 | C08/C09 | Limites de arena RAM | Arena TFLM ≤ 48KB / 64KB conforme configuração |
| QG13 | C08 | Watchdog | Reseta após timeout de inferência |
| QG14 | C08 | Reservado — segurança/LGPD no firmware | Verificação futura |
| QG15 | C08 | Reservado — OTA/update seguro | Verificação futura |
| QG16 | C08/C10 | Filtros C vs Python | RMSE < 1e-6 |
| QG17 | C08/C10 | Pipeline C vs Python | Equivalência funcional |
| QG18 | C08/C10 | Detector R-peak | Sens/PPV ≥ 90% vs AMPT Python |
| QG19 | C09 | Consumo energético | < 50 mA médio, < 165 mJ/batimento, > 10 h autonomia |
| QG-C11 | C11 | Knowledge index | DB < 500 MB, zero PII, MRR@5 ≥ 0.7, recall camada ≥ 0.9 |
| QG-MEM | Memory | Artifact registry | Checksum único por artefato, path resolvível, FK para run |

## Regras de Ouro
1. Nunca usar Radix UI
2. Sempre validar com Zod/pydantic (contratos de dados)
3. Sempre testar antes de commitar (TDD)
4. GroupKFold por paciente é obrigatório
5. Padding com zeros é proibido
6. SMOTE apenas no espaço de features
7. Augmentation apenas no treino de fine-tuning
8. Normalização Z-score global
9. AMPT usa banda 5–15 Hz
10. Tolerância AMPT: 150 ms
11. Input shape: (500, 1)
12. FlatBuffer TFLM < 64KB, arena < 64KB
13. Senhas hasheadas (Argon2id/bcrypt) — se houver camada de auth
14. LGPD: nenhum PII em logs
15. Revisão humana para código crítico
16. Classe Q — **v3.0.0**: renomeada `Q_OR_UNKNOWN` (classe de rejeição/abstenção, decisão D4); excluída dos alvos clínicos (Stage 1 e Stage 2) e roteada para `ABSTAIN_*`; em v2.x era tratada como "Anormal" no Estágio 1 (encerrado — DQ-05)
17. **v3.0.0** — Ontologia clínica única versionada em `src/features/ontology_v3.py` (fonte única símbolo→classe); `F` = `FUSION` (somente fusão V+N, **nunca** fibrilação atrial); símbolos desconhecidos são excluídos, nunca mapeados para Q
18. **v3.0.0** — Índices de anotação WFDB são reescalonados do fs nativo para o relógio canônico de 500 Hz **antes** da segmentação e das features (`round(s × 500/fs_nativo)`); índice nativo sobre sinal reamostrado é defeito fatal (DQ-01/DQ-02)

> **Nota v3.0.0 (2026-07-18):** após a auditoria forense (`reports/forensic_data_quality_report_v1.0.md`),
> o projeto está em reconstrução conforme `docs/rebuild_spec/`. Artefatos legados (modelos, scalers,
> thresholds v2.x) são **inválidos para novo treinamento** (`LEGACY_ARTIFACTS_INVALID_FOR_NEW_TRAINING`).
> AFDB contribui como tarefa de ritmo (AFIB/AFL) em escopo de episódio, fora do classificador de
> batimentos (decisão D3). Splits são congelados em `data/splits/groupkfold_5_stratified/v3/`.

> **Nota:** A arquitetura atual não inclui autenticação. Esta regra é condicional e só se aplica se uma camada de auth for introduzida no futuro.

> **Nota E07R-PD (2026-07-26):** remediação patient-disjoint concluída. MIT-BIH 201/202 unificados por
> evidência oficial PhysioNet; splits `v4.0-patient-disjoint` congelados em
> `data/splits/stage2_multiclass_patient_disjoint_v4.0/`; Stage 2 r5 `data/features/v3.1.0-r5-stage2-pd/`;
> freeze de 101 pins (`experiments/stage2_v2.4_research/integrity/e07r_freeze_manifest.json`) com
> preflight fail-closed 9/9. E06.5-PD 100/100 células: **H6 não supera baseline** (ΔF1(F) = −0,1601,
> IC95 [−0,398; +0,153]) → `NO_VALID_CANDIDATE`; **E07-PD não executado (0/150)** por pré-registro.
> `models/` restaurado ao estado aprovado em gate e congelado por hash — qualquer escrita em `models/`
> derruba o preflight. Publicação: `HOLD`. Evidência: `integrity/e07r_evidence_package.json` e
> `docs/e07r_evidence_report.md`.

> **Nota Makefile (FASE 7, 2026-07-26):** o Makefile foi padronizado — 60 alvos públicos por seção
> (`make help`), domínios `data-*`, `mlp-*`, `e07r-*`, `fw-*`, `gates-*`, `kb-*`, `rag-*`, `obs-*`;
> flags `DRY_RUN=1`, `FORCE=1`, `RUN_ID=...`, `STAGE=...`, `JSON=1`. Alvos antigos seguem válidos como
> aliases `DEPRECATED` (ex.: `download-all` → `data-download-all`, `mlp-pipeline` → `mlp-run`,
> `firmware-build` → `fw-build`, `hard-gates` → `gates-firmware`, `knowledge-*` → `kb-*`).
> Referência: `docs/make_commands.md`. Treinos E07R: `make e07r-e065` (resume), `make e07r-e065
> FORCE=1` (re-treino com arquivamento), `make e07r-e07`, dashboard `make e07r-watch`.

## Comando de Verificação
```bash
make lint && make test && make test-e2e
```

## Workflow de Implementação
1. Leia `AGENTS.md` + `docs/camada-XX-*.md` + `docs/SDD_Project-Lewis_v3.md`
2. Gere/atualize `PLAN.md` com tasks decompostas
3. Implemente **uma task por sessão**
4. Valide com quality gates da camada
5. Commit semântico
6. Revisão humana (obrigatória para security, firmware, LGPD)
