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
16. Classe Q (paced/unclassifiable) excluída da classificação final a partir de v2.0 — tratada como "Anormal" no Estágio 1

> **Nota:** A arquitetura atual não inclui autenticação. Esta regra é condicional e só se aplica se uma camada de auth for introduzida no futuro.

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
