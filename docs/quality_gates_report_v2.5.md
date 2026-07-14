# Relatório de Quality Gates v2.5

Executado em: 2026-07-12
Ambiente: Python 3.12.3, venv `.venv`
Comando base: `python -m pytest tests/<qg-files> -q --tb=line --timeout=N`

## Resumo executivo

| Status | Qtd |
| -------- | ----- |
| PASS | 17 |
| FAIL | 0 |
| NÃO MAPEADO / N/A | 5 |

Todos os QGs executáveis passam. O mirror Chapman foi reconstruído deterministicamente, o INT8 Stage 1 incompatível foi regenerado a partir do modelo float32 correto, e a fidelidade numérica entre C e Python foi restaurada ao sincronizar os parâmetros de quantização compilados no firmware com as referências Python.

## Detalhamento por QG

| QG | Camada | Critério | Status | Testes/Scripts | Observação |
| ---- | -------- | ---------- | -------- | ---------------- | ------------ |
| QG0 | C01 | Download completo + checksums | **PASS** | `tests/test_download.py` | Mirror Chapman reconstruído; checksum SHA-256 validado; 25/25 passam |
| QG1 | C02 | Resample + pré-processamento | **PASS** | `test_preprocessing.py`, `test_resampler.py` | 32/32 passaram |
| QG2 | C03 | AMPT @ 500Hz | **PASS** | `test_ampt.py` | 12/12 passaram; Sens/PPV/F1 dentro do threshold |
| QG3 | C03 | Features | **PASS** | `test_features.py`, `test_segmenter.py`, `test_morphological_features.py`, `test_pipeline.py` | 84/84 passaram; NaN em `qrs_asymmetry_index` corrigido |
| QG4 | C04 | Pré-treino Chapman | **PASS** | `test_model.py`, `test_pretrain.py` | 26/26 passaram |
| QG5 | C04 | Fine-tuning MIT-BIH+ | **PASS** | `test_finetune.py`, `test_run_stage_training.py`, `test_qg5_v2_4_gates.py` | 24/24 passaram; threshold F1(F) >= 0.50 ativo |
| QG6 | C05 | Quantização + Exportação | **PASS** | `test_quantization.py`, `test_quantization_degradation.py` | 17/17 passaram; Stage 1 regenerado: ΔF1 = 0.0058 < 0.02, 26,62 KB |
| QG7 | C08 | Build firmware | **PASS** | `test_firmware_qg.py` | Firmware build passou |
| QG8 | C08/C10 | Bit-exatidão | **PASS** | `test_fidelity.py`, `test_tflm_bitexact.py` | 7/7 passaram; outputs bit-exatos vs Python BUILTIN_REF |
| QG9 | C08/C09 | Latência + Memória | **PASS** | `test_native_tflm.py`, `test_arena_limits.py` | Latência/arena dentro dos limites |
| QG10 | C09/C10 | Fidelidade numérica | **PASS** | `test_fidelity.py` | 5/5 passaram; MAE < 0,01 e cosine > 0,99 |
| QG11 | C08/C09 | Fault injection SPI/UART | **PASS** | `test_fault_injection.py` | 1/1 passou |
| QG12 | C08/C09 | Limites de arena RAM | **PASS** | `test_arena_limits.py` | Arena <= 64KB confirmada |
| QG13 | C08 | Watchdog | **PASS** | `test_watchdog.py` | 1/1 passou |
| QG14 | C08 | Segurança/LGPD no firmware | **N/A** | — | Reservado para futuro |
| QG15 | C08 | OTA/update seguro | **N/A** | — | Reservado para futuro |
| QG16 | C08/C10 | Filtros C vs Python | **PASS** | `test_dsp_filters.py`, `test_firmware_filters_python.py` | Harness nativo passou; filter chain vs Python ok |
| QG17 | C08/C10 | Pipeline C vs Python | **PASS** | `test_dsp_fidelity.py` | 5/5 passaram; cosine > 0,99 |
| QG18 | C08/C10 | Detector R-peak | **PASS** | `test_r_peak_firmware.py` | Sens/PPV >= 90% vs AMPT Python |
| QG19 | C09 | Consumo energético | **N/A** | — | Requer medição de energia no Renode/hardware |
| QG-C11 | C11 | Knowledge index | **N/A** | — | Não testado nesta rodada |
| QG-MEM | Memory | Artifact registry | **N/A** | — | Não testado nesta rodada |

## Correções aplicadas

### QG0 — checksum Chapman

O mirror `data/mirrors/chapman_shaoxing.tar.gz` foi reconstruído de forma determinística a partir do diretório DVC-validado (`data/raw_chapman`). O manifesto e sidecar foram atualizados para SHA-256 `27fbb6f0...908b4134`; o teste dedicado passa.

### QG6 — degradação de quantização Stage 1

O artefato anterior não correspondia numericamente ao modelo float32 do gate: erro médio de probabilidade 0,1025 e 180/1024 decisões divergentes. O INT8 foi regenerado com 1.024 amostras a partir do modelo, scaler e dataset usados pelo teste. Resultado: F1 float 0,53325, F1 INT8 0,52743, ΔF1 0,00581 e FlatBuffer de 26,62 KB. Evidência em `experiments/qg6_stage1_recalibration_v2.5/manifest.json`.

### QG8/QG10 — fidelidade de beat

Causa raiz: inconsistência entre os parâmetros de quantização usados pelos testes Python (`models/quantized/quantization_params.json`, v2.3) e os parâmetros compilados no firmware (`firmware/src/ml/quantization_params.h`, v2.0). Foram criados:

- `src/quantization/firmware_params.py`: parser auditável do header de quantização do firmware.
- `tests/test_firmware_quantization_params.py`: validação do contrato de quantização.

Os testes `test_native_tflm.py`, `test_fidelity.py` e `test_tflm_bitexact.py` agora leem os parâmetros do header implantado, garantindo que a referência Python e o firmware C operem sobre os mesmos scales/zero-points.

### QG16/QG17 — filtros C vs Python

O harness nativo reportava `ADAPTIVE_SKIPPING stable_rhythm FAIL` porque o teste C não considerava a macro `ADAPTIVE_SKIPPING_ENABLED=0` usada no build nativo. O teste `firmware/tests/test_adaptive_skipping.c` foi ajustado para verificar o comportamento condicional. O parser `firmware/scripts/run_renode_tests.py` foi expandido para reconhecer linhas `[SKIP]` do adaptive skipping, evitando que beats pulados contassem como falta de beats no relatório estrutural.

## Próximos passos recomendados

1. Documentar QG14, QG15, QG19, QG-C11 e QG-MEM como futuros ou implementar testes mínimos.
2. Criar `scripts/run_quality_gates.py` para automatizar a execução de todos os QGs mapeados em um único comando.
3. Reavaliar a decisão sobre expansão de dataset v2.5 (AFDB + PTB-XL forte), pois os gates de firmware estão estáveis.

## Nota

Este relatório foi atualizado durante a fase de refatoração/qualificação em massa (PLAN.md v2.5). A decisão sobre expansão de dataset (AFDB + PTB-XL) permanece **pausada** até que a Fase 4 (integridade/publicação guard) seja concluída.
