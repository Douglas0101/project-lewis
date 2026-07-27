# E07R — remediação dos gates de integridade full-suite (task #18)

**Data:** 2026-07-26
**Decisão:** `RESOLVED`
**Escopo:** QG5' Stage 1 e QG8/QG10 bit-exatidão TFLM — pré-condições do freeze E07R
**Autoridade:** `AUTONOMOUS_GOVERNANCE_PREAUTH` (sem relaxamento de thresholds, sem promoção de modelos)

## 1. Resumo

Os dois gates vermelhos que bloqueavam o freeze E07R foram resolvidos por **restauração byte-a-byte dos artefatos pinados v2.0 para o último estado aprovado em gate**, com proveniência verificada por hash, histórico git e testes de consistência float32↔int8. Nenhum threshold foi alterado, nenhum modelo novo foi promovido e nenhum byte histórico foi apagado (artefatos defeituosos preservados em `artifacts/e07r_gate_remediation/bad_era/` com SHA-256).

## 2. Gate 1 — QG5' Stage 1 (recall Anormal 0.0661 < 0.30)

### Causa raiz

`models/stage1_float32_v2.0.keras` e `models/input_scaler_stage1_v2.0.pkl` foram sobrescritos em 2026-07-04 pelo experimento `20260704_033953_stage1_v2.0`, cujo próprio registro de lineage declarava `passes_qg5: false` (F1-macro 0.5099, MCC 0.0244, 13.218 parâmetros). A investigação R00–R04 (`artifacts/stage1_recall_investigation/`) tratou o artefato defeituoso como baseline e descartou causas de runtime (dropout, RNG, loader, modo de inferência), sem consultar o lineage.

### Remediação

Restaurado o par modelo+scaler de `experiments/20260622_054028_stage1_v2.0/fold_1/`:

- `stage1_float32_v2.0.keras` → SHA-256 `cb9e1222…a4c5a` (38.594 params, mesma arquitetura do contrato R02)
- `input_scaler_stage1_v2.0.pkl` → SHA-256 `ac9d6fab…214a`

### Validação

- QG5' subset (2.048 amostras, threshold 0,58): **recall(Anormal)=0,3094 ≥ 0,30**, precision=1,0 — margem coerente com a avaliação histórica de 22/jun (0,3254).
- Consistência com o tflite int8 original (QG10): cosine_mean=0,9996, argmatch=0,9609 — prova de que este é o float32 de origem do tflite de 26/jun.
- Candidato falsificado: `20260622_204653_stage1_v2.0/fold_0` (mesma arquitetura, recall 0,6745, mas cosine_mean=0,9578/argmatch=0,6367 contra o tflite original — rejeitado por QG10).

## 3. Gate 2 — QG8/QG10 bit-exatidão (firmware=0 beats)

### Causa raiz (dupla)

1. `reports/firmware_simulation_report.json` estava **stale** (17/jun, firmware v1.2 de 5 classes, sem campo `class=` no log UART): o regex do teste não lia nenhum beat.
2. O header de firmware `firmware/src/ml/stage1_int8_v2.0.h` e o tflite `models/quantized/stage1_int8_v2.0.tflite` foram sobrescritos em 2026-07-11 por uma requantização do modelo defeituoso (27.264 bytes, 13.218 params), divergindo do header original de 26/jun (55.664 bytes, 38.594 params) que sobreviveu em `models/quantized/stage1_int8_v2.0.h`. Além disso, o build do firmware não rastreia dependências de headers de modelo (objetos `.o` stale após troca de header).

### Remediação

- `models/quantized/stage1_int8_v2.0.tflite` → restaurado de `git show f5c6ef1:` (SHA-256 `acf0c8fb…3864`; idêntico aos bytes embutidos no header sobrevivente de 26/jun).
- `firmware/src/ml/stage1_int8_v2.0.h` → restaurado de `git show 305505f:` (SHA-256 `9718b5d2…8f4`).
- `firmware/src/ml/quantization_params.h` → restaurado de `git show 305505f:` (scale 0,0680587143 / zp −20, casando com o tflite original).
- Rebuild **limpo** (`make -C firmware clean && make -C firmware LEWIS_USE_TFLM=1 firmware-test`) para eliminar objetos stale; simulação Renode regenerada em `firmware/build/stm32f4/firmware_simulation_report.json` (2 beats, checks estruturais PASS).
- Estágio 2 verificado íntegro (tflite, header de firmware e float32 consistentes; nenhuma ação).

### Validação

- `tests/test_tflm_bitexact.py::test_bitexact_vs_firmware`: **PASS** (outputs int8 firmware == Python dentro de 5 LSB).
- `tests/test_bit_exact_python_tflm.py` (QG10 float32↔int8): **PASS** (cosine_mean 0,9996 > 0,99; argmatch 0,9609 > 0,94).
- `tests/test_fidelity.py` (QG10 UART Renode): **PASS** 5/5 após regeneração determinística dos fixtures de ground-truth com os artefatos restaurados.

## 4. Arquivos alterados

| Arquivo | Ação |
| --- | --- |
| `models/stage1_float32_v2.0.keras` | restaurado (fold_1 de 20260622_054028) |
| `models/input_scaler_stage1_v2.0.pkl` | restaurado (fold_1 de 20260622_054028) |
| `models/quantized/stage1_int8_v2.0.tflite` | restaurado (git f5c6ef1) |
| `firmware/src/ml/stage1_int8_v2.0.h` | restaurado (git 305505f) |
| `firmware/src/ml/quantization_params.h` | restaurado (git 305505f) |
| `tests/test_stage1_keras_artifact_contract.py` | SHA/params do artefato pinado atualizados (38594 params) |
| `tests/test_stage1_loader_direct_equivalence.py` | SHA do artefato pinado atualizado |
| `data/lineage/models/stage1_float32_v2.0.json` | registro corretivo com proveniência da restauração |
| `artifacts/e07r_gate_remediation/bad_era/` | bytes defeituosos preservados + SHA256SUMS.txt |

## 5. Conformidade com a governança E07R

- Nenhum threshold de gate alterado (QG5' 0,30 e tolerâncias QG8/QG10 preservados).
- Nenhum modelo de pesquisa promovido; `models/` volta ao estado aprovado em gate e será pinado pelo freeze E07R neste estado.
- Remediação executada **antes** do freeze, como pré-condição (#18 ⛓ #13).
- Evidência completa preservada de forma aditiva.
