# E07R — bloqueio pré-freeze por gates de integridade

**Data:** 2026-07-26
**Decisão:** `BLOCKED`
**Escopo:** E07R / E06.5-PD / E07-PD

## Resumo

A remediação patient-disjoint foi implementada e os testes focados passaram, mas a missão foi interrompida antes da publicação do freeze e antes de qualquer treinamento porque a suíte completa contém dois gates vermelhos reproduzíveis.

Nenhuma célula E06.5-PD ou E07-PD foi executada. Nenhum modelo foi promovido e nenhum gate foi relaxado.

## Evidência que passou

- `make lint`: **PASS** (flake8, mypy e bandit; zero achados).
- Pyright nos arquivos E07R alterados: **0 erros, 0 warnings**.
- Suíte focada E07R/PD: **28 passed**.
- Custódia r5: 33.001 linhas, binding ordenado NPZ↔Parquet `PASS`.
- Mapping: 119 records / 76 pacientes autenticados; MIT-BIH 201/202 no mesmo `patient_id`.
- Splits: 5 outer + 20 inner, patient-disjoint e record-disjoint; nenhum zero estrutural.
- Testes obrigatórios cobrem mismatch NPZ↔Parquet, mapping inválido, overlap, mutação pós-freeze simulada, escrita proibida, uso de split legado, promoção para `models/`, samplers train-only e tentativa de E07-PD antes de E06.5-PD.

## Gates que falharam

### 1. QG8 / bit-exatidão TFLM

Teste:

```text
tests/test_tflm_bitexact.py::TestTflmBitexact::test_bitexact_vs_firmware
```

Resultado reproduzido isoladamente:

```text
Numero de beats diverge: firmware=0, python=3
```

### 2. QG5' Stage 1

Teste:

```text
tests/test_two_stage_qg5.py::test_two_stage_qg5_end_to_end
```

Resultado:

```text
Recall(Anormal)=0.0661 < 0.30
```

Resumo da suíte completa:

```text
2 failed, 984 passed, 205 warnings
```

## Estado fail-closed

- `experiments/stage2_v2.4_research/integrity/e07r_freeze_manifest.json`: **não publicado**.
- E06.5-PD: **0/100 células**.
- E07-PD: **0/150 células**.
- `models/`: sem promoção E07R.
- Publicação externa: não autorizada.
- Próxima ação: resolver ou obter disposição explícita de governança para os dois gates vermelhos; até lá, freeze e treinamento permanecem bloqueados.
