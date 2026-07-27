# E07R — relatório consolidado de evidência

**Data:** 2026-07-26
**Responsável:** `AUTONOMOUS_GOVERNANCE_PREAUTH`
**Modo:** `AUTONOMOUS_PREAUTHORIZED`
**Pacote canônico:** `experiments/stage2_v2.4_research/integrity/e07r_evidence_package.json`

## 1. Resultado científico principal

```text
NO_ROBUST_GAIN — H6-PD não supera baseline sob splits patient-disjoint v4.0
```

| Representação | F1(F) OOF | F1-macro OOF | Elegibilidade |
| --- | ---: | ---: | --- |
| baseline (base16) | **0,6325** | **0,7067** | — |
| H6 (pré-comprometida) | 0,4724 | 0,6700 | **inválida para E07-PD** |
| H11 (descritiva) | 0,4942 | 0,6801 | não substitui H6 |
| H12 (descritiva) | 0,5389 | 0,6831 | não substitui H6 |

Delta H6 − baseline F1(F): **−0,1601**, IC95% bootstrap por paciente **[−0,3983; +0,1534]** (10.000 repetições, seeds aninhadas). Razão de decisão: `H6_GAIN_OVER_BASELINE_NOT_ROBUST`.

Sob avaliação patient-disjoint autenticada, o ranking da era record-disjoint não se reproduz: o baseline base16 supera todas as representações com templates em F1(F). A auditoria de sampling E07-PD torna-se cientificamente inválida sob o pré-registro e foi formalmente bloqueada (0/150 células).

## 2. Cadeia de execução

| Task | Resultado | Evidência |
| --- | --- | --- |
| #18 resolver falhas full-suite | `RESOLVED` | `docs/e07r_gate_remediation_report.md`; QG5' recall 0,3094 ≥ 0,30; QG8/QG10 verdes |
| #13 gates de integridade E07R | `PUBLISHED` | freeze manifest `ba1c4aa1…`, 101 pins, preflight 9/9 PASS |
| #14 revalidar E06.5-PD | `COMPLETE` | 100/100 células, 0 falhas; seleção `NO_VALID_CANDIDATE` |
| #15 auditoria E07-PD | `BLOCKED` | `E07_PD_blocked_20260726.json`; exit 10; 0/150 células |
| #16 pacote de evidência | este documento + JSON canônico | hashes de todos os artefatos |
| #17 validação final | ver seção 5 | suíte completa + e2e + checkpoint final |

## 3. Desvios e correções registradas

1. **Artefatos Stage 1 v2.0 corrompidos (04/jul e 11/jul):** restauração byte-a-byte com proveniência tripla (experimento de origem, git, consistência float32↔int8). Detalhes em `docs/e07r_gate_remediation_report.md`. Bytes defeituosos preservados em `artifacts/e07r_gate_remediation/bad_era/`.
2. **Bug de contrato no loader full-template (pré-treino):** `load_full_template_dataset` rejeitava o layout canônico `(n, 500, 1)` do r4 autorizado. Correção em `src/stage2_research/data.py` com testes de regressão (`tests/test_stage2_full_template_loader_v4.py`, 3 testes). O freeze foi republicado (primeira publicação substituída; preflight `e065pd-audit-v1` preservado como evidência histórica do estado anterior).
3. **Flakiness pré-existente Renode:** `test_fidelity` e `test_fault_injection_dummy_spi` falham intermitentemente sob carga em suíte completa; verdes isoladamente. Não relacionados ao escopo E07R (commits 4a57a02/79ad9ea documentam a classe do problema).

## 4. Conformidade com a governança

- `publication_ready = false`
- `models_untouched = true` (nenhum modelo de pesquisa promovido; `models/` restaurado ao estado pinado aprovado em gate e congelado por 101 pins)
- `gates_relaxed = false` (nenhum threshold alterado: QG5' 0,30; QG8 5 LSB; QG10 0,99/0,94; F1(F) ≥ 0,50; bootstrap/Holm conforme pré-registro)
- `next_authorized_stage = NONE`
- Leakage: zero overlap patient/record em outer e inner; 201/202 no mesmo paciente; SVDB fora do fitting confirmatório; quarentena legada preservada
- Split de teste nunca usado para fitting de sampler/pesos/threshold/seleção

## 5. Validação

| Gate | Resultado |
| --- | --- |
| Preflight FREEZE_VALIDATION (9 checks) | PASS (inicial e pós-missão, report `dbaf5f73…`) |
| Preflight E06_5_PD (run-id e065pd-audit-v2) | PASS |
| E07-PD sem candidato | BLOCKED (esperado, exit 10) |
| Testes focados E07R/PD + regressão loader | 17 passed |
| `make lint` | PASS (0 achados) |
| Pyright arquivos alterados | 0 errors, 0 warnings |
| Suíte completa `make test` | **989 passed, 0 failed** (562 s) |
| `make test-e2e` | **17 passed**, 972 deselected (233 s) |

## 6. Hashes-chave

```text
freeze_manifest        ba1c4aa1b109cb9a11c3facde215624a5674853caf8d19faeb67cf75c0f28171
pd_protocol_manifest   d33079cbe56f5721e238573370f12f79f901e3cd625d9c1fb185f8008177d0bf
h_star_pd_selection    bc67148a2f2a1ff3fb216d0617d1318ec6875a6de847c3034d2a5b4f7e079b5e
evidence_package       0184a5eef3bac0d517ff11359aee757ec0874b0744eebf3ca2f6514d533a9287
final_checkpoint       experiments/stage2_v2.4_research/E07R_final_checkpoint_20260726.json
```
