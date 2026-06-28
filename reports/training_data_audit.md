# Relatório de Auditoria de Dados de Treinamento

**Gerado em:** 2026-06-20T04:38:49.139739+00:00
**Status geral:** PASS
**Registros inspecionados:** 2224
**Batimentos inspecionados:** 473248

## Checks por categoria

| Categoria | Check | Status | Count | Detalhes |
| :--- | :--- | :--- | ---: | :--- |
| structural | catalog_valid | PASS | 88974 | 88974 registros no catalog |
| structural | lineage_npy_consistency | PASS | 0 | 88974 registros OK, 0 faltando, 0 .npy órfãos |
| statistical | no_nan_inf | PASS | 0 |  |
| statistical | no_flatline | PASS | 0 |  |
| statistical | zscore_sanity | WARNING | 539 | mean_fora_tol=48, std_fora_tol=491 |
| statistical | checksum_integrity | PASS | 0 |  |
| statistical | duration_in_range | WARNING | 23 |  |
| statistical | low_zero_ratio | PASS | 0 |  |
| annotations | aami_mapping_valid | PASS | 0 | Classes: {'N': 406646, 'S': 16939, 'V': 37192, 'F': 1045, 'Q': 11426} |
| privacy | no_pii_in_lineage | PASS | 0 | PII permitido apenas no catalog de origem |
| balance | min_class_samples_for_smote | PASS | 0 | Classes com < 6 amostras: [] |
| split | group_kfold_feasible | PASS | 224 | Registros/pacientes disponíveis para GroupKFold: 224 |

## Resumo por dataset

| Dataset | Inspecionados | NaN/Inf | Flatline | Distribuição AAMI |
| :--- | ---: | ---: | ---: | :--- |
| chapman | 1000 | 0 | 0 | N=0, S=0, V=0, F=0, Q=0 |
| mitdb | 48 | 0 | 0 | N=90631, S=2781, V=7236, F=803, Q=9096 |
| svdb | 78 | 0 | 0 | N=162339, S=12198, V=9943, F=23, Q=2291 |
| afdb | 23 | 0 | 0 | N=0, S=0, V=0, F=0, Q=0 |
| incart | 75 | 0 | 0 | N=153676, S=1960, V=20013, F=219, Q=39 |
| ptbxl | 1000 | 0 | 0 | N=0, S=0, V=0, F=0, Q=0 |

**DLQ:** `/home/douglas-souza/PycharmProjects/Project-Lewis/data/.dlq/training_data_audit_failures.jsonl`

---
_Relatório gerado por `scripts/audit_training_data.py`._