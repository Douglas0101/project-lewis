# Avaliação avançada — 20260728_053011_pretrain_chapman

## Calibração (macro)
- ECE antes: 0.1508
- Brier antes: 0.1327
- Temperature: 0.374 (NLL 0.4317 → 0.3417)
- ECE depois: 0.0152

## Métricas por classe
| classe | support | auc_roc | auc_pr | P | R | F1 |
|---|---|---|---|---|---|---|
| NORM | 33750 | 0.9742 | 0.9887 | 0.945 | 0.966 | 0.956 |
| CD | 7350 | 0.8249 | 0.5561 | 0.767 | 0.260 | 0.388 |
| MI | 13060 | 0.7872 | 0.6252 | 0.674 | 0.389 | 0.493 |
| HYP | 9930 | 0.8038 | 0.5078 | 0.560 | 0.337 | 0.421 |
| STTC | 12380 | 0.9296 | 0.8548 | 0.849 | 0.733 | 0.787 |

> Thresholds alternativos: análise apenas — não aplicados a gates.