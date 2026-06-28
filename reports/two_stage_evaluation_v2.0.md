# Pipeline Integrado v2.0 — Avaliação QG5'
## Estágio 1 (N vs Anormal)
- Acc: 0.7931
- F1-macro: 0.5927
- MCC: 0.1862
- Passa QG: True

### Por classe
| Classe | Se | PPV | F1 |
|--------|----|-----|----|
| N | 0.8697 | 0.8873 | 0.8784 |
| Anormal | 0.3254 | 0.2904 | 0.3069 |

## Estágio 2 (S vs V vs F)
- Acc: 0.6509
- F1-macro: 0.5185
- MCC: 0.3871
- Passa QG: True

### Por classe
| Classe | Se | PPV | F1 |
|--------|----|-----|----|
| S | 0.7340 | 0.5724 | 0.6432 |
| V | 0.6104 | 0.8474 | 0.7096 |
| F | 0.7481 | 0.1172 | 0.2026 |

## Pipeline Integrado (N, S, V, F)
- Acc: 0.7866
- F1-macro: 0.3162
- MCC: 0.1087
- FPR global: 0.0711
- Passa QG: True

### Por classe
| Classe | Se | PPV | Spe | F1 |
|--------|----|-----|-----|----|
| N | 0.8697 | 0.8950 | 0.2478 | 0.8822 |
| S | 0.2150 | 0.1427 | 0.9508 | 0.1715 |
| V | 0.1537 | 0.1886 | 0.9421 | 0.1694 |
| F | 0.2356 | 0.0228 | 0.9771 | 0.0415 |
