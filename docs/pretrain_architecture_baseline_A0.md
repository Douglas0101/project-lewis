# Baseline arquitetural congelada — A0_baseline_19k

Data do congelamento: 2026-07-28 | Branch de origem: `fix/pretrain-engineering`
Status: **CONGELADA** — nenhuma alteração futura pode sobrescrever esta referência.

## Identidade

| Campo | Valor |
|---|---|
| Nome | `A0_baseline_19k` |
| Builder | `src/models/backbone_1d.py::build_backbone_1d_multilabel` (wrapper congelado em `src/models/backbones/a0_baseline.py`) |
| Run de referência | `experiments/20260728_033533_pretrain_chapman` (`HISTORICAL_REFERENCE`, ver `freeze_manifest.json`) |
| Pesos de referência | `backbone_pretrained.keras` sha256 `73f8a160…b22e50` |
| Parâmetros | **19.933** (pinado por `tests/test_backbone_budget.py`) |
| FlatBuffer estimado | **25 KB** (INT8, est. 1,3 B/param) |

## Diagrama textual

```
Input(500, 1)                          # 1000 ms @ 500 Hz, lead II, z-score
  → Conv1D(16, k=7, relu, same)  → MaxPool1D(2)   # 500 → 250
  → Conv1D(40, k=5, relu, same)  → MaxPool1D(2)   # 250 → 125
  → Conv1D(80, k=3, relu, same)  → MaxPool1D(2)   # 125 → 62
  → GlobalAveragePooling1D()                       # 80
  → Dense(80, relu)                                # embedding
  → Dropout(0.3)
  → Dense(5, sigmoid)                              # NORM, CD, MI, HYP, STTC (multi-label)
```

- Loss: `binary_crossentropy` | Otimizador: Adam lr=1e-3
- Callbacks: EarlyStopping(val_loss, p=5, restore_best), ReduceLROnPlateau(p=3, f=0.5), ModelCheckpoint(best)
- Restrições TFLM honradas: sem BatchNorm/LSTM/SeparableConv/attention (ver docstring `backbone_1d.py`).

## Métricas de referência (run histórico, melhor época = 28)

| Métrica | Valor |
|---|---|
| val_loss | 0.3907 |
| val_auc_roc | 0.8333 |
| val_auc_pr | 0.6734 |
| QG4 | **FAIL** (requer val_auc_roc > 0.85 **e** val_loss < 0.15) |

## Limitações conhecidas

1. Sem conexões residuais → fluxo de gradiente limitado; treino sensível a LR.
2. Calibração pobre (BCE 0.39 muito acima do gate) — saídas superconfiantes.
3. PR-AUC modesto (0.67) — classes minoritárias (CD ~16%) sofrem.
4. GAP logo após 3 blocos conv → capacidade de representação temporal limitada.

## Regra de comparação

Toda variante (A1/A2/A4) deve ser comparada contra esta linha-base nas mesmas
condições (dataset, split, seed-base, avaliação, QG4). Nenhuma variante
substitui A0 sem evidência registrada em `docs/pretrain_experiment_matrix.md`.
