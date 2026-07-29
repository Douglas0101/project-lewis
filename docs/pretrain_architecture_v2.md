# Arquitetura v2 — variantes de backbone para pré-treino Chapman (FASES 6–7)

Data: 2026-07-28 | Branch: `feat/pretrain-architecture-v2`

## Família

| Variante | Tipo | Params | FlatBuffer est. | Status |
|---|---|---|---|---|
| **A0** `a0_baseline` | 3×Conv+MP → GAP → Dense → Dropout → sigmoid | 19.933 (pin) | 25 KB | **CONGELADA** (`docs/pretrain_architecture_baseline_A0.md`) |
| **A1** `a1_stable` | stem conv + 3 blocos residuais + GAP + Dropout + sigmoid | 32.005 | 40 KB | implementada |
| **A2** `a2` | = A1 (arquitetura) + loss de desbalanceamento | 32.005 | 40 KB | implementada (training layer) |
| **A4** `budget_plus` | leve aumento de capacidade | — | — | **não executada** (só se A1/A2 próximas do QG4) |

Orçamento (pinado em `tests/test_backbone_budget.py`): params ≤ 2×A0 (39.866),
FlatBuffer INT8 ≤ 64 KB, conversão TFLite smoke obrigatória.

## A1_stable — decisões arquiteturais

```
Input(500,1)
→ Conv1D(16,k=7,relu) → MaxPool(2)          # 250
→ ResBlock(16,k=5)   → MaxPool(2)           # 125
→ ResBlock(32,k=5)   → MaxPool(2)           # 62
→ ResBlock(64,k=3)                          # 62
→ GlobalAveragePooling1D → Dropout(0.3) → Dense(5, sigmoid)
ResBlock: Conv-relu → Conv → Add(skip; projeção 1×1 se canais diferem) → relu
```

- **Residual sem BatchNorm**: as restrições TFLM do projeto
  (`src/models/backbone_1d.py`, docstring) proíbem BatchNorm/LSTM/SeparableConv/
  attention. A sugestão original de A1 incluía BN; optou-se pela variante
  conservadora (regra 15) — estabilidade via skip connections, mantendo
  compatibilidade total com INT8/TFLite (Conv1D, Add, ReLU, MaxPool, GAP,
  Dense, Dropout — todos built-ins TFLite).
- Projeção 1×1 nos skips com mudança de canal (res2, res3).
- Saída sigmoid + BCE (multi-label), idêntica à A0 — comparabilidade direta.

## A2 — desbalanceamento (training layer, não arquitetura)

- `bce_weighted`: BCE com `pos_weight` por classe, calculado **somente** no
  split de treino (`estimate_pos_weights`, clip [1, 10]).
- `focal`: `BinaryFocalCrossentropy(gamma=2.0)`.
- Medido no split de treino (n=40.637): pos_weight = [1.0, 5.20, 2.42, 3.61, 2.69]
  para [NORM, CD, MI, HYP, STTC] — CD é a classe mais reforçada (mais rara).
- Validação **sem** oversampling/reweighting; thresholds fora do treino.

## A3 — calibração (pós-treino, FASE 8)

- Temperature scaling escalar ajustado na validação (NLL), ECE/MCE/Brier antes
  e depois; relatórios `calibration.json`/`evaluation_report.*` por run.

## A4 — gatilho de execução

Só executar se A1/A2 ficarem **próximas** do QG4 (val_auc_roc ≥ 0,83 com
val_loss em queda). Registrar aumento de params/FlatBuffer; não promover se
violar a restrição embedded.

## Comparabilidade (regras da FASE 7)

Mesmo dataset, split (record-disjoint, val_ratio=0.1), batch=64, LR=1e-3,
avaliação completa de validação, QG4 inalterado, `deterministic.mode=strict`.
Seeds: 13 (base) e 42 (sensibilidade). Métrica primária de triagem:
`val_auc_pr` (apropriada a desbalanceamento), com `val_auc_roc`/`val_loss` como
guardas; registrada em `docs/pretrain_experiment_matrix.md`.
