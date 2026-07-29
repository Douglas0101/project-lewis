# Análise QG4 — por que falhou (run 20260728_033533_pretrain_chapman)

Data: 2026-07-28 | Branch: `fix/pretrain-engineering`

## Definição formal (governança OK — não é BLOCKED_GOVERNANCE)

Fonte: `config/pretrain_v1.0.yaml` → `quality_gate.qg4`
(e referência em `AGENTS.md`, tabela QG4 / C04):

| Métrica | Operador | Threshold | Época avaliada |
|---|---|---|---|
| `val_auc_roc` (macro, multi-label) | `>` estrito | 0.85 | melhor época (`argmin val_loss`) |
| `val_loss` (BCE multi-label) | `<` estrito | 0.15 | mesma época |

Gate bloqueante: `main()` retorna exit 1 e **não** copia o backbone para `models/` quando falha.

## Valores observados (melhor época = 28)

| Métrica | Observado | Requerido | Gap | Status |
|---|---|---|---|---|
| `val_auc_roc` | 0.8333 | > 0.85 | **−0.0167** | FAIL |
| `val_loss` | 0.3907 | < 0.15 | **−0.2407** | FAIL (restrição dominante) |
| `val_auc_pr` | 0.6734 | (não faz parte do gate) | — | informativo |

## Interpretação

1. **O gap de AUC é pequeno** (1,7 p.p.): plausível de fechar com arquitetura mais estável
   (A1 residual), treino balanceado (A2) e/ou mais épocas efetivas.
2. **O gap de loss é dominante** (0.3907 vs 0.15): BCE 0.15 em 5 labels sigmoid exige
   previsões quase perfeitas *e* calibradas. O valor 0.39 indica modelo **superconfiante e
   mal calibrado** (típico de BCE sem regularização de confiança): acertos com probabilidade
   ~0.6–0.8 em vez de ~0.95+ penalizam fortemente.
3. Alavancas legítimas (sem afrouxar o gate):
   - melhor calibração pós-treino (temperature scaling — FASE 8) reduz BCE sem mexer em AUC;
   - treino com `pos_weight`/focal (A2) melhora PR-AUC das minoritárias e tende a reduzir
     overconfidence da majoritária;
   - arquitetura residual (A1) melhora fluxo de gradiente e separabilidade;
   - label smoothing: **usar com cautela** — eleva o piso teórico da BCE (alvos 0/1 viram
     ε/1−ε), podendo dificultar o `< 0.15`; se usado, documentar o trade-off.
4. Recomendação honesta: mesmo com as alavancas, `val_loss < 0.15` é ambicioso para este
   backbone/dataset. Se as variantes não atingirem, registrar como limite arquitetural e
   submeter revisão do threshold **apenas via governança** (nunca por edição local do config).

## Resposta objetiva

**QG4 falhou porque, na melhor época (28), `val_auc_roc=0.8333` não superou 0.85 e
`val_loss=0.3907` ficou muito acima de 0.15 — sendo o loss a restrição dominante, reflexo de
calibração pobre e capacidade arquitetural limitada da A0, não de falha de pipeline.**
