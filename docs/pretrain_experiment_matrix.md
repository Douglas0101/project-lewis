# Matriz de experimentos — pré-treino Chapman (FASE 7, triagem 5 épocas)

Data: 2026-07-28 | Condições comuns: split record-disjoint (40.637/4.515), batch=64,
LR=1e-3, steps=1000/época, validação completa (704 batches), `deterministic.mode=strict`,
mesmo ambiente CPU. Métricas na melhor época (`argmin val_loss`).
Métrica primária de triagem: `val_auc_pr`; guardas: `val_auc_roc`, `val_loss`.

## Tabela comparativa

| exp | run_dir | arch | loss | seed | params | flatbuf est. | val_loss* | val_auc_roc | val_auc_pr | QG4 | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E0 | 20260728_050008 | a0 | bce | 13 | 19.933 | 25 KB | 0.4363 (BCE) | 0.8019 | 0.6279 | FAIL | concluído |
| E1 | 20260728_050641 | a1 | bce | 13 | 32.005 | 40 KB | 0.4077 (BCE) | 0.8316 | 0.6577 | FAIL | concluído |
| E2 | 20260728_051443 | a1 | bce | 42 | 32.005 | 40 KB | 0.4024 (BCE) | 0.8263 | 0.6508 | FAIL | concluído |
| E3 | 20260728_052110 | a2 | focal | 13 | 32.005 | 40 KB | 0.1026 (**focal**†) | **0.8350** | **0.6586** | FAIL | **vencedor da triagem** |
| **E3-full** | 20260728_053011 | a2 | focal | 13 | 32.005 | 40 KB | 0.4226 (BCE†) | **0.8596** | **0.7008** | FAIL (AUC PASS) | **run completo 30 ép.** |
| A0 hist. | 20260728_033533 | a0 | bce | 42 | 19.933 | 25 KB | 0.3907 (BCE) | 0.8333 | 0.6734 | FAIL | HISTORICAL_REFERENCE (30 épocas) |

\* `val_loss` na escala da loss de treino. † **Atenção**: a focal produz valores
sistematicamente menores por construção (γ=2 reduz o peso de exemplos fáceis) —
`0.1026` **não** é comparável ao threshold de BCE (0,15). Por isso o QG4 para
variantes não-BCE passou a ser avaliado sobre um monitor BCE dedicado
(`val_bce_monitor`), implementado nesta branch — o gate permanece definido
sobre BCE, thresholds inalterados.

## Calibração (FASE 8, por run)

| exp | ECE antes | ECE depois (T) | Brier antes | Observação |
|---|---|---|---|---|
| E0 | 0.0553 | 0.0233 (T=0.761) | 0.1391 | subconfiante |
| E1 | 0.0690 | 0.0474 (T=0.789) | 0.1497 | subconfiante |
| E2 | 0.0383 | 0.0396 (T=1.032) | 0.1334 | bem calibrado |
| E3 | 0.1820 | 0.0313 (T=0.305) | 0.0646 | superconfiante (típico da focal); T corrige |

## Estabilidade entre seeds (A1, bce)

E1 (seed 13) vs E2 (seed 42): Δauc_roc = 0,0053 | Δauc_pr = 0,0069 | Δloss = 0,0053
→ variância baixa; a vantagem de A1 sobre A0 (+3,0 p.p. AUC) é muito maior que a
variância de seed → ganho arquitetural real.

## Decisão da triagem

1. **A1 > A0** com folga estatística prática (+3,0 p.p. AUC, +3,0 p.p. PR, loss menor)
   dentro do orçamento (32.005 ≤ 39.866 params; 40 KB ≤ 64 KB).
2. **E3 (a2+focal) é a melhor variante** na triagem (AUC 0,8350, PR 0,6586) →
   promovida a run completo de 30 épocas (em `experiments/`, ver relatório final).
3. QG4: nenhuma variante passou na triagem de 5 épocas (esperado — gate exige
   AUC > 0,85 e BCE < 0,15; A0 histórico com 30 épocas chegou a 0,8333).
4. A4 (budget_plus) **não** executada: A1/A2 ainda não estavam próximas do QG4
   na triagem; reavaliar após o run completo de E3.

## Arquivos de evidência

Cada run contém: `provenance.json` (SHA-256), `history.json`,
`metrics_per_class.json`, `evaluation_report.json/.md`, `calibration.json`,
`backbone_pretrained.keras`, `config.json`, `training.log`, `model_summary.txt`.
