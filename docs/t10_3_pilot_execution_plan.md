# Plano de Execução de Pilotos T10.3

**Versão:** v1.0.0 · **Data:** 2026-08-02 · **Branch:** `develop` @ `f740ce5`
**Status:** PLANO — **não autoriza execução**
**Referências:** `docs/ablation_matrix_v2.md` · `docs/ablation_matrix_v2_appendix_D.md` ·
`configs/ml_protocol/v2/` · `reports/ml_protocol_v2/pretrain_reconciliation.md`

> Este documento estima compute, define ordem de execução e critérios de parada.
> A execução de qualquer célula exige:
> 1. Split pareado v2 gerado (aprovação humana);
> 2. Autorização de compute do owner;
> 3. Revisão humana para promoção de qualquer candidato.

---

## 1. Escopo da primeira rodada

```text
Fase 1 (controle H7 — obrigatória):
  C0 — A0 + BCE + seed 13 + split pareado
  C1 — A1 + BCE + seed 13 + split pareado
  C2 — A1 + Focal γ=2 + seed 13 + split pareado (reproduz A2-full sob protocolo)
  C3 — A0 + Focal γ=2 + seed 13 + split pareado

Fase 2 (destilação baseline — obrigatória):
  D0 — Self-distillation A2-full → A2-full

Fase 3 (condicional, só se Fases 1–2 mostrarem sinal):
  F1–F5 — Ablação de γ (trilha F)
  K1–K3 — Ablação de calibração (trilha K; pós-treino sobre C2)
  T1–T3 — Ablação de threshold (trilha T; pós-treino sobre C2)
  D1–D2 — Teacher 500k/1M (trilha D)
```

**Regra dura:** Fase 1 e D0 são obrigatórias antes de qualquer outra fase. Sem C1, H7 não é
testada (nenhuma conclusão sobre arquitetura é válida); sem C3, nenhuma conclusão sobre loss;
sem D0, o protocolo KD não está validado. Referências de comparação (T9.3, split seed 13 do
A2-full — os pilotos medem-se contra elas até o split pareado existir): macro PR-AUC 0,7065 ·
CD 0,5561 · HYP 0,5078 · ECE pós-T 0,0152 · ECE NORM=0 0,2167.

## 2. Ordem de execução e dependências

| Ordem | Célula | Depends on | Compute ~ (CPU) | Gate de continuação |
|---:|---|---|---:|---|
| 1 | C0 | split pareado | 35 min | — |
| 2 | C1 | split pareado | 40 min | C1 macro PR-AUC ≥ C0 (senão H7 comprometida, §4) |
| 3 | C2 | split pareado | 40 min | C2 macro PR-AUC ≥ C1 e ≈ 0,70 (sanidade A2-full) |
| 4 | C3 | split pareado | 35 min | — (leitura arch×loss) |
| 5 | D0 | C2 (pesos student_init) | 40 min | D0 macro PR-AUC ≥ C2 (KD válido) |
| 6 | F1 | split pareado | 40 min | só se H1 não refutada |
| 7 | F2 | split pareado | 40 min | só se H1 não refutada |
| 8 | K1–K3 | C2 (predições) | 10 min | só se H2 não refutada |
| 9 | T1–T3 | C2 (predições) | 5 min | só se H4 não refutada |
| 10 | D1 | split pareado | 1,7 h | só se D0 válido |
| 11 | D2 | split pareado | 2,7 h | só se D1 CD ≥ +2 p.p. vs C2 |

**Regra de parada mestra:** se C1 não melhorar sobre C0 (Δ macro PR-AUC < 1 p.p. com IC95
sobreposto), H7 perde força e a matriz deve ser revisada **antes** de qualquer fase seguinte.

## 3. Compute estimado (ESTIMATIVA ancorada: 32k/30ép ≈ 33–40 min CPU neste host)

| Fase | Células | CPU-h ~ | GPU-h ~ |
|---|---:|---:|---:|
| Fase 1 (C0–C3) | 4 | 2,5 | 0,5 |
| Fase 2 (D0) | 1 | 0,7 | 0,2 |
| Fase 3 (F1–F5, K1–K3, T1–T3) | 11 | 4,0 | 1,0 |
| Fase 4 (D1–D2) | 2 | 4,4 | 1,0 |
| **Total primeira rodada** | **18** | **~11,6** | **~2,7** |

Fora da primeira rodada: D3 (12 leads — pipeline C02 novo), D4 (5M — aprovação separada), D5,
e as trilhas R/B/P da matriz v2. Avaliação canônica incluída nas estimativas (~4 min/run).

## 4. Critérios de parada por hipótese

| Hipótese | Critério de parada/refutação | Ação se refutada |
|---|---|---|
| H7 (atribuição) | C1 Δ macro PR-AUC < 1 p.p. vs C0 (IC sobreposto) | revisar matriz; ganho não é arquitetura |
| H1 (γ alto) | F1/F2 com ECE pós-T ≥ F3 (γ=2) | γ=2 não é o problema; focar em dados/sampling |
| H2 (calibração NORM=0) | K1/K2 com ECE NORM=0 ≥ 0,15 | T global suficiente; abandonar estratificação |
| H4 (threshold) | T1 ganho < 4 p.p. macro F1 vs T0 (prospectivo) | threshold tuning não é prioridade |
| H8 (capacidade) | D1 CD PR-AUC < 0,57 (D2 < 0,60) | capacidade não é gargalo; redirecionar para P |

## 5. Critérios de promoção piloto → candidato

Os 10 critérios da matriz v2 §11 + os 12 do adendo D §6, mais governança:

```text
1. Todos os critérios técnicos atendidos (PR-AUC, ECE, INT8, TinyML)
2. Reconciliação com A2-full registrada em evaluation_v2/
3. Revisão humana aprovada
4. Freeze de hash do candidato
5. Documentação de risco explícita: QG4-BCE permanece FAIL até RFC T9.5
```

## 6. Aprovações necessárias (bloqueiam o início)

```text
[ ] Split pareado v2 gerado (aprovação humana + freeze de hash)
[ ] Compute autorizado pelo owner (~11,6 CPU-h primeira rodada)
[ ] D4 (5M params) aprovado separadamente (se alcançado)
[ ] Promoção de qualquer candidato aprovada (revisão humana)
[ ] RFC T9.5 concluída (para promoção formal em T11)
```

## 7. Riscos e mitigação

| Risco | Prob. | Impacto | Mitigação |
|---|---|---|---|
| Split pareado não gerado/aprovado | Baixa | Alto | spec normativa pronta (`configs/ml_protocol/v2/split_paired_v2.yaml`) |
| Compute excede estimado | Média | Médio | budget test (T9.4); D4 condicional; execução por fases com relatório parcial |
| C1 não melhora (H7 enfraquece) | Média | Alto | regra de parada mestra (§2); revisão de matriz |
| Teacher não converge | Baixa | Médio | GroupNorm + warmup + AdamW (configs v2) |
| KD diverge (α/τ ruins) | Média | Médio | sweeps {0.5,0.7,0.9}×{2,4,6}; ES por `val_macro_pr_auc` |
| ECE NORM=0 não melhora | Alta | Médio | K1–K3; se falhar, H2 refutada (barato saber) |
| QG4-BCE continua FAIL | Certa | Alto | RFC T9.5; promoção formal bloqueada até decisão |
| Desvio acidental de split | Baixa | Alto | avaliador marca `NON_COMPARABLE` automaticamente |

## 8. Monitoramento e logging

Cada piloto gera:

```text
experiments/<run_pilot>/
  config.yaml            # snapshot do config efetivo (configs/ml_protocol/v2/)
  training_log.csv       # por época
  evaluation_v2/         # conjunto canônico schema 2.0 (7 artefatos)
  reconciliation.json    # vs A2-full
  pilot_status.json      # {"status": "PILOT"} — NUNCA CANDIDATE sem promoção
```

Avaliação obrigatória ao final de cada piloto:

```bash
uv run python -m src.evaluation.canonical_evaluator \
  --run-dir experiments/<run_pilot> \
  --task-profile pretrain_scp_ecg_multilabel \
  --split-name chapman-record-disjoint-paired-v2 \
  --output-dir experiments/<run_pilot>/evaluation_v2 \
  --temperature-source fit \
  --threshold-policy max_f1_per_class \
  --n-bins 15 --seed 13
```

(`protocol_status` PROSPECTIVE exige `--calibration-predictions` do split de calibração —
obrigatório para claims; RETROSPECTIVE vale apenas como análise.)

## 9. Rollback

Se qualquer piloto corromper estado ou divergir:

```text
1. Não promover.
2. Marcar run como REJECTED em pilot_status.json.
3. Não modificar models/, splits ou artefatos congelados.
4. Registrar falha em reports/t10_3_pilot_failures.md.
5. Revisar a matriz antes de continuar a fila.
```

## 10. Declarações finais

> Este plano não autoriza execução de treinos.
> Nenhum split foi gerado.
> Nenhum compute foi consumido.
> A execução de T10.3 exige as aprovações explícitas da seção 6.
> O QG4-BCE permanece FAIL; promoção bloqueada até a RFC T9.5.
