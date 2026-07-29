# Relatório final — correção estrutural do pré-treino Chapman

Data: 2026-07-28 | Branches: `fix/pretrain-engineering`, `feat/pretrain-architecture-v2`

## 1. Problemas originais

Run `20260728_033533_pretrain_chapman` (A0, 30 épocas): warning `ran out of data`,
erro de teardown `GeneratorDataset`, exit code corrompido, QG4 FAIL
(val_auc_roc=0.8333, val_loss=0.3907), sem proveniência.

## 2. Correções de engenharia (FASES 0–4) — FIXED

| Problema | Causa raiz | Correção | Evidência |
|---|---|---|---|
| Warning ran out of data | Falso positivo do Keras 3 (`epoch_iterator.catch_stop_iteration`) ao esgotar validação gerador com cardinalidade desconhecida | `.repeat()` + `validation_steps` explícito via `build_datasets()`; teste de regressão | 0 ocorrências nos smokes e runs (`/tmp/exp_E*.log`) |
| Teardown GeneratorDataset | Finalização do iterator no shutdown do interpretador | cleanup (`del`, `gc`, `clear_session`) + wrapper que perdoa **só** esse erro e nunca mascara falhas reais | `tests/test_pretrain_pipeline.py` (8 casos); wrapper E2E |
| Exit code | QG4 fail + teardown misturados | wrapper mapeia: 0 (sucesso real), 1 (QG4 fail), código original (crash) | `make pretrain-smoke` exit 0; E3-full exit 1 (gate real) |
| Sem proveniência | — | `provenance.json` + SHA-256 + `history.json` + `metrics_per_class.json` por run; validador `--strict` | todos os runs E0–E3-full |

Detalhes: `docs/pretrain_engineering_fix.md` | Inventário: `docs/pretrain_engineering_inventory.md`

## 3. Correções de arquitetura (FASES 5–8) — VARIANT_BETTER (AUC/PR), sem promoção

- **A0 congelada** (19.933 params pinados; `docs/pretrain_architecture_baseline_A0.md`).
- **A1_stable**: residual, sem BatchNorm (restrição TFLM do projeto — decisão conservadora).
- **A2**: A1 + focal loss (γ=2); `pos_weight`/focal só com estatísticas do split de treino.
- **Calibração**: temperature scaling pós-treino (validação), ECE/MCE/Brier por run.

## 4. Resultados

### Triagem 5 épocas (E0–E3) — `docs/pretrain_experiment_matrix.md`

| exp | arch | val_auc_roc | val_auc_pr | val_loss (BCE) |
|---|---|---|---|---|
| E0 | a0 | 0.8019 | 0.6279 | 0.4363 |
| E1 | a1 | 0.8316 | 0.6577 | 0.4077 |
| E2 | a1 (seed 42) | 0.8263 | 0.6508 | 0.4024 |
| E3 | a2 (focal) | **0.8350** | **0.6586** | n/c (focal)† |

### Run completo 30 épocas — melhor variante (E3: a2 + focal, seed 13)

Run: `experiments/20260728_053011_pretrain_chapman` | best_epoch=18

| Métrica | E3-full | A0 histórico (30 ép.) | Δ |
|---|---|---|---|
| val_auc_roc | **0.8596** | 0.8333 | **+2,6 p.p.** |
| val_auc_pr | **0.7008** | 0.6734 | **+2,7 p.p.** |
| val_loss (BCE) | 0.4226 | 0.3907 | +0,032 (pior em BCE) |
| ECE (antes → depois T) | 0.151 → **0.014** (T=0.374) | 0.055 → 0.023 | calibrável |

PR-AUC por classe (E3-full): NORM 0.989 | CD 0.556 | MI 0.625 | HYP 0.508 | STTC 0.855
(todas as classes melhoram vs triagem E0; CD/HYP seguem como classes difíceis).

† A focal produz valores de loss em escala própria (não comparável). O QG4 —
definido sobre **BCE** — passou a ser avaliado para variantes não-BCE sobre um
monitor BCE dedicado (`val_bce_monitor`), implementado nesta branch. **Sem essa
proteção, o val_loss focal=0.0914 "passaria" artificialmente o braço < 0,15 do
gate.** Thresholds e operadores do QG4: **inalterados**.

## 5. QG4 — FAIL (honesto)

| Braço | Requerido | E3-full | Status |
|---|---|---|---|
| val_auc_roc | > 0.85 | 0.8596 | **PASS** |
| val_loss (BCE) | < 0.15 | 0.4226 | **FAIL** |

Status: **QG4_FAIL_RESEARCH_CANDIDATE**. O braço de AUC passou pela primeira vez;
o braço de BCE permanece fora de alcance para esta família de arquiteturas/treino
(focal piora BCE por descalibrar; A0-BCE também não atinge 0.15). **Não promover.**
Revisão do threshold 0.15 somente via governança, com a evidência desta matriz
(ver `docs/qg4_analysis.md`).

## 6. Riscos e limitações

1. CPU-only: matriz em triagem de 5 épocas + 1 run completo (limitação registrada).
2. Split record-disjoint (Chapman não expõe `patient_id`) — `patient_disjoint: null`.
3. EarlyStopping/Checkpoint monitoram a loss de treino (focal); o gate julga o
   monitor BCE — coerente e conservador, documentado.
4. Temperatura é ajustada na validação (padrão pós-treino; não usada em treino/gate).

## 7. Recomendação

- **Manter pesquisa; não promover; não publicar.**
- Preparar fine-tuning somente se aprovado downstream (o backbone A2-full é
  candidato a feature extractor superior à A0 — decisão de gate downstream).
- Se QG4-BCE for mandatório: estudar A2 com `bce_weighted` (calibrável) + label
  smoothing como iteração futura; eventual revisão do threshold 0.15 via governança.
