# configs/ml_protocol/v2 — Contratos de treino ML Protocol v2

Estes YAMLs são **contratos normativos**, não execuções. Referência: `docs/ml_protocol_v2.md`,
`docs/ablation_matrix_v2.md`, `docs/ablation_matrix_v2_appendix_D.md`.
Nenhum treino, piloto ou geração de split sem governança explícita (T10.3/T11).

## Índice

| Arquivo | Tipo | Propósito |
|---|---|---|
| `pretrain_scp_ecg_multilabel.yaml` | task profile | Pré-treino Chapman 5 superclasses (sigmoid multi-label) |
| `beat_classification_aami.yaml` | task profile | Batimentos AAMI 2 estágios (softmax; splits v4.0-PD congelados) |
| `rhythm_afib_afl.yaml` | task profile | Ritmo AFIB/AFL em episódio (AFDB; futuro/contexto — decisão D3) |
| `teacher_resnet1d.yaml` | teacher (trilha D) | Arquétipos ResNet1D 500k/1M para D1/D2 (offline, nunca produção) |
| `teacher_inception1d.yaml` | teacher (trilha D) | Arquétipo Inception1D ~5M para D4 (condicional + aprovação humana) |
| `distillation_kd.yaml` | protocolo KD | Destilação sigmoid por rótulo (α/τ sweeps; student A2-full/64k) |
| `split_paired_v2.yaml` | split spec | `chapman-record-disjoint-paired-v2` — spec normativa; **geração pendente** |

## Regras transversais

1. Early stopping SEMPRE por métrica equalizada (`val_macro_pr_auc` pré-treino, `val_macro_f1`
   batimentos) — nunca `val_loss` cru.
2. Calibração obrigatória (`temperature_scaling`, `n_bins=15`, split de calibração separado).
3. Thresholds ajustados SOMENTE em calibration, aplicados congelados ao teste.
4. `smote_on_validation`, `smote_on_test`, `test_threshold_tuning` proibidos em todos os perfis.
5. Teachers não têm restrições TFLM e não geram artefatos de produção; students herdam QG6/QG9.
6. Configs legados de treino vivem em `config/` (singular) e seguem válidos para o pipeline
   atual — este diretório normatiza apenas o protocolo v2.

Validação estrutural: `tests/test_ml_protocol_configs.py`.
