# Uso — pipeline de pré-treino Chapman (pós-correção estrutural)

## Comandos

```bash
make pretrain-smoke      # smoke de engenharia (1 época; QG4 informativo) — exit 0
make pretrain            # run completo (30 épocas; QG4 bloqueante; wrapper seguro)
make pretrain-check      # flake8 + testes rápidos do pipeline
make pretrain-validate   # valida artefatos do último run (base)
make pretrain-export-smoke  # exporta TFLite float32/INT8 + valida FlatBuffer < 64KB
```

Validação estrita (exige provenance/history/metrics_per_class):

```bash
.venv/bin/python scripts/validate_pretrain_artifacts.py --strict
```

Avaliação avançada de um run (métricas por classe, calibração, temperature):

```bash
.venv/bin/python scripts/evaluate_pretrain_run.py [experiments/<run_id>]
```

## Variantes experimentais (FASE 7)

```bash
.venv/bin/python scripts/pretrain_wrapper.py \
    --epochs 5 --steps-per-epoch 1000 \
    --architecture a1 --loss bce --seed 13
```

- `--architecture a0|a1|a2` — `a0` baseline congelada; `a1` residual; `a2` = a1 + loss de desbalanceamento.
- `--loss bce|bce_weighted|focal` — `bce_weighted` usa `pos_weight` calculado **somente no split de treino**.
- `--seed N` — reprodutibilidade (default: config = 42).
- QG4 permanece `val_auc_roc > 0.85` e `val_loss < 0.15` na melhor época (`config/pretrain_v1.0.yaml`), bloqueante fora do modo `--smoke`.
- Promoção para `models/` é opt-in (`--promote`) e **não** deve ser usada sem autorização de gate downstream (freeze E07R).

## Artefatos por run

`experiments/<ts>_pretrain_chapman/`:
`backbone_pretrained.keras`, `config.json`, `metrics.json`, `training.log`,
`model_summary.txt`, `history.json`, `metrics_per_class.json`, `provenance.json`
(+ `evaluation_report.*`/`calibration.json` após `evaluate_pretrain_run.py`).
