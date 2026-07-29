# Inventário — Engenharia do Pré-treino Chapman (FASE 0)

Data: 2026-07-28 | Branch: `fix/pretrain-engineering` | Commit base: `48931a7`

> Nota de mapeamento: a missão referencia `src/camada04/*`; no repositório real
> os módulos vivem em `src/models/*` (`chapman_dataset.py`, `backbone_1d.py`,
> `pretrain_chapman.py`). Este inventário usa os caminhos reais.

## Arquivos envolvidos

| Arquivo | Papel |
|---|---|
| `Makefile` (target `pretrain`, linha ~437) | Entry point: `.venv/bin/python -m src.models.pretrain_chapman` |
| `src/models/pretrain_chapman.py` | CLI + loop de treino + QG4 (best-epoch) |
| `src/models/chapman_dataset.py` | Split por registro, `tf.data.Dataset.from_generator`, `estimate_n_segments` |
| `src/models/backbone_1d.py` | A0: 3×Conv1D+MaxPool → GAP → Dense → Dropout → Dense(5) |
| `src/data/chapman_labels.py` | SNOMED-CT → 5 superclasses SCP-ECG (multi-hot) |
| `config/pretrain_v1.0.yaml` | Hiperparâmetros + thresholds QG4 |
| `data/catalog/dataset_catalog.jsonl` | Catálogo de registros (fonte do split) |
| `data/processed/chapman/*_II.npy` | Sinais processados (5000 amostras, lead II) |

## Fluxo de dados

1. `chapman_train_val_split(val_ratio=0.1, seed=42)` → split **por registro** (40.637 train / 4.515 val).
2. `from_generator` yielda segmentos de 500 amostras (10/registro), multi-hot do registro.
3. Treino: ordem de registros re-embaralhada por passada (`seed + iteração`), `.batch(64)`, `.prefetch(AUTOTUNE)`, `.repeat()` no `main`.
4. Validação: ordem determinística (catálogo), **sem** `.repeat()`, `validation_steps=None` → passada completa (~704 batches).

## Fluxo de treino

- Modelo A0 multilabel: 19.933 params, FlatBuffer estimado 25 KB, saída sigmoid.
- Otimizador Adam lr=1e-3; loss `binary_crossentropy`; métricas AUC-ROC/AUC-PR (multi_label).
- Callbacks: EarlyStopping(val_loss, p=5, restore_best), ReduceLROnPlateau(p=3, f=0.5), ModelCheckpoint(best val_loss), CSVLogger.
- `steps_per_epoch=1000` (config), `validation_steps=null` (val completo), `epochs=30`.
- QG4 avaliado na **melhor época** (`argmin val_loss`).

## Definição do QG4

Fonte: `config/pretrain_v1.0.yaml` → `quality_gate.qg4`:
- `min_val_auc_roc_macro: 0.85` (estrito `>`)
- `max_val_loss: 0.15` (estrito `<`)
- Gate bloqueante: `main()` retorna 1 se falhar; só copia para `models/` se passar.

## Pontos frágeis identificados

1. Warning `Your input ran out of data` — causa a confirmar no smoke (FASE 2); hipótese principal: interação `from_generator` + exaustão de iterator em boundary de época.
2. Erro de finalização `GeneratorDataset iterator: Python interpreter state is not initialized` — teardown do `from_generator` no shutdown do interpretador; tratado via cleanup + wrapper (FASE 3).
3. Exit code do `make pretrain` reflete QG4 (correto), mas erros de teardown podem mascarar conclusão real — wrapper diferencia (FASE 3).
4. Sem `provenance.json`, sem SHA-256, sem `history.json`, sem métricas por classe (FASE 4).
5. Determinismo parcial: seeds fixadas, mas oneDNN/TF nondeterminism não controlado (FASE 4, modo strict).
6. Estimativa de steps assume comprimento uniforme de registro (documentado em `estimate_n_segments`).

## Riscos

- CPU-only: experimentos comparativos completos são caros (~45–60 min/run) → triagem por smoke (FASE 7).
- `from_generator` + `itertools.count` no closure: contador é por-dataset; seguro, mas documentado.
- BatchNorm é proibido pelas restrições TFLM do projeto (ver `backbone_1d.py` docstring) → A1 usará residual sem BN (decisão registrada na FASE 6).

## Dependências

- TensorFlow 2.21 (Keras 3, `TF_USE_LEGACY_KERAS=0`), NumPy, PyYAML.
- Catálogo + sinais processados (C01/C02) presentes localmente.

## Comandos make atuais

- `make pretrain` → `python -m src.models.pretrain_chapman` (30 épocas, QG4 bloqueante).

## Artefatos gerados por run

- `experiments/<ts>_pretrain_chapman/`: `backbone_pretrained.keras`, `config.json`, `metrics.json`, `model_summary.txt`, `training.log`.

## Congelamento (FASE 0)

- Run histórico `20260728_033533_pretrain_chapman` registrado como `HISTORICAL_REFERENCE` em seu `freeze_manifest.json` (SHA-256 reais).
- Melhor época 28: val_loss=0.3907, val_auc_roc=0.8333, val_auc_pr=0.6734, QG4=FAIL.
