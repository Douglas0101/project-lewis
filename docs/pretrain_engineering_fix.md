# Correção estrutural — engenharia do pré-treino Chapman (FASES 0–4)

Data: 2026-07-28 | Branches: `fix/pretrain-engineering` → `feat/pretrain-architecture-v2`

## Problemas originais (run 20260728_033533_pretrain_chapman)

1. Warning `Your input ran out of data; interrupting training`.
2. Erro de finalização `GeneratorDataset iterator: Python interpreter state is not initialized`.
3. `make pretrain` com exit não-zero mesmo quando o treino concluía.
4. QG4 fail (val_auc_roc=0.8333, val_loss=0.3907).
5. Sem proveniência, hashes, history ou métricas por classe.

## Correções

### FASE 0 — Inventário e congelamento

- `docs/pretrain_engineering_inventory.md` (fluxo, pontos frágeis, riscos).
- Run histórico congelado como `HISTORICAL_REFERENCE` com SHA-256 reais:
  `experiments/20260728_033533_pretrain_chapman/freeze_manifest.json`.

### FASE 1 — Diagnóstico QG4

- `docs/qg4_analysis.md`: gate formalmente definido (`val_auc_roc > 0.85` **e**
  `val_loss < 0.15`, melhor época). Falha por gap de AUC (−1,7 p.p.) e,
  dominantemente, de loss (−0,24; calibração pobre). Threshold **não** alterado.

### FASE 2 — Pipeline de dados

- **Causa do warning identificada**: falso positivo do Keras 3
  (`epoch_iterator.catch_stop_iteration`) ao esgotar um dataset de validação
  baseado em gerador (cardinalidade desconhecida) com `validation_steps=None`.
- **Correção**: `build_datasets()` (`src/models/pretrain_chapman.py`) aplica
  `.repeat()` em treino **e** validação, com `validation_steps` explícito
  (= n_val estimado → validação completa, sem viés de prefixo); logging de
  cardinalidade (`train_batches`, `val_batches`, batch, steps).
- Teste de regressão: `test_fit_one_epoch_has_no_ran_out_of_data_warning`.
- Verificado em smoke real: **0 ocorrências** do warning (antes: presente).

### FASE 3 — Encerramento e exit code

- Cleanup explícito no fim do `main()`: `del` datasets/modelo, `gc.collect()`,
  `tf.keras.backend.clear_session()`.
- `scripts/pretrain_wrapper.py`: executa o treino, valida artefatos e mapeia o
  exit code — perdoa **somente** o erro conhecido de teardown quando o treino
  concluiu e o QG4 passou; **nunca** mascara QG4 fail, crash ou artefato ausente.
- `scripts/validate_pretrain_artifacts.py` (+ `--strict` para artefatos FASE 4).
- `make pretrain` agora usa o wrapper; `make pretrain-smoke` = checagem de
  engenharia (QG4 informativo, não bloqueante **somente** no smoke).
- Promoção para `models/` virou opt-in (`--promote`) — protege o freeze E07R.
- Testes: `tests/test_pretrain_pipeline.py` (8 casos, incl. não-mascaramento).

### FASE 4 — Reprodutibilidade e proveniência

- `deterministic.mode: strict` no config (op determinism + seeds); `fast` disponível.
- `src/models/pretrain_provenance.py`: por run, grava
  `provenance.json` (git, versões, dataset, modelo, treino, métricas, QG4,
  SHA-256 de modelo/config/history/metrics_per_class),
  `history.json` (histórico completo) e
  `metrics_per_class.json` (ROC/PR-AUC, P/R/F1@0.5, support por superclasse,
  computados na validação — somente avaliação).
- Testes: `tests/test_pretrain_artifacts.py` (+ integração `slow` com wrapper).

## Estado após FASES 0–4

| Item | Antes | Depois |
|---|---|---|
| Warning ran out of data | presente | **ausente** (0 ocorrências) |
| Exit code | corrompido por teardown | mapeado pelo wrapper (0/1 correto) |
| Cardinalidade | não registrada | logada por run |
| Provenance/hashes/history/per-class | inexistentes | gerados por run |
| QG4 | FAIL (inalterado) | FAIL — aguardando FASES 6–8 (arquitetura) |
