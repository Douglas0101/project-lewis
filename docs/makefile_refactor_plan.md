# Refactor UX do Makefile (FASE 8)

Data: 2026-07-29 | Branch: `feat/pretrain-architecture-v2`

## Resultado

| Métrica | Antes | Depois |
|---|---|---|
| Alvos públicos (com `##`, visíveis no `make help`) | 66 | **35** |
| Alvos totais (funcionais) | ~136 | ~136 (nenhum removido) |
| Seções no help | 8 | 8 |
| Aliases DEPRECATED | mantidos | mantidos (testados) |

## Princípios aplicados

- **PRN-007**: Makefile é orquestração — nenhuma receita movida alterou lógica;
  as receitas complexas novas já vivem em scripts (`pretrain_wrapper.py`,
  `validate_pretrain_artifacts.py`, `evaluate_pretrain_run.py`, `export_tflite_smoke.py`).
- **Compatibilidade**: alvos internalizados apenas perderam o `##` (ficam ocultos
  do help) — continuam executáveis diretamente e como dependências.
- Aliases legados (`download-*`, `qg0`, `mlp-pipeline`, `knowledge-*` etc.) intactos.

## Alvos públicos finais (35)

- **Geral (9)**: help, setup, doctor, check, lint, test, test-e2e, clean, status
- **Dados (4)**: data-download-all, data-download-chapman, data-download-mitbih, data-verify-chapman
- **MLP v2.3 (2)**: mlp-run, mlp-qg5
- **E07R (4)**: e07r, e07r-e065, e07r-e07, e07r-watch
- **Firmware & Gates (4)**: fw-build, fw-test, gates-firmware, gates-ci
- **Knowledge & RAG (3)**: kb-index, kb-query, kb-status
- **Observabilidade & Memória (3)**: obs-up, obs-down, memory-commit
- **Pré-treino (6)**: pretrain, pretrain-qg, pretrain-smoke, pretrain-check, pretrain-validate, pretrain-export-smoke

## Internalizados (funcionais, ocultos)

- Dados: data-catalog, data-process, data-features, data-audit-train, data-qg0, data-provenance, data-mirror-create, data-mirror-restore, data-dlq-replay
- Geral: watch (atalho duplicado de e07r-watch)
- MLP: mlp-train, mlp-train-stage1/2, mlp-select-best, mlp-quantize, mlp-validate-quant, mlp-clean
- E07R: e07r-check, e07r-status, e07r-freeze, e07r-report
- Firmware: fw-native, fw-run, fw-verify-renode, fw-check-markers, fw-check-no-stub
- KB/RAG: kb-test, kb-validate, kb-clean, kb-reindex-changed, rag-eval-hybrid, rag-eval-ragas

## Flags padronizadas (inalteradas)

`DRY_RUN=1`, `FORCE=1`, `RUN_ID=...`, `STAGE=...`, `JSON=1` (+ `FEATURES=1`, `TFLM=1`, `STUB=1`).

## Exit codes nos targets de pré-treino

| Target | Execução OK + QG4 PASS | Execução OK + QG4 FAIL | Falha real |
|---|---|---|---|
| `make pretrain` | 0 | 0 | ≠0 |
| `make pretrain-qg` | 0 | **10** | ≠0 |
| `make pretrain-smoke` | 0 | 0 | ≠0 |
