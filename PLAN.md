# PLAN.md — Project-Lewis

Plano de execução de tasks decompostas. Atualizar a cada sessão.

## Ativas

- [x] **Finalização v3.0** — Resolver ambiente TFLM, validar e integrar branch `feature/plano-v3-rag-memory`
  - [x] Instalar toolchain ARM e Renode na worktree (`scripts/install_firmware_tools.sh`).
  - [x] Clonar/buildar `firmware/third_party/tflite-micro` no commit pinado (via cache do repo principal).
  - [x] Reexecutar `pytest tests/` e confirmar que erros de ambiente foram resolvidos.
  - [x] Criar PR para `develop` com revisão humana: https://github.com/Douglas0101/project-lewis/pull/4
  - [ ] Limpar worktree e branch após merge.

## Backlog

- [ ] C06 — Adicionar schema tests estruturais (pydantic) nos pipelines de dados
- [ ] C08 — Migrar firmware de modelo monolítico v1.1 para dois modelos TFLM v2.0
- [ ] C09 — Expandir relatório de energia com cenários de sleep/stop/standby
- [ ] C10 — Gerar fixtures Python para testes de harness DSP/inference/R-peak

## Concluídas

### v3.0 — Plano de Engenharia ML + RAG + Memória (2026-06-28)

Implementação concluída na branch `feature/plano-v3-rag-memory`:

- [x] **Fase 0 — Memória do Sistema**
  - RAG reindexado (`data/knowledge.db`, 2490 chunks).
  - Camada `PLANO` adicionada ao `LAYER_MAP`.
  - Tabela `Artifact` e repositórios no tracking.
- [x] **Fase 1 — Correção Estatística**
  - `WeightedFocalLoss`, `StratifiedGroupKFold`, `threshold_optimizer`, `SmoteSequence`.
- [x] **Fase 2 — Reengenharia Arquitetural**
  - `ECGClassifierV3`, notebook de ablação, ensemble top-2 weighted.
- [x] **Fase 3 — Quantização Robusta**
  - Pipeline QAT INT8 e `StrifiedRepresentativeDataset`.
- [x] **Fase 4 — Métricas e Quality Gates**
  - `bootstrap_ci`, matriz de confusão normalizada, drift de scaler.
- [x] **Fase 5 — Manutenção da Memória**
  - Restaurado `src/memory/checksums.py` e integrado `Artifact` ao tracking.
  - Adicionado target `make knowledge-reindex-if-docs-changed`.
  - Corrigido `session_scope` para usar `@contextmanager`.

Validação:
- `make lint` PASS (flake8, mypy, bandit).
- `pytest tests/test_memory/ tests/test_knowledge/` PASS (42/42).
- `pytest tests/` parcial: 299 passed, 48 skipped, 8 errors (ambiente TFLM ausente na worktree).

### Histórico

- [x] 2026-06-20 — Auditoria de qualidade dos dados pré-treino (QG0–QG3 + dataset de features)
- [x] 2026-06-20 — Correções finais no `docs/SDD_Project-Lewis_v3.md`
- [x] 2026-06-20 — Criar `mcp.json`, `.kimi/sdd-context.md`, `.opencode/sdd-context.md`, `AGENTS.md`
- [x] 2026-06-21 — C07: Sincronizar `pyproject.toml`
- [x] 2026-06-21 — C10: Implementar Test Harness de Firmware (native + Renode)
- [x] 2026-06-21/22 — **C04 v2.0 correção UNIFIED_DOCUMENT_v2.0**
- [x] 2026-06-26 — **C01 restabelecido**: Chapman 45.152, MIT-BIH+, PTB-XL 43.598
- [x] 2026-06-26 — **C02 reprocessado**: Chapman e PTB-XL com sucesso
- [x] 2026-06-26 — **Correção de transfer learning**: `freeze_backbone` propagado nos scripts stage1/stage2; testes de regressão adicionados
