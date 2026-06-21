# PLAN.md — Project-Lewis

Plano de execução de tasks decompostas. Atualizar a cada sessão.

## Ativas

- [ ] C06 — Adicionar schema tests estruturais (pydantic) nos pipelines de dados
- [ ] C09 — Expandir relatório de energia com cenários de sleep/stop/standby

## Backlog

- [ ] C10 — Gerar fixtures Python para testes de harness DSP/inference/R-peak

## Concluídas

- [x] 2026-06-21 — Onboarding e execução intuitiva em equipe
  - Targets auto-documentados no Makefile (`make help`, `make setup`, `make doctor`, `make dev`)
  - Script `scripts/check_environment.py` para checagem de ambiente
  - `CONTRIBUTING.md` com workflow de equipe
  - Dockerfile corrigido para Python 3.12 + dev dependencies
  - README atualizado com fluxo de primeiro acesso
- [x] 2026-06-20 — Auditoria de qualidade dos dados pré-treino (QG0–QG3 + dataset de features)
  - Schemas pydantic: `src/data/training_schemas.py`
  - Auditoria: `scripts/audit_training_data.py` + relatórios JSON/Markdown + DLQ
  - Pipeline de features: `src/features/pipeline.py` (MIT-BIH family → parquet + npz + manifest)
  - Integração: targets `features` / `audit-training-data` no `Makefile` + workflow `.github/workflows/data-quality-audit.yml`
  - Testes: `tests/test_training_data_audit.py`
  - Validação: `make lint` PASS; novos testes PASS; dataset em geração
- [x] 2026-06-20 — Correções finais no `docs/SDD_Project-Lewis_v3.md`
- [x] 2026-06-20 — Criar `mcp.json`, `.kimi/sdd-context.md`, `.opencode/sdd-context.md`, `AGENTS.md`
- [x] 2026-06-20 — C07: Sincronizar `pyproject.toml` (pytest-cov, pytest-xdist; remover classificador Python 3.13)
- [x] 2026-06-20 — C10: Implementar Test Harness de Firmware (native + Renode)
  - Artefatos: `firmware/tests/harness.{h,c}`, `firmware/tests/harness_main.c`, `firmware/tests/test_dsp.c`, `firmware/tests/test_r_peak.c`, `firmware/tests/test_inference.cpp`, `firmware/scripts/run_harness.py`, `firmware/renode/harness.resc`
  - Targets: `harness-native`, `harness-renode`, `harness`
  - Resultado: 7/7 PASS em native e Renode; relatório em `firmware/test_harness_report.json`
