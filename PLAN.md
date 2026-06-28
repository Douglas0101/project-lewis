# PLAN.md — Project-Lewis

Plano de execução de tasks decompostas. Atualizar a cada sessão.

## Ativas

- [ ] **C04 v2.0 — Task 4.3** — Retreinar Estágio 1 (N vs Anormal) com transfer learning efetivo
  - Script: `scripts/run_stage1_training.py --pretrained ... --freeze-backbone --n-splits 5`
  - Em andamento (`task bash-f8lvhq0d`)
  - Correção: `freeze_backbone` agora é propagado para `train_group_kfold`
  - Testes de regressão: `tests/test_run_stage_training.py`
- [ ] **C04 v2.0 — Task 4.4** — Retreinar Estágio 2 (S vs V vs F) com transfer learning efetivo
  - Script: `scripts/run_stage2_training.py --pretrained ... --freeze-backbone --n-splits 5`
  - Em andamento (`task bash-67fplm0b`)
- [ ] **C04 v2.0 — Task 4.5** — Avaliar pipeline integrado QG5' (4 classes)
  - Script: `scripts/run_two_stage_pipeline.py`
  - Implementado; precisa dos modelos treinados para executar
- [ ] **C04 v2.0 — Task 4.6** — Quantizar ambos os modelos e validar QG6
  - Script: `scripts/quantize_two_stage_v2.0.py`
  - Quantização do Stage2 validada (14.69 KB); Stage1 pendente de modelo
- [ ] **C04 v2.0 — Task 4.7** — Testes, lineage, relatório e DLQ vazia
  - Testes unitários para threshold/pipeline criados: `tests/test_two_stage_pipeline.py` (4 PASS)
  - Novos testes de regressão: `tests/test_run_stage_training.py` (4 PASS)
  - Faltam: testes de integração dos scripts de treinamento e relatório final

## Backlog

- [ ] C06 — Adicionar schema tests estruturais (pydantic) nos pipelines de dados
- [ ] C08 — Migrar firmware de modelo monolítico v1.1 para dois modelos TFLM v2.0
- [ ] C09 — Expandir relatório de energia com cenários de sleep/stop/standby
- [ ] C10 — Gerar fixtures Python para testes de harness DSP/inference/R-peak

## Concluídas

- [x] 2026-06-20 — Auditoria de qualidade dos dados pré-treino (QG0–QG3 + dataset de features)
- [x] 2026-06-20 — Correções finais no `docs/SDD_Project-Lewis_v3.md`
- [x] 2026-06-20 — Criar `mcp.json`, `.kimi/sdd-context.md`, `.opencode/sdd-context.md`, `AGENTS.md`
- [x] 2026-06-21 — C07: Sincronizar `pyproject.toml`
- [x] 2026-06-21 — C10: Implementar Test Harness de Firmware (native + Renode)
- [x] 2026-06-21/22 — **C04 v2.0 correção UNIFIED_DOCUMENT_v2.0**
  - Datasets stage1/stage2: `data/features/stage1_binary.*`, `data/features/stage2_multiclass.*`
  - Configs: `config/stage1_binary.yaml`, `config/stage2_multiclass.yaml`
  - `AGENTS.md` atualizado com thresholds QG5' v2.0
  - Threshold tuning binário em `src/models/evaluate.py` e callback `F1MacroCheckpoint`
  - Pipeline integrado `src/models/two_stage_pipeline.py` + `scripts/run_two_stage_pipeline.py`
  - Quantização `scripts/quantize_two_stage_v2.0.py`
  - `make lint` PASS; novos testes `tests/test_two_stage_pipeline.py` PASS
- [x] 2026-06-26 — **C01 restabelecido**: Chapman 45.152, MIT-BIH+, PTB-XL 43.598
- [x] 2026-06-26 — **C02 reprocessado**: Chapman e PTB-XL com sucesso
- [x] 2026-06-26 — **Correção de transfer learning**: `freeze_backbone` propagado nos scripts stage1/stage2; testes de regressão adicionados
