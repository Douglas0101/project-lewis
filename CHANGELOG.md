# Changelog

Todas as mudanças notáveis do Project-Lewis serão documentadas neste arquivo.

O formato é baseado em [Keep a Changelog](https://keepachangelog.com/pt-BR/1.0.0/),
e este projeto adere ao [Semantic Versioning](https://semver.org/lang/pt-BR/spec/v2.0.0.html).

> **Nota sobre versionamento:** este changelog utiliza SemVer para marcos de
desenvolvimento. O repositório também contém tags não-SemVer para marcos de
simulação: `v1.2-sim-locked` e `v1.3-sim-deep`. A versão do pipeline de ML
reportada no README é `v2.2+c11+fase2` e evolui independentemente dos marcos
deste changelog.

## [Unreleased]

### Adicionado

- **Vistoria SLHA e caminho CPU/GPU**
  - `docs/vistoria_slha_gpu_adaptacao_v2.4.md` documentando a vistoria avançada do Sistema de Leitura de Hardware Automático (SLHA) e a adaptação de treinamento sem GPU.
  - Verificação de 15 testes SLHA e discovery funcional em CPU-only.
- **Experimentos Stage 2 v11–v16**
  - Variações de MLP com `hidden=256` e `dropout=0.5` (v11, v12), focal loss agressiva (v13, v14), arquitetura com duas camadas ocultas (v15) e `class_weight` (v16).
  - Suporte a `--optimize-thresholds`, `--class-weight`, `--hidden-units-2` e `--dropout-2` em `scripts/train_stage2_mlp.py`.
- **Quality Gate QG5' atualizado**
  - `tests/test_two_stage_mlp_qg5.py`: threshold `F1(F) >= 0.50` (anterior 0.15).
  - `scripts/select_best_mlp_fold.py`: critério de publicação com `F1(F) >= 0.50`.

### Modificado

- `scripts/train_stage2_mlp.py`: adicionadas flags `--optimize-thresholds`, `--threshold-metric`, `--class-weight`, `--hidden-units-2` e `--hidden-units-2`.
- `scripts/select_best_mlp_fold.py`: tratamento de exceções, tipagem `dict` nativa e threshold F1(F) elevado para 0.50.
- `tests/test_two_stage_mlp_qg5.py`: `STAGE2_MIN_F1_F` atualizado para 0.50 e ajustes de `zero_division` para compatibilidade de tipos.

### Publicado

- Artefatos v2.3 em `models/`:
  - `stage1_float32_v2.3.keras` (origem: `experiments/stage1_mlp_features_v2.3_retrain`).
  - `stage2_float32_v2.3.keras` (origem: `experiments/stage2_mlp_features_v2.3_focal_smote_v14`, fold 4).
  - Thresholds Stage 1 e Stage 2 otimizados por Youden/F1.

### Resultados

- Stage 1: Recall(Anormal)=0.8352, Precision(Anormal)=0.8286, F1-macro=0.9021.
- Stage 2 (subset balanceado QG5): F1-macro=0.8654, F1(S)=0.8544, F1(V)=0.8038, F1(F)=0.9379.
- 18 testes relevantes passaram (`test_two_stage_mlp_qg5` + `test_slha_*`).

### Research Branch v2.4 — Classe F em Stage 2

- **E00–E09 concluídas** com checkpoints `PASS` ou `PASS_HYPOTHESIS_REJECTED`.
- **E00**: snapshot forense do baseline v14 em `experiments/stage2_v2.4_research/E00_baseline_snapshot/`.
- **E01**: auditoria de distribuição de F por registro; 70% concentrado em 208/213.
- **E02**: manifestos imutáveis de dataset/features e validador em `src/inference/manifest_validator.py`.
- **E03**: protocolo de split `StratifiedGroupKFold` selecionado; F presente em todos os folds.
- **E04**: QG5 redesenhado com gate `QG5_PATIENTWISE`; status `RESEARCH_CANDIDATE_NOT_PUBLICATION_READY`.
- **E05**: auditoria de separabilidade das 16 features; top features RR-dominadas.
- **E06**: engenharia de features enhanced (33 dimensões); hipótese rejeitada.
- **E07**: reescrita de rótulos não justificada; reamostragem por paciente melhorou F1(F) baseline para 0.465.
- **E08**: MLP 256 + focal loss + class-weight sobre F reamostrado; F1-macro=0.607, F1(F)=0.453 (ainda < 0.50).
- **E09**: relatório final em `docs/stage2_v2.4_research_report.md`; decisão **NÃO PUBLICAR v2.4**; artefatos v2.3 preservados.
- **Investigação PTB-XL**: 21.799 registros locais, 48 com `AFIB=100`, 961 com texto validado, potencial de ~1.000–21.000 batimentos F. Documentado em `docs/ptbxl_afib_investigation_report.md` e proposta em `docs/dataset_update_proposal_v2.5.md`.
- **Testes**: 79 testes passaram (`tests/test_v2_4_*`, `tests/test_two_stage_mlp_qg5.py`, `tests/test_slha_*.py`).

## [3.1.0] — 2026-07-06

### Adicionado

- **Observabilidade**
  - `src/observability/metrics.py` com coleta padronizada de métricas de ML.
  - `LatencyTracker` para instrumentação de latência por fase do pipeline.
  - Endpoints FastAPI `/metrics` e `/health` para exposição de health check e métricas Prometheus.
  - Stack Docker Compose com Prometheus + Grafana e dashboard JSON em `observability/grafana/dashboards/lewis.json`.
- **Busca Híbrida**
  - `src/knowledge/hybrid_search.py` combinando BM25 (FTS5), busca vetorial e RRF.
  - Migration `src/knowledge/migrations/001_fts5_hybrid.sql` criando índice FTS5 para BM25.
  - Integração do hybrid search no retriever RAG, CLI e MCP server.
  - Conjunto de dados de referência para avaliação e `scripts/eval_hybrid.py`.
  - Novo target `make hybrid-eval`.
- **Timeline materializada**
  - Migration `002_materialized_timeline.sql` criando tabela materializada.
  - `TimelineRefresher` async para refresh completo/periódico pós-run.
  - Refresh hook disponível para ser invocado após cada run via `refresh_timeline_after_run()`.
  - Teste de performance com 1M runs em `tests/stress/test_ponte3_timeline.py`.
- **Quality Gate de Stress em CI**
  - Workflow `.github/workflows/stress-gate.yml` para testes de carga e de longa duração.
  - Target `make stress-test` para execução local do stress gate.
- **Avaliação RAGAS**
  - `src/observability/ragas_eval.py` para avaliação contínua do RAG.
  - `src/observability/ragas_eval_cli.py` como CLI de execução.
  - Conjunto de dados de referência para avaliação.
  - Target `make ragas-eval` (requer grupo opcional de dependências `eval`).
- **Segurança em Nível de Linha e Auditoria**
  - Migration `003_owner_id.sql` adicionando coluna `owner_id` nas tabelas `experiment` e `run`.
  - `src/security/rls.py` com helper de RLS baseado em `owner_id`.
  - Integração de RLS em `src/knowledge/structured_query.py`.
  - Migration `004_owner_id_timeline.sql` propagando `owner_id` para a timeline.
  - Audit log completo com `user_id`, SQL final e parâmetros executados.
- **Testes Adversariais**
  - `tests/stress/adversarial_corpus.jsonl` com prompts maliciosos conhecidos.
  - `tests/stress/corpus_loader.py` para carregamento e execução do corpus.
  - `tests/stress/test_adversarial_fuzz.py` para fuzz testing da camada NL→SQL.

### Segurança

- RLS por `owner_id` nas consultas NL→SQL, garantindo isolamento de dados entre usuários.
- Audit trail completo de consultas estruturadas, incluindo SQL final e parâmetros.
- Corpus adversarial e fuzz testing para rejeição de SQL injection e prompt injection.

### Modificado

- README atualizado com as funcionalidades da Fase 2.

### Corrigido

- Validação de `max_rows` em `execute_custom_sql` e `answer_question`.
- Correção de aliases de tabela em `validate_sql`.
- Ajustes no README para comandos e contagem de testes.

## 1.1.0 — 2026-06-30

### Adicionado

- **Ponte RAG ↔ Banco Estruturado de Métricas**
  - `src/knowledge/experiment_summarizer.py`: gera resumos markdown de experimentos a partir de `lewis_metrics.db` e os persiste em `data/lineage/experiments/` para indexação no RAG.
  - `src/knowledge/structured_query.py`: catálogo determinístico NL→SQL + validação de SQL personalizada, execução read-only e audit trail.
  - `src/tracking/timeline.py` + migration `001_experiment_timeline.sql`: view consolidada `experiment_timeline` com health status e filtros Pydantic.
  - Novas tools MCP: `query_training_logs` e `execute_validated_sql`.
  - Testes de robustez para as 3 pontes em `tests/stress/`.

### Segurança

- 100% das consultas SQL parametrizadas; nenhuma concatenação de strings em SQL.
- Validação Pydantic v2 em todas as interfaces de timeline e query.
- Sanitização de markdown via `bleach` antes da indexação no RAG.

## 1.0.0 — 2026-06-27

### Adicionado

- Camada C11 — Knowledge Layer com RAG local (`sqlite-vec`), indexador semântico, retriever com filtros 3D e MCP server.

[Unreleased]: https://github.com/Douglas0101/project-lewis/compare/v3.1.0...HEAD
[3.1.0]: https://github.com/Douglas0101/project-lewis/compare/v3.0.0...v3.1.0
