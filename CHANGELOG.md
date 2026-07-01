# Changelog

Todas as mudanças notáveis do Project-Lewis serão documentadas neste arquivo.

O formato é baseado em [Keep a Changelog](https://keepachangelog.com/pt-BR/1.0.0/),
e este projeto adere ao [Semantic Versioning](https://semver.org/lang/pt-BR/spec/v2.0.0.html).

> **Nota sobre versionamento:** este changelog utiliza SemVer para marcos de
desenvolvimento. O repositório também contém tags não-SemVer para marcos de
simulação: `v1.2-sim-locked` e `v1.3-sim-deep`. A versão do pipeline de ML
reportada no README é `v2.2+c11+fase2` e evolui independentemente dos marcos
deste changelog.

## [Unreleased] — Fase 2

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

[Unreleased]: https://github.com/Douglas0101/project-lewis/compare/v1.3-sim-deep...HEAD
