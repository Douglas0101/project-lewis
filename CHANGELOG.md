# Changelog

Todas as mudanças notáveis do Project-Lewis serão documentadas neste arquivo.

O formato é baseado em [Keep a Changelog](https://keepachangelog.com/pt-BR/1.0.0/),
e este projeto adere ao [Semantic Versioning](https://semver.org/lang/pt-BR/spec/v2.0.0.html).

## [1.1.0] — 2026-06-30

### Adicionado

- **Ponte RAG ↔ Banco Estruturado de Métricas**
  - `src/knowledge/experiment_summarizer.py`: gera resumos markdown de experimentos a partir de `lewis_metrics.db` e os persiste em `data/lineage/experiments/` para indexação no RAG.
  - `src/knowledge/structured_query.py`: catálogo determinístico NL→SQL + validação de SQL customizado, execução read-only e audit trail.
  - `src/tracking/timeline.py` + migration `001_experiment_timeline.sql`: view consolidada `experiment_timeline` com health status e filtros Pydantic.
  - Novas tools MCP: `query_training_logs` e `execute_validated_sql`.
  - Testes de robustez para as 3 pontes em `tests/test_knowledge/` e `tests/test_tracking/`.

### Segurança

- 100% das queries SQL parametrizadas; nenhuma concatenação de string em SQL.
- Validação Pydantic v2 em todas as interfaces de timeline e query.
- Sanitização de markdown via `bleach` antes da indexação no RAG.

## [1.0.0] — 2026-06-27

### Adicionado

- Camada C11 — Knowledge Layer com RAG local (`sqlite-vec`), indexador semântico, retriever com filtros 3D e MCP server.
- Tracking estruturado de experimentos (`src/tracking/`) com SQLite/SQLAlchemy 2.0.
