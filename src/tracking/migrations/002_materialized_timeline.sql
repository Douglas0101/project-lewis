-- Migration 002: materialized copy of experiment_timeline
-- Cria tabela materializada com índices para consultas rápidas de linha do tempo.

DROP TABLE IF EXISTS experiment_timeline_materialized;

CREATE TABLE experiment_timeline_materialized AS
SELECT * FROM experiment_timeline;

CREATE UNIQUE INDEX idx_timeline_run_id ON experiment_timeline_materialized(run_id);
CREATE INDEX idx_timeline_stage_status ON experiment_timeline_materialized(stage, status);
CREATE INDEX idx_timeline_health ON experiment_timeline_materialized(health_status);
CREATE INDEX idx_timeline_created ON experiment_timeline_materialized(run_start);
