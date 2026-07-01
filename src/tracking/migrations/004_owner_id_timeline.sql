-- Migration 004: expose owner_id in experiment_timeline view for RLS
-- A tabela materializada é reconstruída a partir da view; isso é aceitável
-- porque o refresh é periódico e os dados de tracking podem ser regenerados.

BEGIN;

DROP VIEW IF EXISTS experiment_timeline;

CREATE VIEW experiment_timeline AS
SELECT
    r.id AS run_id,
    e.id AS experiment_id,
    e.name AS experiment_name,
    e.stage AS stage,
    e.owner_id AS owner_id,
    COALESCE(
        json_extract(r.extra, '$.model_name'),
        json_extract(e.extra, '$.model_name'),
        e.name
    ) AS model_name,
    r.run_type AS run_type,
    r.status AS status,
    r.start_time AS run_start,
    r.end_time AS finished_at,
    CASE
        WHEN r.end_time IS NOT NULL THEN
            ROUND((julianday(r.end_time) - julianday(r.start_time)) * 86400, 2)
        ELSE NULL
    END AS duration_seconds,

    -- Métricas finais (namespace global), com fallback case-insensitive para nomes comuns
    MAX(CASE WHEN m.namespace = 'global' AND m.name = 'accuracy' THEN m.value END) AS final_accuracy,
    MAX(CASE WHEN m.namespace = 'global' AND m.name = 'loss' THEN m.value END) AS final_loss,
    COALESCE(
        MAX(CASE WHEN m.namespace = 'global' AND LOWER(m.name) = 'f1_macro' THEN m.value END),
        MAX(CASE WHEN m.namespace = 'global' AND LOWER(m.name) = 'f1-macro' THEN m.value END)
    ) AS final_f1_macro,
    MAX(CASE WHEN m.namespace = 'global' AND LOWER(m.name) = 'auc_roc' THEN m.value END) AS final_auc_roc,

    -- Baselines (podem vir do extra do experimento)
    CAST(json_extract(e.extra, '$.baseline_accuracy') AS REAL) AS baseline_accuracy,
    CAST(json_extract(e.extra, '$.baseline_loss') AS REAL) AS baseline_loss,
    CAST(json_extract(e.extra, '$.baseline_f1_macro') AS REAL) AS baseline_f1_macro,
    CAST(json_extract(e.extra, '$.baseline_auc_roc') AS REAL) AS baseline_auc_roc,

    -- Deltas calculados
    (
        COALESCE(
            MAX(CASE WHEN m.namespace = 'global' AND LOWER(m.name) = 'f1_macro' THEN m.value END),
            MAX(CASE WHEN m.namespace = 'global' AND LOWER(m.name) = 'f1-macro' THEN m.value END)
        )
        - CAST(json_extract(e.extra, '$.baseline_f1_macro') AS REAL)
    ) AS f1_macro_delta,

    -- Alertas
    COUNT(DISTINCT CASE WHEN a.severity = 'critical' THEN a.id END) AS critical_alerts,
    COUNT(DISTINCT CASE WHEN a.severity = 'warning' THEN a.id END) AS warning_alerts,
    COUNT(DISTINCT CASE WHEN a.severity = 'info' THEN a.id END) AS info_alerts,

    -- Flag de qualidade
    CASE
        WHEN r.status = 'failed' THEN 'FAILED'
        WHEN COUNT(DISTINCT CASE WHEN a.severity = 'critical' THEN a.id END) > 0 THEN 'UNSTABLE'
        WHEN (
            COALESCE(
                MAX(CASE WHEN m.namespace = 'global' AND LOWER(m.name) = 'f1_macro' THEN m.value END),
                MAX(CASE WHEN m.namespace = 'global' AND LOWER(m.name) = 'f1-macro' THEN m.value END)
            )
            < CAST(json_extract(e.extra, '$.baseline_f1_macro') AS REAL)
        ) THEN 'REGRESSION'
        ELSE 'HEALTHY'
    END AS health_status
FROM experiment e
JOIN run r ON r.experiment_id = e.id
LEFT JOIN metric m ON m.run_id = r.id
LEFT JOIN alert a ON a.run_id = r.id
GROUP BY
    r.id,
    e.id,
    e.name,
    e.stage,
    e.owner_id,
    e.extra,
    r.run_type,
    r.status,
    r.start_time,
    r.end_time;

-- Recria tabela materializada a partir da view atualizada
DROP TABLE IF EXISTS experiment_timeline_materialized;

CREATE TABLE experiment_timeline_materialized AS
SELECT * FROM experiment_timeline;

CREATE UNIQUE INDEX idx_timeline_run_id ON experiment_timeline_materialized(run_id);
CREATE INDEX idx_timeline_stage_status ON experiment_timeline_materialized(stage, status);
CREATE INDEX idx_timeline_health ON experiment_timeline_materialized(health_status);
CREATE INDEX idx_timeline_created ON experiment_timeline_materialized(run_start);
CREATE INDEX idx_timeline_owner_id ON experiment_timeline_materialized(owner_id);

COMMIT;
