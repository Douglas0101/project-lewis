-- Migration 003: add owner_id for row-level security filtering
ALTER TABLE experiment ADD COLUMN owner_id TEXT;
ALTER TABLE run ADD COLUMN owner_id TEXT;

CREATE INDEX IF NOT EXISTS idx_experiment_owner_id ON experiment(owner_id);
CREATE INDEX IF NOT EXISTS idx_run_owner_id ON run(owner_id);
