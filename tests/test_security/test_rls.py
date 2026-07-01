from sqlalchemy import text

from src.tracking.db import get_engine, init_schema


def test_owner_id_column_exists(tmp_path):
    db_path = tmp_path / "rls.db"
    engine = get_engine(db_path)
    init_schema(engine)
    with engine.connect() as conn:
        cols_exp = {
            r["name"] for r in conn.execute(text("PRAGMA table_info(experiment)")).mappings()
        }
        cols_run = {
            r["name"] for r in conn.execute(text("PRAGMA table_info(run)")).mappings()
        }
        assert "owner_id" in cols_exp
        assert "owner_id" in cols_run
