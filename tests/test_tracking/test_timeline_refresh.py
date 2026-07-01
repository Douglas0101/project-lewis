from sqlalchemy import text

from src.tracking.db import get_engine, init_schema


def test_materialized_table_created(tmp_path):
    db_path = tmp_path / "mat.db"
    engine = get_engine(db_path)
    init_schema(engine)
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name='experiment_timeline_materialized'"
            )
        ).fetchone()
        assert row is not None
