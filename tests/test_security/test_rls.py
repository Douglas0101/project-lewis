from src.security.rls import RowLevelSecurity


def test_rls_adds_where_clause():
    rls = RowLevelSecurity()
    sql, params = rls.apply_filter("SELECT * FROM run WHERE status = 'completed'", "user_a")
    assert "owner_id = ?" in sql
    assert params == ("user_a",)


def test_rls_adds_where_without_existing_where():
    rls = RowLevelSecurity()
    sql, params = rls.apply_filter("SELECT * FROM run ORDER BY id", "user_a")
    assert "WHERE owner_id = ?" in sql
    assert "ORDER BY id" in sql
    assert params == ("user_a",)


def test_admin_bypass():
    rls = RowLevelSecurity()
    sql, params = rls.apply_filter("SELECT * FROM run", "admin", roles=["admin"])
    assert "owner_id" not in sql
    assert params == ()


def test_rls_named_params():
    rls = RowLevelSecurity()
    sql, params = rls.apply_filter("SELECT * FROM run WHERE status = :status", "user_a")
    assert "owner_id = :owner_id" in sql
    assert params == {"owner_id": "user_a"}
