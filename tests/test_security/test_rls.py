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


def test_admin_bypass_case_insensitive():
    rls = RowLevelSecurity()
    sql, params = rls.apply_filter("SELECT * FROM run", "admin", roles=["Admin"])
    assert "owner_id" not in sql
    assert params == ()


def test_admin_bypass_exact_role_not_substring():
    rls = RowLevelSecurity()
    sql, params = rls.apply_filter("SELECT * FROM run", "user", roles=["superadmin"])
    assert "owner_id = ?" in sql
    assert params == ("user",)


def test_rls_named_params():
    rls = RowLevelSecurity()
    sql, params = rls.apply_filter("SELECT * FROM run WHERE status = :status", "user_a")
    assert "owner_id = :owner_id" in sql
    assert params == {"owner_id": "user_a"}


def test_rls_ignores_named_bind_inside_string_literal():
    rls = RowLevelSecurity()
    sql, params = rls.apply_filter("SELECT * FROM run WHERE name = ':not_a_bind'", "user_a")
    assert "owner_id = ?" in sql
    assert params == ("user_a",)


def test_rls_empty_user_id_bypasses():
    rls = RowLevelSecurity()
    sql, params = rls.apply_filter("SELECT * FROM run", "")
    assert "owner_id" not in sql
    assert params == ()


def test_rls_whitespace_user_id_bypasses():
    rls = RowLevelSecurity()
    sql, params = rls.apply_filter("SELECT * FROM run", "   ")
    assert "owner_id" not in sql
    assert params == ()


def test_rls_group_by_prepends_where():
    rls = RowLevelSecurity()
    sql, params = rls.apply_filter(
        "SELECT stage, COUNT(*) FROM experiment GROUP BY stage", "user_a"
    )
    assert "WHERE owner_id = ? GROUP BY stage" in sql
    assert params == ("user_a",)


def test_rls_having_prepends_where():
    rls = RowLevelSecurity()
    sql, params = rls.apply_filter(
        "SELECT stage, COUNT(*) FROM experiment GROUP BY stage HAVING COUNT(*) > 1",
        "user_a",
    )
    assert "WHERE owner_id = ? GROUP BY stage HAVING" in sql
    assert params == ("user_a",)


def test_rls_limit_prepends_where():
    rls = RowLevelSecurity()
    sql, params = rls.apply_filter("SELECT * FROM experiment LIMIT 10", "user_a")
    assert "WHERE owner_id = ? LIMIT 10" in sql
    assert params == ("user_a",)


def test_rls_ignores_keywords_inside_parentheses():
    rls = RowLevelSecurity()
    sql, params = rls.apply_filter(
        "SELECT * FROM run WHERE status IN (SELECT status FROM other WHERE x = 1)", "user_a"
    )
    # Apenas um WHERE no nível top-level; o WHERE interno não deve gerar cláusula extra
    assert sql.count("WHERE") == 2
    assert "owner_id = ? AND" in sql
    assert params == ("user_a",)


def test_rls_table_alias_and_owner_column():
    rls = RowLevelSecurity()
    sql, params = rls.apply_filter(
        "SELECT * FROM experiment ORDER BY id", "user_a", table_alias="e", owner_column="owner_id"
    )
    assert "e.owner_id = ?" in sql
    assert params == ("user_a",)


def test_rls_lowercase_where():
    rls = RowLevelSecurity()
    sql, params = rls.apply_filter("select * from run where status = 'ok'", "user_a")
    assert "where owner_id = ? AND status = 'ok'" in sql
    assert params == ("user_a",)


def test_rls_mixed_case_where():
    rls = RowLevelSecurity()
    sql, params = rls.apply_filter("SELECT * FROM run Where status = 'ok'", "user_a")
    assert "Where owner_id = ? AND status = 'ok'" in sql
    assert params == ("user_a",)


def test_rls_ignores_keyword_inside_string_literal():
    rls = RowLevelSecurity()
    sql, params = rls.apply_filter("SELECT 'WHERE' AS x FROM run WHERE status = 'ok'", "user_a")
    assert sql.count("WHERE") == 2  # original WHERE + injected filter
    assert "WHERE owner_id = ? AND status = 'ok'" in sql
    assert params == ("user_a",)


def test_rls_lowercase_group_by():
    rls = RowLevelSecurity()
    sql, params = rls.apply_filter(
        "select stage, count(*) from experiment group by stage", "user_a"
    )
    assert "WHERE owner_id = ? group by stage" in sql
    assert params == ("user_a",)
