from __future__ import annotations

import pytest


@pytest.mark.parametrize("sql", [
    "SELECT COUNT(*) FROM dim_hcp",
    "SELECT region_name, COUNT(*) FROM dim_territory JOIN dim_region USING (region_id) GROUP BY 1",
    "WITH q AS (SELECT month, SUM(trx) AS trx FROM fact_rx GROUP BY month) "
    "SELECT cur.month, cur.trx - prev.trx AS delta "
    "FROM q cur JOIN q prev ON prev.month < cur.month",
    "SELECT month, SUM(trx) OVER (ORDER BY month) AS running FROM fact_rx",
    "SELECT t.n FROM (SELECT COUNT(*) AS n FROM dim_rep) t",
    "SELECT brand_name AS b FROM dim_product ORDER BY b",
])
def test_accepts_valid_queries(validator, sql):
    result = validator.validate(sql)
    assert result.ok, result.errors


@pytest.mark.parametrize("sql, fragment", [
    ("SELECT SUM(s.revenue) FROM fact_sales s", "does not exist in fact_sales"),
    ("SELECT * FROM sales", "Did you mean fact_sales"),
    ("SELECT SUM(revenue) FROM fact_sales", "does not exist in the referenced tables"),
])
def test_flags_schema_hallucinations(validator, sql, fragment):
    result = validator.validate(sql)
    assert not result.ok and result.schema_hallucination
    assert fragment in " ".join(result.errors)


@pytest.mark.parametrize("sql", [
    "DELETE FROM fact_sales",
    "DROP TABLE dim_hcp",
    "INSERT INTO dim_region VALUES (9, 'Mars')",
    "PRAGMA table_info(dim_hcp)",
    "ATTACH DATABASE 'x.db' AS x",
    "SELECT 1; SELECT 2",
])
def test_blocks_non_read_only_sql(validator, sql):
    result = validator.validate(sql)
    assert not result.ok and not result.schema_hallucination


def test_reports_syntax_errors(validator):
    assert "syntax error" in validator.validate("SELEC 1 FROM").errors[0]


@pytest.mark.parametrize("sql", [
    "SELECT SUM(trx) / SUM(trx) FROM fact_rx",
    "SELECT ROUND(SUM(CASE WHEN product_id = 1 THEN trx ELSE 0 END) / SUM(trx) * 100, 1) "
    "FROM fact_rx",
    "SELECT COUNT(*) / COUNT(DISTINCT hcp_id) FROM fact_call",
    "SELECT SUM(r.nrx) * 100 / SUM(r.trx) FROM fact_rx r",
])
def test_flags_integer_division(validator, sql):
    result = validator.validate(sql)
    assert not result.ok and "Integer division" in result.errors[0]


@pytest.mark.parametrize("sql", [
    "SELECT 100.0 * SUM(nrx) / SUM(trx) FROM fact_rx",
    "SELECT CAST(SUM(nrx) AS REAL) / SUM(trx) FROM fact_rx",
    "SELECT SUM(net_sales) / SUM(gross_sales) FROM fact_sales",
    "SELECT 1.0 * COUNT(*) / COUNT(DISTINCT hcp_id) FROM fact_call",
    "SELECT SUM(s.net_sales) / q.target_net_sales FROM fact_sales s, fact_quota q",
])
def test_allows_float_division(validator, sql):
    assert validator.validate(sql).ok


def test_missing_column_hint_gives_the_join_path(validator):
    result = validator.validate("SELECT SUM(r.trx) FROM fact_rx r WHERE r.territory_id = 3")
    message = result.errors[0]
    assert "lives in dim_hcp" in message
    assert "dim_hcp.hcp_id = fact_rx.hcp_id" in message


def test_flags_joins_on_unrelated_keys(validator):
    result = validator.validate(
        "SELECT t.territory_name, SUM(r.trx) FROM fact_rx r "
        "JOIN dim_territory t ON t.territory_id = r.hcp_id GROUP BY 1")
    assert not result.ok
    assert "unrelated keys" in result.errors[0] and "dim_hcp" in result.errors[0]


def test_accepts_fk_joins_in_either_direction(validator):
    assert validator.validate(
        "SELECT COUNT(*) FROM dim_hcp h JOIN fact_rx r ON h.hcp_id = r.hcp_id "
        "JOIN dim_territory t ON h.territory_id = t.territory_id").ok
