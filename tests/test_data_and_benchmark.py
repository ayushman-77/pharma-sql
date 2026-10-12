"""Data invariants and benchmark integrity: every gold query is valid and non-empty."""

from __future__ import annotations

import pytest
import yaml

from pharmasql.config import PROJECT_ROOT
from pharmasql.data.generate import npi_with_check_digit
from pharmasql.eval.runner import load_benchmark

BENCHMARK = load_benchmark(PROJECT_ROOT / "evals" / "benchmark.yaml")
BANK = yaml.safe_load((PROJECT_ROOT / "config" / "verified_queries.yaml").read_text())


def scalar(executor, sql):
    return executor.run(sql).rows[0][0]


def test_nrx_never_exceeds_trx(executor):
    assert scalar(executor, "SELECT COUNT(*) FROM fact_rx WHERE nrx > trx") == 0


def test_net_below_gross_and_sales_only_for_own_brands(executor):
    assert scalar(executor, "SELECT COUNT(*) FROM fact_sales WHERE net_sales > gross_sales") == 0
    assert scalar(executor, "SELECT COUNT(*) FROM fact_sales s JOIN dim_product p "
                            "USING (product_id) WHERE p.is_own_brand = 0") == 0


def test_no_prescriptions_before_launch(executor):
    assert scalar(executor, "SELECT COUNT(*) FROM fact_rx r JOIN dim_product p "
                            "USING (product_id) WHERE r.month < substr(p.launch_date, 1, 7) "
                            "|| '-01'") == 0


def test_sales_units_reconcile_with_prescriptions(executor):
    diff = scalar(executor, """
        SELECT ABS((SELECT SUM(units) FROM fact_sales)
                 - (SELECT SUM(r.trx) FROM fact_rx r JOIN dim_product p USING (product_id)
                    WHERE p.is_own_brand = 1))""")
    assert diff == 0


def test_npi_check_digit_is_luhn_valid():
    assert npi_with_check_digit("123456789") == "1234567893"  # CMS published example


def test_benchmark_shape():
    ids = [b.id for b in BENCHMARK]
    assert len(ids) == len(set(ids)) == 80
    assert sum(not b.answerable for b in BENCHMARK) == 15


def test_benchmark_and_fewshot_bank_are_disjoint():
    bank = {item["question"].strip().lower() for item in BANK}
    assert not bank & {b.question.strip().lower() for b in BENCHMARK}


@pytest.mark.parametrize("item", [b for b in BENCHMARK if b.answerable], ids=lambda b: b.id)
def test_gold_sql_is_valid_and_non_empty(item, validator, executor):
    for sql in [item.sql, *item.alt_sql]:
        assert validator.validate(sql).ok
        assert executor.run(sql).rows


@pytest.mark.parametrize("item", [b for b in BANK if b["sql"]], ids=lambda b: b["question"][:40])
def test_fewshot_bank_sql_is_valid_and_non_empty(item, validator, executor):
    assert validator.validate(item["sql"]).ok
    assert executor.run(item["sql"]).rows
