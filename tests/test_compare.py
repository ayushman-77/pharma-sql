from __future__ import annotations

from pharmasql.eval.compare import results_match


def test_row_order_is_ignored():
    assert results_match([("a", 1), ("b", 2)], [("b", 2), ("a", 1)])


def test_extra_predicted_columns_are_allowed():
    assert results_match([(51.3,)], [("Glucovia", 51.3)])


def test_column_order_is_ignored():
    assert results_match([("West", 10)], [(10, "West")])


def test_rounding_is_not_an_error():
    assert results_match([(51.2617,)], [(51.3,)])
    assert results_match([(5.455077,)], [(5.5,)])
    assert results_match([(15020621.4,)], [(15020621.0,)])
    assert results_match([(0.1234,)], [(0.12,)])


def test_rounding_awareness_does_not_hide_real_differences():
    assert not results_match([(51.2617,)], [(51.4,)])
    assert not results_match([(5.455,)], [(5.6,)])
    assert not results_match([(0.4,)], [(0,)])  # truncated integer division


def test_wrong_values_fail():
    assert not results_match([(51.3,)], [(0.513,)])  # fraction vs percentage
    assert not results_match([("a", 1)], [("a", 2)])


def test_row_count_mismatch_fails():
    assert not results_match([("a",)], [("a",), ("b",)])


def test_missing_gold_column_fails():
    assert not results_match([("West", 10)], [("West",)])


def test_numeric_strings_match_numbers():
    assert results_match([("2024", 61.8)], [(2024, 61.77)])


def test_rows_must_align_across_columns():
    # Same column multisets, different pairing: must not match.
    assert not results_match([("a", 1), ("b", 2)], [("a", 2), ("b", 1)])
