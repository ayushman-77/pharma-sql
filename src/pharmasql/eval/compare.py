"""Execution-accuracy comparison between a predicted and a gold result set.

Rules (documented in the README):
* Row order is ignored; the gold query's LIMIT already encodes "top N" questions.
* Every gold column must map to a distinct predicted column with matching values;
  extra predicted columns (e.g. a label next to the number) are allowed.
* Numbers match if equal up to float noise, or if one side equals the other rounded to the
  first side's displayed precision (51.26 vs ROUND(.., 1) = 51.3), so ROUND() choices do
  not count as errors. Text is compared exactly after stripping.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from typing import Any

Row = Sequence[Any]


def _decimals(x: float) -> int | None:
    """Displayed decimal places of a value that looks deliberately rounded (<= 4 places)."""
    text = repr(float(x))
    if "e" in text:
        return None
    places = len(text.split(".")[1].rstrip("0"))
    return places if places <= 4 else None


def _numbers_match(a: float, b: float) -> bool:
    if math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-9):
        return True
    # Rounding is only a valid explanation if it loses little information: a truncated
    # integer ratio (0 vs 0.4) must not pass as "rounded".
    if abs(a - b) > 0.05 * max(abs(a), abs(b)):
        return False
    for exact, shown in ((a, b), (b, a)):
        places = _decimals(shown)
        if places is not None and abs(round(exact, places) - shown) < 10 ** -places / 2 + 1e-9:
            return True
    return False


def values_match(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, int | float) and isinstance(b, int | float):
        return _numbers_match(float(a), float(b))
    if isinstance(a, int | float) or isinstance(b, int | float):
        try:
            return values_match(float(a), float(b))
        except (TypeError, ValueError):
            return False
    return str(a).strip() == str(b).strip()


def _rows_match(gold: list[tuple], pred: list[tuple]) -> bool:
    """Multiset equality of rows under tolerant value comparison (greedy matching)."""
    unused = list(range(len(pred)))
    for g in gold:
        for pos, idx in enumerate(unused):
            if all(values_match(x, y) for x, y in zip(g, pred[idx], strict=True)):
                unused.pop(pos)
                break
        else:
            return False
    return True


def _column_values_match(gold_col: list[Any], pred_col: list[Any]) -> bool:
    return _rows_match([(v,) for v in gold_col], [(v,) for v in pred_col])


def results_match(gold_rows: list[Row], pred_rows: list[Row], max_mappings: int = 5000) -> bool:
    if len(gold_rows) != len(pred_rows):
        return False
    if not gold_rows:
        return True
    n_gold, n_pred = len(gold_rows[0]), len(pred_rows[0])
    if n_pred < n_gold:
        return False
    gold_cols = [[r[i] for r in gold_rows] for i in range(n_gold)]
    pred_cols = [[r[j] for r in pred_rows] for j in range(n_pred)]
    candidates = [
        [j for j in range(n_pred) if _column_values_match(gold_cols[i], pred_cols[j])]
        for i in range(n_gold)
    ]
    if any(not c for c in candidates):
        return False
    gold_t = [tuple(r) for r in gold_rows]
    for count, mapping in enumerate(itertools.product(*candidates)):
        if count >= max_mappings:
            break
        if len(set(mapping)) != len(mapping):
            continue
        projected = [tuple(r[j] for j in mapping) for r in pred_rows]
        if _rows_match(gold_t, projected):
            return True
    return False
