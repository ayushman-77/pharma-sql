"""Benchmark runner: executes the agent over the benchmark and computes metrics.

Metric definitions
------------------
execution_accuracy      answerable questions whose final result set matches the gold result
abstention_accuracy     unanswerable questions where the agent correctly declined
overall_accuracy        correct outcomes over all questions (answer or abstain)
valid_sql_rate          answerable questions that ended with SQL that executed successfully
schema_hallucination    first-attempt SQL referencing a table/column that does not exist
answer_hallucination    unanswerable questions the agent "answered" anyway (fabricated SQL)
false_abstention        answerable questions the agent wrongly declined
silent_wrong            answers that executed fine but were wrong (the dangerous failure mode)
repair_rescues          questions answered correctly only after >= 1 self-correction step
"""

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from pharmasql.agent.graph import EMPTY_RESULT, SQLAgent
from pharmasql.eval.compare import results_match
from pharmasql.executor import SQLExecutor


@dataclass
class BenchmarkItem:
    id: str
    difficulty: str
    question: str
    sql: str | None = None
    alt_sql: list[str] = field(default_factory=list)
    reason: str | None = None

    @property
    def answerable(self) -> bool:
        return self.sql is not None


def load_benchmark(path: Path) -> list[BenchmarkItem]:
    return [BenchmarkItem(**item) for item in yaml.safe_load(path.read_text(encoding="utf-8"))]


def evaluate_item(agent: SQLAgent, item: BenchmarkItem, executor: SQLExecutor) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        state = agent.ask(item.question)
    except Exception as e:  # noqa: BLE001 - an infrastructure error must not abort the whole run
        state = {"status": "failed", "attempts": [], "error": f"{type(e).__name__}: {e}",
                 "tokens": 0, "trace": []}
    latency = time.perf_counter() - start
    status = state["status"]
    attempts = state.get("attempts", [])
    final_sql = state["validation"].sql if status == "answered" else None

    correct = False
    if item.answerable and status == "answered":
        pred = executor.run(final_sql, max_rows=None).rows
        for gold_sql in [item.sql, *item.alt_sql]:
            if results_match(executor.run(gold_sql, max_rows=None).rows, pred):
                correct = True
                break
    elif not item.answerable:
        correct = status == "abstained"

    errors = [a.error for a in attempts]
    real_errors = [e for e in errors if e and e != EMPTY_RESULT]
    return {
        "id": item.id,
        "difficulty": item.difficulty,
        "question": item.question,
        "answerable": item.answerable,
        "status": status,
        "correct": correct,
        "sql": final_sql,
        "attempts": len(attempts),
        "errors": errors,
        "first_attempt_has_sql": bool(attempts and attempts[0].sql.strip()),
        "first_attempt_hallucination": bool(attempts and attempts[0].schema_hallucination),
        "repaired": bool(real_errors) and status == "answered",
        "latency_s": round(latency, 3),
        "tokens": state.get("tokens", 0),
        "infra_error": state.get("error") if not attempts else None,
    }


def _rate(num: int, den: int) -> float | None:
    return round(100 * num / den, 1) if den else None


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    ans = [r for r in records if r["answerable"]]
    unans = [r for r in records if not r["answerable"]]
    answered = [r for r in records if r["status"] == "answered"]
    generated_sql = [r for r in records if r["first_attempt_has_sql"]]
    latencies = sorted(r["latency_s"] for r in records)
    by_diff = {}
    for diff in ("easy", "medium", "hard"):
        rows = [r for r in ans if r["difficulty"] == diff]
        by_diff[diff] = _rate(sum(r["correct"] for r in rows), len(rows))
    return {
        "n": len(records),
        "execution_accuracy": _rate(sum(r["correct"] for r in ans), len(ans)),
        "accuracy_by_difficulty": by_diff,
        "abstention_accuracy": _rate(sum(r["correct"] for r in unans), len(unans)),
        "overall_accuracy": _rate(sum(r["correct"] for r in records), len(records)),
        "valid_sql_rate": _rate(sum(r["status"] == "answered" for r in ans), len(ans)),
        "schema_hallucination_rate": _rate(
            sum(r["first_attempt_hallucination"] for r in generated_sql), len(generated_sql)),
        "answer_hallucination_rate": _rate(
            sum(r["status"] == "answered" for r in unans), len(unans)),
        "false_abstention_rate": _rate(sum(r["status"] == "abstained" for r in ans), len(ans)),
        "silent_wrong_rate": _rate(
            sum(r["answerable"] and not r["correct"] for r in answered), len(answered)),
        "repair_rescues": sum(r["repaired"] and r["correct"] for r in records),
        "avg_attempts": round(statistics.mean(r["attempts"] for r in records), 2),
        "latency_p50_s": round(statistics.median(latencies), 2),
        "latency_p95_s": round(latencies[int(0.95 * (len(latencies) - 1))], 2),
        "avg_tokens": round(statistics.mean(r["tokens"] for r in records)),
        "infra_errors": sum(bool(r["infra_error"]) for r in records),
    }


def run_benchmark(
    agent: SQLAgent,
    items: list[BenchmarkItem],
    executor: SQLExecutor,
    out_dir: Path,
    run_name: str,
    on_item: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    with (out_dir / f"{run_name}.jsonl").open("w", encoding="utf-8") as fh:
        for item in items:
            record = evaluate_item(agent, item, executor)
            records.append(record)
            fh.write(json.dumps(record) + "\n")
            fh.flush()
            if on_item:
                on_item(record)
    summary = {"run": run_name, **summarize(records)}
    (out_dir / f"{run_name}.summary.json").write_text(json.dumps(summary, indent=2))
    return summary
