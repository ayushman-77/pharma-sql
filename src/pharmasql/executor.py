"""Read-only SQL execution with a wall-clock timeout and a row cap."""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ExecutionError(Exception):
    pass


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[tuple[Any, ...]]
    truncated: bool
    elapsed_ms: float

    def to_markdown(self, limit: int = 20) -> str:
        if not self.rows:
            return "(no rows)"
        head = "| " + " | ".join(self.columns) + " |"
        sep = "|" + "---|" * len(self.columns)
        body = ["| " + " | ".join(_fmt(v) for v in row) + " |" for row in self.rows[:limit]]
        more = [f"... {len(self.rows) - limit} more rows"] if len(self.rows) > limit else []
        return "\n".join([head, sep, *body, *more])


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:,.2f}"
    return str(v)


class SQLExecutor:
    def __init__(self, db_path: Path, timeout_s: float = 10.0, max_rows: int | None = 1000):
        if not db_path.exists():
            raise FileNotFoundError(f"{db_path} not found. Run `pharmasql generate-data` first.")
        self.uri = db_path.resolve().as_uri() + "?mode=ro"
        self.timeout_s = timeout_s
        self.max_rows = max_rows

    def run(self, sql: str, max_rows: int | None = -1) -> QueryResult:
        limit = self.max_rows if max_rows == -1 else max_rows
        con = sqlite3.connect(self.uri, uri=True, check_same_thread=False)
        deadline = time.monotonic() + self.timeout_s
        con.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
        start = time.perf_counter()
        try:
            con.execute("PRAGMA query_only = ON")
            cur = con.execute(sql)
            columns = [d[0] for d in cur.description or []]
            rows = cur.fetchall() if limit is None else cur.fetchmany(limit + 1)
        except sqlite3.Error as e:
            msg = str(e)
            if msg == "interrupted":
                msg = f"Query exceeded the {self.timeout_s:.0f}s timeout."
            raise ExecutionError(msg) from e
        finally:
            con.close()
        truncated = limit is not None and len(rows) > limit
        return QueryResult(columns, rows[:limit] if truncated else rows, truncated,
                           (time.perf_counter() - start) * 1000)
