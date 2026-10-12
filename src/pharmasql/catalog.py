"""Schema catalog: introspects the database and renders it for prompts and validation."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from pharmasql.semantic import SemanticLayer

# Text columns with at most this many distinct values get their values listed in the prompt,
# so the model can match filters like region_name = 'Northeast' exactly.
MAX_ENUM_VALUES = 12


@dataclass
class Column:
    name: str
    type: str
    primary_key: bool = False
    references: str | None = None
    values: list[str] = field(default_factory=list)


@dataclass
class Table:
    name: str
    columns: list[Column]
    ddl: str
    row_count: int


class Catalog:
    def __init__(self, tables: dict[str, Table]):
        self.tables = tables

    @classmethod
    def from_sqlite(cls, db_path: Path) -> Catalog:
        con = sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)
        try:
            tables: dict[str, Table] = {}
            for name, ddl in con.execute(
                "SELECT name, sql FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ):
                fks = {row[3]: f"{row[2]}.{row[4]}"
                       for row in con.execute(f"PRAGMA foreign_key_list({name})")}
                columns = []
                for _, col, col_type, _, _, pk in con.execute(f"PRAGMA table_info({name})"):
                    column = Column(col, col_type, bool(pk), fks.get(col))
                    if col_type == "TEXT":
                        distinct = con.execute(
                            f'SELECT DISTINCT "{col}" FROM {name} LIMIT {MAX_ENUM_VALUES + 1}'
                        ).fetchall()
                        if len(distinct) <= MAX_ENUM_VALUES:
                            column.values = sorted(str(v[0]) for v in distinct)
                    columns.append(column)
                count = con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
                tables[name] = Table(name, columns, ddl, count)
        finally:
            con.close()
        return cls(tables)

    def columns_of(self, table: str) -> set[str]:
        t = self.tables.get(table.lower())
        return {c.name.lower() for c in t.columns} if t else set()

    def foreign_keys(self) -> list[tuple[str, str, str, str]]:
        """(table, column, referenced_table, referenced_column) for every declared FK."""
        fks = []
        for t in self.tables.values():
            for c in t.columns:
                if c.references:
                    ref_table, ref_col = c.references.split(".")
                    fks.append((t.name, c.name, ref_table, ref_col))
        return fks

    def join_path(self, src: str, dst: str) -> list[str] | None:
        """Shortest chain of join conditions from src to dst over the FK graph (BFS)."""
        graph: dict[str, list[tuple[str, str]]] = {t: [] for t in self.tables}
        for t, c, rt, rc in self.foreign_keys():
            graph[t].append((rt, f"{rt}.{rc} = {t}.{c}"))
            graph[rt].append((t, f"{t}.{c} = {rt}.{rc}"))
        frontier, seen = [(src, [])], {src}
        while frontier:
            nxt = []
            for node, path in frontier:
                if node == dst:
                    return path
                for neighbour, cond in graph.get(node, []):
                    if neighbour not in seen:
                        seen.add(neighbour)
                        nxt.append((neighbour, [*path, cond]))
            frontier = nxt
        return None

    def tables_with_column(self, column: str) -> list[str]:
        return [t for t in self.tables if column.lower() in self.columns_of(t)]

    def column_type(self, table: str, column: str) -> str | None:
        t = self.tables.get(table.lower())
        for c in t.columns if t else []:
            if c.name.lower() == column.lower():
                return c.type.upper()
        return None

    def render_ddl(self) -> str:
        """Bare CREATE TABLE statements: the zero-shot baseline's view of the schema."""
        return "\n\n".join(t.ddl + ";" for t in self.tables.values())

    def render_enriched(self, semantic: SemanticLayer) -> str:
        """Schema with descriptions, keys and enumerated values (an 'M-Schema'-style view)."""
        blocks = []
        for t in self.tables.values():
            desc = semantic.tables.get(t.name, "")
            lines = [f"## {t.name} ({t.row_count:,} rows) -- {desc}"]
            for c in t.columns:
                parts = [c.type]
                if c.primary_key:
                    parts.append("PK")
                if c.references:
                    parts.append(f"FK -> {c.references}")
                note = semantic.column_description(t.name, c.name)
                if note:
                    parts.append(note)
                elif c.values:
                    parts.append("values: " + ", ".join(repr(v) for v in c.values))
                lines.append(f"  - {c.name}: " + "; ".join(parts))
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)
