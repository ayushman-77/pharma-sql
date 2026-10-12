"""Static SQL validation: safety (read-only, single statement) and schema grounding.

Every table and column the model references is checked against the catalog before the
query runs. Unknown identifiers are the classic text-to-SQL hallucination; catching them
here gives the repair step a precise, actionable error ("did you mean net_sales?").
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp

from pharmasql.catalog import Catalog

FORBIDDEN = (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter,
             exp.Command, exp.Pragma, exp.Attach, exp.Detach, exp.Transaction)


@dataclass
class ValidationResult:
    sql: str
    errors: list[str] = field(default_factory=list)
    unknown_tables: list[str] = field(default_factory=list)
    unknown_columns: list[str] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def schema_hallucination(self) -> bool:
        return bool(self.unknown_tables or self.unknown_columns)


def _suggest(name: str, candidates: set[str]) -> str:
    match = difflib.get_close_matches(name, sorted(candidates), n=2, cutoff=0.5)
    return f" Did you mean {' or '.join(match)}?" if match else ""


class SQLValidator:
    def __init__(self, catalog: Catalog):
        self.catalog = catalog
        self.table_names = {t.lower() for t in catalog.tables}

    def validate(self, sql: str) -> ValidationResult:
        sql = sql.strip().rstrip(";").strip()
        result = ValidationResult(sql=sql)
        if not sql:
            result.errors.append("Empty SQL.")
            return result
        try:
            statements = [s for s in sqlglot.parse(sql, read="sqlite") if s is not None]
        except sqlglot.errors.ParseError as e:
            result.errors.append(f"SQL syntax error: {str(e).splitlines()[0]}")
            return result
        if len(statements) != 1:
            result.errors.append("Exactly one SQL statement is allowed.")
            return result
        tree = statements[0]
        if not isinstance(tree, exp.Query) or any(tree.find_all(*FORBIDDEN)):
            result.errors.append("Only read-only SELECT queries are allowed.")
            return result

        alias_to_table = self._check_identifiers(tree, result)
        if result.ok:
            self._check_integer_division(tree, alias_to_table, result)
        return result

    def _check_integer_division(self, tree: exp.Expression, alias_to_table: dict[str, str],
                                result: ValidationResult) -> None:
        """Flag int/int division: SQLite truncates it (3/4 = 0), a silent wrong-answer bug."""

        def column_is_int(col: exp.Column) -> bool:
            tables = ([alias_to_table[col.table.lower()]] if col.table.lower() in alias_to_table
                      else list(alias_to_table.values()))
            types = {self.catalog.column_type(t, col.name) for t in tables} - {None}
            return types == {"INTEGER"}

        def is_int(e: exp.Expression | None) -> bool:
            if e is None:
                return False
            if isinstance(e, exp.Paren | exp.Window | exp.Sum | exp.Max | exp.Min | exp.Abs):
                return is_int(e.this)
            if isinstance(e, exp.Literal):
                return e.is_number and not any(ch in e.this.lower() for ch in ".e")
            if isinstance(e, exp.Column):
                return column_is_int(e)
            if isinstance(e, exp.Count | exp.EQ | exp.NEQ | exp.GT | exp.GTE | exp.LT | exp.LTE):
                return True
            if isinstance(e, exp.Add | exp.Sub | exp.Mul | exp.Mod):
                return is_int(e.left) and is_int(e.right)
            if isinstance(e, exp.Case):
                branches = [i.args.get("true") for i in e.args.get("ifs", [])]
                default = e.args.get("default")
                return all(is_int(b) for b in branches) and (default is None or is_int(default))
            if isinstance(e, exp.Coalesce):
                return is_int(e.this) and all(is_int(x) for x in e.expressions)
            if isinstance(e, exp.Cast):
                return e.to.this in exp.DataType.INTEGER_TYPES
            return False

        for div in tree.find_all(exp.Div):
            if is_int(div.this) and is_int(div.expression):
                result.errors.append(
                    f"Integer division in `{div.sql(dialect='sqlite')}`: SQLite truncates "
                    "integer/integer (e.g. 3/4 = 0). Multiply the numerator by 1.0, or by 100.0 "
                    "for a percentage.")

    def _check_identifiers(self, tree: exp.Expression,
                           result: ValidationResult) -> dict[str, str]:
        cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
        derived_aliases = {sq.alias.lower() for sq in tree.find_all(exp.Subquery) if sq.alias}

        alias_to_table: dict[str, str] = {}
        cte_aliases: set[str] = set()  # e.g. "FROM q cur JOIN q prev" where q is a CTE
        for table in tree.find_all(exp.Table):
            name = table.name.lower()
            if name in cte_names:
                cte_aliases.add((table.alias or table.name).lower())
                continue
            if name not in self.table_names:
                result.unknown_tables.append(table.name)
                result.errors.append(
                    f"Unknown table '{table.name}'.{_suggest(name, self.table_names)}")
                continue
            result.tables.append(name)
            alias_to_table[(table.alias or table.name).lower()] = name

        referenced_columns = set().union(*(self.catalog.columns_of(t) for t in result.tables))
        output_aliases = {a.alias.lower() for a in tree.find_all(exp.Alias)}
        cte_columns = {
            col.lower()
            for cte in tree.find_all(exp.CTE)
            for col in (cte.this.named_selects if isinstance(cte.this, exp.Query) else [])
        }
        known_unqualified = referenced_columns | output_aliases | cte_columns

        for column in tree.find_all(exp.Column):
            name = column.name.lower()
            if not name or name == "*":
                continue
            qualifier = column.table.lower()
            if qualifier:
                table = alias_to_table.get(qualifier)
                if table is None:
                    if qualifier not in cte_names | cte_aliases | derived_aliases | output_aliases:
                        result.unknown_tables.append(qualifier)
                        result.errors.append(f"Unknown table alias '{column.table}'.")
                    continue
                columns = self.catalog.columns_of(table)
                if name not in columns:
                    result.unknown_columns.append(f"{table}.{name}")
                    result.errors.append(
                        f"Column '{name}' does not exist in {table}. "
                        f"{self._locate(name, table, columns)}"
                        f"Columns of {table}: {', '.join(sorted(columns))}.")
            elif name not in known_unqualified:
                result.unknown_columns.append(name)
                tables = sorted(set(result.tables))
                result.errors.append(
                    f"Column '{name}' does not exist in the referenced tables "
                    f"({', '.join(tables) or 'none'}). "
                    + (self._locate(name, tables[0], referenced_columns) if tables else ""))
        result.tables = sorted(set(result.tables))
        self._check_join_keys(tree, alias_to_table, result)
        return alias_to_table

    def _locate(self, column: str, from_table: str, candidates: set[str]) -> str:
        """Hint for a missing column: the join path to a table that has it, else a near-miss."""
        best: list[str] | None = None
        for holder in self.catalog.tables_with_column(column):
            path = self.catalog.join_path(from_table, holder)
            if path and (best is None or len(path) < len(best)):
                best = path
        if best:
            holder = best[-1].split(" = ")[0].split(".")[0]
            joins = " then JOIN ".join(best)
            return f"'{column}' lives in {holder}; reach it with JOIN {joins}. "
        return _suggest(column, candidates).strip() + " " if _suggest(column, candidates) else ""

    def _check_join_keys(self, tree: exp.Expression, alias_to_table: dict[str, str],
                         result: ValidationResult) -> None:
        """Flag joins on two differently-named *_id keys with no FK between them
        (e.g. territory_id = hcp_id): they run, but silently produce garbage."""
        related = set()
        for t, c, rt, rc in self.catalog.foreign_keys():
            related |= {(t, c, rt, rc), (rt, rc, t, c)}
        for join in tree.find_all(exp.Join):
            on = join.args.get("on")
            for eq in on.find_all(exp.EQ) if on else []:
                left, right = eq.this, eq.expression
                if not (isinstance(left, exp.Column) and isinstance(right, exp.Column)):
                    continue
                lt = alias_to_table.get(left.table.lower())
                rt = alias_to_table.get(right.table.lower())
                lc, rc = left.name.lower(), right.name.lower()
                if not (lt and rt) or lc == rc or not (lc.endswith("_id") and rc.endswith("_id")):
                    continue
                if (lt, lc, rt, rc) not in related:
                    path = self.catalog.join_path(lt, rt)
                    hint = f" Use: JOIN {' then JOIN '.join(path)}." if path else ""
                    result.errors.append(
                        f"Suspicious join `{eq.sql(dialect='sqlite')}`: {lt}.{lc} and {rt}.{rc} "
                        f"are unrelated keys.{hint}")
