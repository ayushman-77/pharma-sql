"""Prompt construction. Kept separate from the graph so prompts are easy to diff and test."""

from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from pharmasql.agent.schemas import Example, SQLGeneration

SYSTEM_TEMPLATE = """\
You are a senior pharma commercial analytics engineer. You translate business questions \
into SQLite queries over the company's commercial data mart. Today is {as_of_date}.

# Database schema
{schema}
{glossary_block}
# Instructions
- Plan briefly in `reasoning` (max 3 sentences): tables, joins, filters, metric definitions.
- If the database does not contain the data needed (e.g. a metric, entity or attribute that \
does not exist in the schema), set `answerable` to false, leave `sql` empty, and say what is \
missing in `reasoning`. Never invent tables or columns.
- Otherwise write ONE SQLite SELECT statement (CTEs allowed) answering exactly what was asked.
- Use only the tables and columns listed above, with explicit JOINs and table aliases.
- Return only the columns needed to answer, with readable column aliases.
- Record any interpretation you made in `assumptions`.
{examples_block}"""

ANSWER_TEMPLATE = """\
Question: {question}

SQL that was run:
{sql}

Result ({n_rows} rows{truncated}):
{table}

Write a concise answer (1-3 sentences) for a pharma business stakeholder. Quote the key \
numbers from the result exactly (format dollars like $1,234,567 and keep percentages as given). \
Do not add numbers that are not in the result. If the result is empty, say no data matched."""


def render_examples(examples: list[Example]) -> str:
    if not examples:
        return ""
    parts = ["\n# Verified example queries (same database)"]
    for ex in examples:
        if ex.sql.strip():
            parts.append(f"Question: {ex.question}\nSQL:\n{ex.sql.strip()}\n")
        else:
            parts.append(f"Question: {ex.question}\nNot answerable: {ex.reason}\n")
    return "\n".join(parts)


def build_messages(
    *,
    question: str,
    schema: str,
    as_of_date: str,
    glossary: str | None,
    examples: list[Example],
    history: list[tuple[SQLGeneration, str]],
    note: str | None = None,
) -> list[BaseMessage]:
    """System prompt + question, followed by one (previous output, error) turn per repair."""
    glossary_block = f"\n# Business glossary\n{glossary}\n" if glossary else ""
    system = SYSTEM_TEMPLATE.format(
        as_of_date=as_of_date,
        schema=schema,
        glossary_block=glossary_block,
        examples_block=render_examples(examples),
    )
    messages: list[BaseMessage] = [SystemMessage(system), HumanMessage(question)]
    for generation, feedback in history:
        messages.append(AIMessage(generation.model_dump_json()))
        messages.append(HumanMessage(feedback))
    if note:
        messages.append(HumanMessage(note))
    return messages


MALFORMED_OUTPUT_NOTE = (
    "Your previous reply was not valid JSON for the schema (it was malformed or cut off). "
    "Reply again with compact JSON only: reasoning in at most 3 short sentences, one SQL query, "
    "and at most 3 short assumptions.")


def repair_feedback(sql: str, error: str) -> str:
    return (f"That query failed.\nSQL:\n{sql}\nError: {error}\n"
            "Fix the root cause while still answering the original question exactly; do not "
            "change what a filter or metric means just to make the query run. If the data needed "
            "does not exist, set answerable to false. Return the complete corrected JSON.")


def empty_result_feedback(sql: str) -> str:
    return (f"That query ran but returned no data (0 rows or only NULLs).\nSQL:\n{sql}\n"
            "Check that text filters use the exact values listed in the schema (case-sensitive), "
            "dates use 'YYYY-MM-DD' month starts, and joins are correct. If the database simply "
            "does not hold this data, set answerable to false. If an empty result is genuinely "
            "correct, return the same SQL unchanged.")
