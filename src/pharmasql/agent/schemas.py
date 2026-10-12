"""Typed contracts for the agent: structured LLM output, feature flags, and graph state."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, Field

from pharmasql.executor import QueryResult
from pharmasql.validator import ValidationResult


class SQLGeneration(BaseModel):
    """Schema the LLM is constrained to (Ollama JSON-schema structured outputs).

    Field order matters: `reasoning` is decoded first, so it acts as a short plan
    the SQL is then conditioned on.
    """

    reasoning: str = Field(
        description="Brief plan, max 3 sentences: tables, joins, filters, metric definitions."
    )
    answerable: bool = Field(
        description="false if the database does not contain the data needed to answer."
    )
    sql: str = Field(description="One SQLite SELECT statement, or '' when not answerable.")
    assumptions: list[str] = Field(
        default_factory=list,
        description="Interpretations made, e.g. \"'revenue' = net sales\". May be empty.",
    )


class Example(BaseModel):
    question: str
    sql: str = ""
    reason: str = ""  # why the question is not answerable, when sql is empty


class AgentConfig(BaseModel):
    """Feature flags; the ablation study toggles these one at a time."""

    name: str = "full"
    semantic_layer: bool = True
    fewshot: Literal["none", "static", "dynamic"] = "dynamic"
    self_correction: bool = True
    max_retries: int = 3
    summarize: bool = True


ABLATIONS: dict[str, AgentConfig] = {
    "zero_shot": AgentConfig(name="zero_shot", semantic_layer=False, fewshot="none",
                             self_correction=False),
    "semantic_layer": AgentConfig(name="semantic_layer", fewshot="none", self_correction=False),
    "static_fewshot": AgentConfig(name="static_fewshot", fewshot="static", self_correction=False),
    "dynamic_fewshot": AgentConfig(name="dynamic_fewshot", fewshot="dynamic",
                                   self_correction=False),
    "full": AgentConfig(name="full"),
}

Status = Literal["answered", "abstained", "failed"]


class Attempt(BaseModel):
    sql: str
    error: str | None = None
    schema_hallucination: bool = False


class AgentState(TypedDict, total=False):
    question: str
    examples: list[Example]
    generation: SQLGeneration | None
    validation: ValidationResult | None
    result: QueryResult | None
    error: str | None
    empty_retry_used: bool
    attempts: list[Attempt]
    status: Status
    answer: str
    tokens: Annotated[int, operator.add]
    trace: Annotated[list[dict[str, Any]], operator.add]
