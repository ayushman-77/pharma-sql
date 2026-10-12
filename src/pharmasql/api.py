"""REST API: `uv run pharmasql serve`, then open http://127.0.0.1:8000/docs."""

from __future__ import annotations

import time
from functools import lru_cache
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from pharmasql.agent.graph import SQLAgent
from pharmasql.agent.schemas import ABLATIONS
from pharmasql.config import get_settings

app = FastAPI(
    title="Pharma Text-to-SQL Agent",
    version="0.1.0",
    description="Ask commercial analytics questions in plain English; get validated SQL, "
                "the result set, and a short answer, all from a local open-source LLM.",
)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500,
                          examples=["What was Glucovia's market share last quarter?"])
    config: Literal["zero_shot", "semantic_layer", "static_fewshot", "dynamic_fewshot",
                    "full"] = "full"


class AskResponse(BaseModel):
    status: Literal["answered", "abstained", "failed"]
    answer: str
    sql: str | None
    columns: list[str]
    rows: list[list[Any]]
    truncated: bool
    reasoning: str | None
    assumptions: list[str]
    attempts: int
    latency_ms: float
    trace: list[dict[str, Any]]


@lru_cache(maxsize=8)
def _agent_for(config: str) -> SQLAgent:
    from pharmasql.factory import build_agent

    return build_agent(ABLATIONS[config].model_copy(
        update={"max_retries": get_settings().max_retries}))


def get_agent_factory():
    """Dependency hook; tests override this with a scripted agent."""
    return _agent_for


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": get_settings().sql_model}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, factory: Annotated[Any, Depends(get_agent_factory)]) -> AskResponse:
    agent = factory(req.config)
    start = time.perf_counter()
    try:
        state = agent.ask(req.question)
    except Exception as e:  # noqa: BLE001 - surface model-server outages as 503s
        raise HTTPException(503, f"LLM backend unavailable: {e}") from e
    result = state.get("result") if state["status"] == "answered" else None
    generation = state.get("generation")
    return AskResponse(
        status=state["status"],
        answer=state["answer"],
        sql=state["validation"].sql if result is not None else None,
        columns=result.columns if result else [],
        rows=[list(r) for r in result.rows] if result else [],
        truncated=result.truncated if result else False,
        reasoning=generation.reasoning if generation else None,
        assumptions=generation.assumptions if generation else [],
        attempts=len(state["attempts"]),
        latency_ms=round((time.perf_counter() - start) * 1000, 1),
        trace=state["trace"],
    )
