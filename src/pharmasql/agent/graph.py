"""The agent loop as a LangGraph state machine.

    retrieve_examples -> generate_sql --(not answerable)--> abstain
                              |
                              v
                          validate --(error, retries left)--> generate_sql
                              |                    \\--(error, no retries)--> fail
                              v
                          execute ---(error / empty, retries left)--> generate_sql
                              |                    \\--(error, no retries)--> fail
                              v
                          summarize -> END

Every node appends a timed entry to `trace`, which the UI renders and the eval harness
uses for per-step metrics (attempts, schema hallucinations, repair success).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

from langgraph.graph import END, START, StateGraph

from pharmasql.agent import prompts
from pharmasql.agent.fewshot import ExampleStore
from pharmasql.agent.llm import AnswerWriter, GenerationError, SQLGenerator
from pharmasql.agent.schemas import AgentConfig, AgentState, Attempt, SQLGeneration
from pharmasql.catalog import Catalog
from pharmasql.executor import ExecutionError, SQLExecutor
from pharmasql.semantic import SemanticLayer
from pharmasql.validator import SQLValidator

FEWSHOT_K = 4
EMPTY_RESULT = "EMPTY_RESULT"  # sentinel error: query ran but matched no rows


@dataclass
class AgentDeps:
    catalog: Catalog
    semantic: SemanticLayer
    validator: SQLValidator
    executor: SQLExecutor
    generator: SQLGenerator
    examples: ExampleStore | None = None
    writer: AnswerWriter | None = None
    fewshot_k: int = FEWSHOT_K


def _timed(step: str, fn: Callable[[AgentState], dict[str, Any]]):
    def node(state: AgentState) -> dict[str, Any]:
        start = time.perf_counter()
        update = fn(state)
        detail = update.pop("_detail", {})
        entry = {"step": step, "ms": round((time.perf_counter() - start) * 1000, 1), **detail}
        update["trace"] = [entry]
        return update

    return node


class SQLAgent:
    def __init__(self, deps: AgentDeps, config: AgentConfig | None = None):
        self.deps = deps
        self.config = config or AgentConfig()
        self._schema_text = (deps.catalog.render_enriched(deps.semantic)
                             if self.config.semantic_layer else deps.catalog.render_ddl())
        self._glossary = deps.semantic.render_glossary() if self.config.semantic_layer else None
        self.graph = self._build().compile()

    # ---- public API ------------------------------------------------------------------
    def ask(self, question: str) -> AgentState:
        return self.graph.invoke(self._initial(question))

    def stream(self, question: str) -> Iterator[tuple[str, dict[str, Any]]]:
        """Yield (node_name, state_update) as each step finishes, for live UIs."""
        for chunk in self.graph.stream(self._initial(question), stream_mode="updates"):
            yield from chunk.items()

    @staticmethod
    def _initial(question: str) -> AgentState:
        return {"question": question, "attempts": [], "trace": [], "tokens": 0,
                "empty_retry_used": False}

    # ---- graph -------------------------------------------------------------------------
    def _build(self) -> StateGraph:
        g = StateGraph(AgentState)
        g.add_node("retrieve_examples", _timed("retrieve_examples", self._retrieve_examples))
        g.add_node("generate_sql", _timed("generate_sql", self._generate_sql))
        g.add_node("validate", _timed("validate", self._validate))
        g.add_node("execute", _timed("execute", self._execute))
        g.add_node("summarize", _timed("summarize", self._summarize))
        g.add_node("abstain", _timed("abstain", self._abstain))
        g.add_node("fail", _timed("fail", self._fail))

        g.add_edge(START, "retrieve_examples")
        g.add_edge("retrieve_examples", "generate_sql")
        g.add_conditional_edges("generate_sql", self._after_generate,
                                ["validate", "abstain", "generate_sql", "fail"])
        g.add_conditional_edges("validate", self._after_validate,
                                ["execute", "generate_sql", "fail"])
        g.add_conditional_edges("execute", self._after_execute,
                                ["summarize", "generate_sql", "fail"])
        for terminal in ("summarize", "abstain", "fail"):
            g.add_edge(terminal, END)
        return g

    def _can_retry(self, state: AgentState) -> bool:
        return self.config.self_correction and len(state["attempts"]) <= self.config.max_retries

    def _after_generate(self, state: AgentState) -> str:
        if state.get("generation") is None:
            return "generate_sql" if self._can_retry(state) else "fail"
        return "validate" if state["generation"].answerable else "abstain"

    def _after_validate(self, state: AgentState) -> str:
        if state["validation"].ok:
            return "execute"
        return "generate_sql" if self._can_retry(state) else "fail"

    def _after_execute(self, state: AgentState) -> str:
        if state.get("error"):
            return "generate_sql" if self._can_retry(state) else "fail"
        # _execute flags an empty result for one repair attempt (decided before the update).
        if state["attempts"][-1].error == EMPTY_RESULT:
            return "generate_sql"
        return "summarize"

    # ---- nodes ---------------------------------------------------------------------------
    def _retrieve_examples(self, state: AgentState) -> dict[str, Any]:
        store, mode = self.deps.examples, self.config.fewshot
        if store is None or mode == "none":
            examples = []
        elif mode == "static":
            examples = store.static(self.deps.fewshot_k)
        else:
            examples = store.similar(state["question"], self.deps.fewshot_k)
        return {"examples": examples,
                "_detail": {"mode": mode, "examples": [e.question for e in examples]}}

    def _generate_sql(self, state: AgentState) -> dict[str, Any]:
        attempts = list(state["attempts"])
        # Only the latest failed attempt is replayed: the model repairs its own query with the
        # exact error, without the prompt growing on every retry.
        history: list[tuple[SQLGeneration, str]] = []
        prev = state.get("generation")
        if attempts and prev is not None:
            last = attempts[-1]
            feedback = (prompts.empty_result_feedback(last.sql) if last.error == EMPTY_RESULT
                        else prompts.repair_feedback(last.sql, last.error or "unknown error"))
            history.append((prev, feedback))
        # After malformed output there is no generation to replay; change the prompt so a
        # deterministic (temperature 0, fixed seed) model doesn't repeat the same failure.
        note = prompts.MALFORMED_OUTPUT_NOTE if attempts and prev is None else None
        messages = prompts.build_messages(
            question=state["question"], schema=self._schema_text,
            as_of_date=self.deps.semantic.as_of_date, glossary=self._glossary,
            examples=state.get("examples", []), history=history, note=note,
        )
        try:
            generation, tokens = self.deps.generator.generate(messages)
        except GenerationError as e:
            attempts.append(Attempt(sql="", error=str(e)))
            return {"generation": None, "attempts": attempts, "error": str(e),
                    "_detail": {"attempt": len(attempts), "error": str(e)}}
        attempts.append(Attempt(sql=generation.sql))
        return {
            "generation": generation, "attempts": attempts, "tokens": tokens,
            "error": None, "validation": None, "result": None,
            "_detail": {"attempt": len(attempts), "answerable": generation.answerable,
                        "reasoning": generation.reasoning, "sql": generation.sql,
                        "assumptions": generation.assumptions, "tokens": tokens},
        }

    def _validate(self, state: AgentState) -> dict[str, Any]:
        validation = self.deps.validator.validate(state["generation"].sql)
        attempts = list(state["attempts"])
        if not validation.ok:
            attempts[-1] = attempts[-1].model_copy(update={
                "error": "; ".join(validation.errors),
                "schema_hallucination": validation.schema_hallucination})
        return {"validation": validation, "attempts": attempts,
                "error": None if validation.ok else attempts[-1].error,
                "_detail": {"ok": validation.ok, "errors": validation.errors,
                            "tables": validation.tables}}

    def _execute(self, state: AgentState) -> dict[str, Any]:
        sql = state["validation"].sql
        attempts = list(state["attempts"])
        try:
            result = self.deps.executor.run(sql)
        except ExecutionError as e:
            msg = str(e)
            attempts[-1] = attempts[-1].model_copy(update={
                "error": msg,
                "schema_hallucination": msg.startswith(("no such table", "no such column"))})
            return {"result": None, "error": msg, "attempts": attempts,
                    "_detail": {"ok": False, "error": msg}}
        update: dict[str, Any] = {"result": result, "error": None,
                                  "_detail": {"ok": True, "rows": len(result.rows),
                                              "db_ms": round(result.elapsed_ms, 1)}}
        no_data = not result.rows or all(v is None for row in result.rows for v in row)
        if no_data and self._can_retry(state) and not state["empty_retry_used"]:
            attempts[-1] = attempts[-1].model_copy(update={"error": EMPTY_RESULT})
            update.update(attempts=attempts, empty_retry_used=True)
        return update

    def _summarize(self, state: AgentState) -> dict[str, Any]:
        result = state["result"]
        if not self.config.summarize or self.deps.writer is None:
            answer = f"Query returned {len(result.rows)} row(s)."
            return {"status": "answered", "answer": answer, "_detail": {}}
        prompt = prompts.ANSWER_TEMPLATE.format(
            question=state["question"], sql=state["validation"].sql, n_rows=len(result.rows),
            truncated=", truncated" if result.truncated else "", table=result.to_markdown(30))
        answer, tokens = self.deps.writer.write(prompt)
        return {"status": "answered", "answer": answer, "tokens": tokens,
                "_detail": {"tokens": tokens}}

    def _abstain(self, state: AgentState) -> dict[str, Any]:
        reason = state["generation"].reasoning
        return {"status": "abstained",
                "answer": f"This can't be answered from the available data. {reason}",
                "_detail": {"reason": reason}}

    def _fail(self, state: AgentState) -> dict[str, Any]:
        error = state.get("error") or "unknown error"
        return {"status": "failed",
                "answer": f"I couldn't produce a valid query after "
                          f"{len(state['attempts'])} attempt(s). Last error: {error}",
                "_detail": {"error": error}}
