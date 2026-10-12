"""Agent-loop behaviour with a scripted LLM: no model server required."""

from __future__ import annotations

from conftest import gen

from pharmasql.agent.llm import GenerationError

GOOD = ("SELECT p.brand_name, SUM(s.net_sales) AS net_sales FROM fact_sales s "
        "JOIN dim_product p ON p.product_id = s.product_id GROUP BY p.brand_name")
HALLUCINATED = "SELECT SUM(revenue) FROM fact_sales"


def steps(state) -> list[str]:
    return [t["step"] for t in state["trace"]]


def test_happy_path(make_agent):
    agent, generator = make_agent([gen(GOOD)])
    state = agent.ask("Net sales by brand?")
    assert state["status"] == "answered"
    assert len(state["result"].rows) == 5
    assert len(state["attempts"]) == 1
    assert steps(state) == ["retrieve_examples", "generate_sql", "validate", "execute", "summarize"]


def test_hallucinated_column_is_repaired_with_feedback(make_agent):
    agent, generator = make_agent([gen(HALLUCINATED), gen(GOOD)])
    state = agent.ask("Revenue by brand?")
    assert state["status"] == "answered"
    assert [a.schema_hallucination for a in state["attempts"]] == [True, False]
    repair_prompt = generator.calls[1][-1].content
    assert "does not exist" in repair_prompt and "revenue" in repair_prompt


def test_abstains_without_touching_the_database(make_agent):
    agent, _ = make_agent([gen("", answerable=False)])
    state = agent.ask("What was our TV ad spend?")
    assert state["status"] == "abstained"
    assert "validate" not in steps(state)


def test_without_self_correction_fails_fast(make_agent):
    agent, generator = make_agent([gen(HALLUCINATED), gen(GOOD)], self_correction=False)
    state = agent.ask("Revenue?")
    assert state["status"] == "failed"
    assert len(generator.calls) == 1


def test_retry_budget_is_enforced(make_agent):
    agent, generator = make_agent([gen(HALLUCINATED)], max_retries=2)
    state = agent.ask("Revenue?")
    assert state["status"] == "failed"
    assert len(generator.calls) == 3  # first attempt + 2 repairs


def test_empty_result_triggers_one_repair(make_agent):
    lower = "SELECT COUNT(*) FROM dim_region WHERE region_name = 'northeast' GROUP BY region_name"
    fixed = "SELECT COUNT(*) FROM dim_region WHERE region_name = 'Northeast'"
    agent, generator = make_agent([gen(lower), gen(fixed)])
    state = agent.ask("How many Northeast regions?")
    assert state["status"] == "answered"
    assert state["result"].rows == [(1,)]
    assert "0 rows" in generator.calls[1][-1].content


def test_runtime_error_is_repaired(make_agent):
    # Passes static validation, fails inside SQLite.
    agent, _ = make_agent([gen("SELECT median(decile) FROM dim_hcp"),
                           gen("SELECT AVG(decile) FROM dim_hcp")])
    state = agent.ask("median decile?")
    assert state["status"] == "answered"
    assert "no such function" in state["attempts"][0].error
    assert not state["attempts"][0].schema_hallucination


def test_write_statements_never_execute(make_agent, executor):
    agent, _ = make_agent([gen("DROP TABLE dim_hcp")], self_correction=False)
    state = agent.ask("drop it")
    assert state["status"] == "failed"
    assert executor.run("SELECT COUNT(*) FROM dim_hcp").rows == [(1200,)]


def test_malformed_model_output_is_retried(make_agent):
    agent, generator = make_agent([GenerationError("bad json"), gen(GOOD)])
    state = agent.ask("Net sales by brand?")
    assert state["status"] == "answered"
    assert len(generator.calls) == 2


def test_retry_after_malformed_output_changes_the_prompt(make_agent):
    agent, generator = make_agent([GenerationError("cut off"), gen(GOOD)])
    agent.ask("Net sales by brand?")
    assert "not valid JSON" in generator.calls[1][-1].content
