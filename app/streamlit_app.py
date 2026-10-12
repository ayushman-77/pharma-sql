"""Analyst UI: `uv run pharmasql ui`."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st

from pharmasql.agent.graph import SQLAgent
from pharmasql.agent.schemas import ABLATIONS
from pharmasql.config import PROJECT_ROOT, get_settings
from pharmasql.factory import build_deps

SERIES_BLUE = "#2a78d6"
RUNS_DIR = PROJECT_ROOT / "results" / "runs"
SAMPLES = [
    "What was Glucovia's market share of the diabetes market last quarter?",
    "Which 5 territories had the lowest Cardiozen quota attainment in 2026-Q3?",
    "Show monthly Oncorex TRx since launch.",
    "How did our net sales by brand change quarter over quarter in 2026?",
    "Which payers don't cover Pulmova?",
    "What was our TV advertising spend last quarter?",
]

st.set_page_config(page_title="PharmaSQL Agent", page_icon="💊", layout="wide")


@st.cache_resource(show_spinner="Loading schema, semantic layer and models...")
def get_deps(model: str):
    return build_deps(get_settings(), model)


@st.cache_data(ttl=60)
def installed_models() -> list[str]:
    try:
        tags = httpx.get(f"{get_settings().ollama_base_url}/api/tags", timeout=3).json()
        return [m["name"] for m in tags["models"] if "embed" not in m["name"]]
    except httpx.HTTPError:
        return []


def auto_chart(df: pd.DataFrame) -> None:
    """Pick a chart from the result's shape: time on x -> line, category on x -> bar."""
    if len(df) < 2 or len(df) > 60 or df.shape[1] < 2:
        return
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    labels = [c for c in df.columns if c not in numeric]
    if not numeric or not labels:
        return
    x, y = labels[0], numeric[-1]
    data = df[[x, y]].set_index(x)
    is_time = any(tok in x.lower() for tok in ("month", "quarter", "date", "year", "period"))
    if is_time:
        st.line_chart(data, color=SERIES_BLUE, height=280)
    else:
        st.bar_chart(data.sort_values(y, ascending=False), color=SERIES_BLUE, height=280)


def render_trace(trace: list[dict]) -> None:
    rows = []
    for t in trace:
        detail = {k: v for k, v in t.items() if k not in ("step", "ms", "reasoning", "sql")}
        rows.append({"step": t["step"], "ms": t["ms"],
                     "detail": json.dumps(detail, default=str)[:160]})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def render_answer(state: dict, deps, question: str, key: str) -> None:
    status = state["status"]
    icon = {"answered": "✅", "abstained": "🛑", "failed": "⚠️"}[status]
    st.markdown(f"{icon} **{status.capitalize()}**: {state['answer']}")
    generation = state.get("generation")
    if generation and generation.assumptions:
        st.caption("Assumptions: " + " · ".join(generation.assumptions))
    if status == "answered":
        result = state["result"]
        df = pd.DataFrame(result.rows, columns=result.columns)
        sql_tab, data_tab, chart_tab = st.tabs(["SQL", f"Result ({len(df)} rows)", "Chart"])
        with sql_tab:
            st.code(state["validation"].sql, language="sql")
            if generation:
                st.caption(f"Plan: {generation.reasoning}")
        with data_tab:
            st.dataframe(df, hide_index=True, use_container_width=True)
        with chart_tab:
            auto_chart(df)
        if deps.examples and st.button("👍 Save as verified query", key=f"save-{key}"):
            deps.examples.add(question, state["validation"].sql)
            st.toast("Added to the verified-query bank; future questions can retrieve it.")
    with st.expander(f"Agent trace · {len(state['attempts'])} attempt(s)"):
        for attempt in state["attempts"]:
            if attempt.error:
                st.markdown(f"- ❌ `{attempt.error[:220]}`")
        render_trace(state["trace"])


# ---------------------------------------------------------------- sidebar
settings = get_settings()
with st.sidebar:
    st.title("💊 PharmaSQL")
    st.caption("Commercial analytics questions → validated SQL, on a local open-source LLM.")
    models = installed_models() or [settings.sql_model]
    default = models.index(settings.sql_model) if settings.sql_model in models else 0
    model = st.selectbox("Model (Ollama)", models, index=default)
    config_name = st.selectbox("Agent configuration", list(ABLATIONS), index=len(ABLATIONS) - 1,
                               help="Ablation settings from the benchmark; 'full' is the "
                                    "production agent.")
    st.divider()
    st.markdown("**Try a question**")
    for q in SAMPLES:
        if st.button(q, use_container_width=True):
            st.session_state.pending = q

if not Path(settings.db_path).exists():
    st.error("Database not found. Run `uv run pharmasql generate-data` first.")
    st.stop()

deps = get_deps(model)
agent = SQLAgent(deps, ABLATIONS[config_name].model_copy(
    update={"max_retries": settings.max_retries}))

ask_tab, bench_tab, schema_tab = st.tabs(["Ask", "Benchmark results", "Data model"])

# ---------------------------------------------------------------- ask
with ask_tab:
    st.session_state.setdefault("history", [])
    for i, (question, state) in enumerate(st.session_state.history):
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            render_answer(state, deps, question, key=str(i))

    question = st.chat_input("Ask about prescriptions, sales, quotas, calls, payers...")
    question = question or st.session_state.pop("pending", None)
    if question:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            final: dict = {}
            with st.status("Working...", expanded=True) as status:
                for node, update in agent.stream(question):
                    entry = (update.get("trace") or [{}])[0]
                    note = {
                        "retrieve_examples": f"Retrieved {len(update.get('examples', []))} "
                                             "verified examples",
                        "generate_sql": f"Generated SQL (attempt {entry.get('attempt')})",
                        "validate": "Validated against schema" if entry.get("ok")
                                    else f"Validation failed → repairing: "
                                         f"{(entry.get('errors') or [''])[0][:120]}",
                        "execute": f"Executed: {entry.get('rows')} rows" if entry.get("ok")
                                   else f"Execution error → repairing: {entry.get('error')}",
                        "summarize": "Summarized the answer",
                        "abstain": "Declined: data not available",
                        "fail": "Gave up after the retry budget",
                    }.get(node, node)
                    st.write(f"{note}  ·  {entry.get('ms', 0):.0f} ms")
                    for key, value in update.items():
                        if key in ("trace", "tokens"):
                            final[key] = final.get(key, [] if key == "trace" else 0) + value
                        else:
                            final[key] = value
                status.update(label=f"Done · {final['status']}", state="complete",
                              expanded=False)
            render_answer(final, deps, question, key=str(len(st.session_state.history)))
        st.session_state.history.append((question, final))

# ---------------------------------------------------------------- benchmark
with bench_tab:
    summaries = [json.loads(p.read_text()) for p in sorted(RUNS_DIR.glob("*.summary.json"))]
    if not summaries:
        st.info("No eval runs yet. Run `uv run pharmasql eval --config all`.")
    else:
        table = pd.DataFrame([{
            "run": s["run"], "exec. accuracy %": s["execution_accuracy"],
            "abstention %": s["abstention_accuracy"], "overall %": s["overall_accuracy"],
            "schema halluc. %": s["schema_hallucination_rate"],
            "answered unanswerable %": s["answer_hallucination_rate"],
            "p50 latency s": s["latency_p50_s"],
        } for s in summaries])
        st.dataframe(table, hide_index=True, use_container_width=True)
        st.bar_chart(table.set_index("run")["exec. accuracy %"], color=SERIES_BLUE, height=300,
                     horizontal=True)
        run = st.selectbox("Inspect run", [s["run"] for s in summaries])
        records = [json.loads(line) for line in (RUNS_DIR / f"{run}.jsonl").open()]
        only_failed = st.toggle("Only incorrect", value=True)
        df = pd.DataFrame(records)
        if only_failed:
            df = df[~df["correct"]]
        st.dataframe(df[["id", "difficulty", "status", "attempts", "question", "sql", "errors"]],
                     hide_index=True, use_container_width=True)

# ---------------------------------------------------------------- schema
with schema_tab:
    left, right = st.columns(2)
    with left:
        st.subheader("Star schema")
        st.code(deps.catalog.render_enriched(deps.semantic), language="markdown")
    with right:
        st.subheader("Semantic layer")
        st.code(deps.semantic.render_glossary(), language="markdown")
