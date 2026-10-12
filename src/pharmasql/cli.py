"""Command-line entry point: `uv run pharmasql --help`."""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from pharmasql.config import PROJECT_ROOT, get_settings

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="Pharma text-to-SQL agent on local open-source LLMs.")
console = Console()
RUNS_DIR = PROJECT_ROOT / "results" / "runs"
BENCHMARK = PROJECT_ROOT / "evals" / "benchmark.yaml"


@app.command("generate-data")
def generate_data(seed: int = 42) -> None:
    """Build the synthetic pharma commercial database (data/pharma.db)."""
    from pharmasql.data.generate import generate

    path = get_settings().db_path
    with console.status("Generating synthetic data..."):
        counts = generate(path, seed=seed)
    table = Table("table", "rows", title=str(path))
    for name, n in counts.items():
        table.add_row(name, f"{n:,}")
    console.print(table)


@app.command()
def ask(
    question: str,
    model: Annotated[str | None, typer.Option(help="Ollama model tag")] = None,
    config: Annotated[str, typer.Option(help="Agent config / ablation name")] = "full",
    show_trace: bool = True,
) -> None:
    """Ask a business question in plain English."""
    from pharmasql.agent.schemas import ABLATIONS
    from pharmasql.factory import build_agent

    agent = build_agent(ABLATIONS[config], model=model)
    with console.status("Thinking..."):
        state = agent.ask(question)
    if state.get("result") is not None and state["status"] == "answered":
        console.print(Syntax(state["validation"].sql, "sql", word_wrap=True))
        console.print(state["result"].to_markdown(15))
    console.print(Panel(state["answer"], title=f"[bold]{state['status']}"))
    if show_trace:
        for step in state["trace"]:
            extra = {k: v for k, v in step.items() if k in ("attempt", "errors", "error", "rows")}
            console.print(f"[dim]{step['step']:<18} {step['ms']:>8.0f} ms  {extra or ''}")


@app.command("eval")
def evaluate(
    model: Annotated[str | None, typer.Option(help="Ollama model tag")] = None,
    config: Annotated[str, typer.Option(help="Ablation name, or 'all'")] = "full",
    limit: Annotated[int | None, typer.Option(help="Only the first N questions")] = None,
    ids: Annotated[str | None, typer.Option(help="Comma-separated question ids")] = None,
) -> None:
    """Run the benchmark and write per-question results + a summary to results/runs/."""
    from pharmasql.agent.graph import SQLAgent
    from pharmasql.agent.schemas import ABLATIONS
    from pharmasql.eval.runner import load_benchmark, run_benchmark
    from pharmasql.executor import SQLExecutor
    from pharmasql.factory import build_deps

    settings = get_settings()
    model = model or settings.sql_model
    items = load_benchmark(BENCHMARK)
    if ids:
        wanted = set(ids.split(","))
        items = [i for i in items if i.id in wanted]
    items = items[:limit] if limit else items
    executor = SQLExecutor(settings.db_path, settings.query_timeout_s, max_rows=None)
    deps = build_deps(settings, model)
    configs = list(ABLATIONS) if config == "all" else [config]

    for name in configs:
        cfg = ABLATIONS[name].model_copy(update={"summarize": False,
                                                 "max_retries": settings.max_retries})
        agent = SQLAgent(deps, cfg)
        run_name = f"{model.replace(':', '-').replace('/', '-')}__{name}"
        console.rule(f"{run_name} ({len(items)} questions)")

        def show(r: dict) -> None:
            mark = "[green]✓" if r["correct"] else "[red]✗"
            console.print(f"{mark}[/] {r['id']:<4} {r['status']:<10} attempts={r['attempts']} "
                          f"{r['latency_s']:>6.1f}s  {r['question'][:70]}")

        summary = run_benchmark(agent, items, executor, RUNS_DIR, run_name, on_item=show)
        console.print_json(json.dumps(summary))


@app.command()
def report() -> None:
    """Aggregate all eval runs into results/REPORT.md and charts."""
    from pharmasql.eval.report import build_report

    path = build_report(RUNS_DIR, PROJECT_ROOT / "results")
    console.print(f"Wrote {path}")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Start the FastAPI service."""
    import uvicorn

    uvicorn.run("pharmasql.api:app", host=host, port=port)


@app.command()
def ui(port: int = 8501) -> None:
    """Launch the Streamlit analyst UI."""
    script = PROJECT_ROOT / "app" / "streamlit_app.py"
    raise typer.Exit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(script),
                                      "--server.port", str(port)]))


if __name__ == "__main__":
    app()
