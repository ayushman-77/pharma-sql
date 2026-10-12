# Convenience targets (Linux/macOS). On Windows, run the `uv run ...` commands directly.
.PHONY: setup data test lint eval ablation models report serve ui clean

setup:            ## install dependencies and pull the models
	uv sync
	ollama pull qwen2.5-coder:7b
	ollama pull nomic-embed-text

data:             ## build data/pharma.db (deterministic, seed 42)
	uv run pharmasql generate-data

test:             ## offline tests (no LLM needed)
	uv run pytest

lint:
	uv run ruff check src tests app

eval:             ## full agent on the 80-question benchmark
	uv run pharmasql eval --config full

ablation:         ## all five ablation configs
	uv run pharmasql eval --config all

models:           ## compare other local models on the full agent
	uv run pharmasql eval --config full --model qwen3.5:4b
	uv run pharmasql eval --config full --model llama3.2:3b

report:           ## results/REPORT.md + charts
	uv run pharmasql report

serve:            ## REST API on :8000 (docs at /docs)
	uv run pharmasql serve

ui:               ## Streamlit app on :8501
	uv run pharmasql ui

clean:
	rm -rf .cache .pytest_cache .ruff_cache
