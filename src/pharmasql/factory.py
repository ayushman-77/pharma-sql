"""Composition root: builds the agent's dependencies from settings in one place."""

from __future__ import annotations

from pharmasql.agent.fewshot import ExampleStore, NomicEmbeddings
from pharmasql.agent.graph import AgentDeps, SQLAgent
from pharmasql.agent.llm import OllamaAnswerWriter, OllamaSQLGenerator
from pharmasql.agent.schemas import AgentConfig
from pharmasql.catalog import Catalog
from pharmasql.config import Settings, get_settings
from pharmasql.executor import SQLExecutor
from pharmasql.semantic import SemanticLayer
from pharmasql.validator import SQLValidator


def build_deps(settings: Settings | None = None, model: str | None = None) -> AgentDeps:
    s = settings or get_settings()
    model = model or s.sql_model
    catalog = Catalog.from_sqlite(s.db_path)
    embeddings = NomicEmbeddings(s.embedding_model, s.ollama_base_url)
    return AgentDeps(
        catalog=catalog,
        semantic=SemanticLayer.load(s.semantic_layer_path),
        validator=SQLValidator(catalog),
        executor=SQLExecutor(s.db_path, s.query_timeout_s, s.max_rows),
        generator=OllamaSQLGenerator(model, s.ollama_base_url, s.num_ctx, s.temperature),
        examples=ExampleStore(s.verified_queries_path, s.vector_dir, embeddings),
        writer=OllamaAnswerWriter(model, s.ollama_base_url, s.num_ctx),
        fewshot_k=s.fewshot_k,
    )


def build_agent(config: AgentConfig | None = None, model: str | None = None,
                settings: Settings | None = None) -> SQLAgent:
    s = settings or get_settings()
    config = config or AgentConfig(max_retries=s.max_retries)
    return SQLAgent(build_deps(s, model), config)
