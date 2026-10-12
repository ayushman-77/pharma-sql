from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import BaseMessage

from pharmasql.agent.graph import AgentDeps, SQLAgent
from pharmasql.agent.llm import GenerationError
from pharmasql.agent.schemas import AgentConfig, SQLGeneration
from pharmasql.catalog import Catalog
from pharmasql.config import PROJECT_ROOT, get_settings
from pharmasql.executor import SQLExecutor
from pharmasql.semantic import SemanticLayer
from pharmasql.validator import SQLValidator


@pytest.fixture(scope="session")
def db_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Use the generated database if present; otherwise build one (CI)."""
    path = get_settings().db_path
    if path.exists():
        return path
    from pharmasql.data.generate import generate

    path = tmp_path_factory.mktemp("data") / "pharma.db"
    generate(path)
    return path


@pytest.fixture(scope="session")
def catalog(db_path: Path) -> Catalog:
    return Catalog.from_sqlite(db_path)


@pytest.fixture(scope="session")
def semantic() -> SemanticLayer:
    return SemanticLayer.load(PROJECT_ROOT / "config" / "semantic_layer.yaml")


@pytest.fixture(scope="session")
def validator(catalog: Catalog) -> SQLValidator:
    return SQLValidator(catalog)


@pytest.fixture(scope="session")
def executor(db_path: Path) -> SQLExecutor:
    return SQLExecutor(db_path, timeout_s=60, max_rows=None)


class ScriptedGenerator:
    """Returns pre-written generations in order and records every prompt it receives."""

    def __init__(self, outputs: list[SQLGeneration | Exception]):
        self.outputs = list(outputs)
        self.calls: list[list[BaseMessage]] = []

    def generate(self, messages: list[BaseMessage]) -> tuple[SQLGeneration, int]:
        self.calls.append(messages)
        out = self.outputs.pop(0) if len(self.outputs) > 1 else self.outputs[0]
        if isinstance(out, Exception):
            raise out
        return out, 100


def gen(sql: str, answerable: bool = True) -> SQLGeneration:
    return SQLGeneration(reasoning="test", answerable=answerable, sql=sql, assumptions=[])


@pytest.fixture
def make_agent(catalog, semantic, validator, db_path):
    def _make(outputs: list[SQLGeneration | Exception],
              **config) -> tuple[SQLAgent, ScriptedGenerator]:
        generator = ScriptedGenerator(outputs)
        deps = AgentDeps(catalog=catalog, semantic=semantic, validator=validator,
                         executor=SQLExecutor(db_path), generator=generator)
        cfg = AgentConfig(fewshot="none", summarize=False, **config)
        return SQLAgent(deps, cfg), generator

    return _make


__all__ = ["GenerationError", "gen"]
