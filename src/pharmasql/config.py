"""Runtime settings, overridable via environment variables prefixed with PHARMASQL_."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PHARMASQL_", env_file=PROJECT_ROOT / ".env", extra="ignore"
    )

    db_path: Path = PROJECT_ROOT / "data" / "pharma.db"
    semantic_layer_path: Path = PROJECT_ROOT / "config" / "semantic_layer.yaml"
    verified_queries_path: Path = PROJECT_ROOT / "config" / "verified_queries.yaml"
    vector_dir: Path = PROJECT_ROOT / ".cache" / "chroma"

    ollama_base_url: str = "http://localhost:11434"
    sql_model: str = "qwen2.5-coder:7b"
    embedding_model: str = "nomic-embed-text"
    num_ctx: int = 8192
    temperature: float = 0.0

    max_retries: int = 3
    fewshot_k: int = 4
    max_rows: int = 1000
    query_timeout_s: float = 10.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
