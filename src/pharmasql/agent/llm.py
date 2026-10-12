"""LLM adapters. The graph depends on these small protocols, not on a vendor SDK,
so tests inject scripted fakes and the backend can be swapped (Ollama, vLLM, ...)."""

from __future__ import annotations

from typing import Any, Protocol

import httpx
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_ollama import ChatOllama

from pharmasql.agent.schemas import SQLGeneration

THINKING_MODEL_PREFIXES = ("qwen3", "deepseek-r1", "gpt-oss", "magistral")


class GenerationError(Exception):
    pass


class SQLGenerator(Protocol):
    def generate(self, messages: list[BaseMessage]) -> tuple[SQLGeneration, int]: ...


class AnswerWriter(Protocol):
    def write(self, prompt: str) -> tuple[str, int]: ...


# Constrained JSON decoding can occasionally loop (runaway whitespace / repeated list items).
# A token cap plus a client timeout turns that into a normal, retryable GenerationError.
MAX_OUTPUT_TOKENS = 900
REQUEST_TIMEOUT_S = 120


def _chat_model(model: str, base_url: str, num_ctx: int, temperature: float) -> ChatOllama:
    kwargs: dict[str, Any] = {"model": model, "base_url": base_url, "num_ctx": num_ctx,
                              "temperature": temperature, "seed": 42,
                              "num_predict": MAX_OUTPUT_TOKENS,
                              "client_kwargs": {"timeout": REQUEST_TIMEOUT_S}}
    if model.startswith(THINKING_MODEL_PREFIXES):
        # Hybrid reasoning models: disable the hidden <think> phase. The schema's
        # `reasoning` field already gives a short, visible plan at a fraction of the latency.
        kwargs["reasoning"] = False
    return ChatOllama(**kwargs)


def _tokens(message: Any) -> int:
    usage = getattr(message, "usage_metadata", None) or {}
    return int(usage.get("total_tokens", 0))


class OllamaSQLGenerator:
    """Structured generation via Ollama's JSON-schema constrained decoding."""

    def __init__(self, model: str, base_url: str, num_ctx: int = 8192, temperature: float = 0.0):
        self.model = model
        llm = _chat_model(model, base_url, num_ctx, temperature)
        self._structured = llm.with_structured_output(
            SQLGeneration, method="json_schema", include_raw=True
        )

    def generate(self, messages: list[BaseMessage]) -> tuple[SQLGeneration, int]:
        try:
            out = self._structured.invoke(messages)
        except httpx.TimeoutException as e:
            raise GenerationError(f"LLM request timed out after {REQUEST_TIMEOUT_S}s") from e
        parsed = out.get("parsed")
        if parsed is None:
            raise GenerationError(
                f"Output did not match the JSON schema: {out.get('parsing_error')}")
        return parsed, _tokens(out.get("raw"))


class OllamaAnswerWriter:
    def __init__(self, model: str, base_url: str, num_ctx: int = 8192):
        self._llm = _chat_model(model, base_url, num_ctx, temperature=0.1)

    def write(self, prompt: str) -> tuple[str, int]:
        message = self._llm.invoke([HumanMessage(prompt)])
        return str(message.content).strip(), _tokens(message)
