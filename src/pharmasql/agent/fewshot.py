"""Verified-query bank with semantic retrieval (dynamic few-shot prompting).

Analysts' verified question/SQL pairs live in YAML (reviewable in a PR) and are indexed in a
persistent Chroma collection. At question time the k most similar examples are pulled into
the prompt. New verified answers from the UI are appended, so retrieval improves with use.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import chromadb
import yaml
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings

from pharmasql.agent.schemas import Example


class NomicEmbeddings(Embeddings):
    """nomic-embed-text is trained with task prefixes; using them improves retrieval."""

    def __init__(self, model: str, base_url: str):
        self._inner = OllamaEmbeddings(model=model, base_url=base_url)
        self._prefix = model.startswith("nomic")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self._prefix:
            texts = [f"search_document: {t}" for t in texts]
        return self._inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._inner.embed_query(f"search_query: {text}" if self._prefix else text)


def load_examples(path: Path) -> list[Example]:
    return [Example(**item) for item in yaml.safe_load(path.read_text(encoding="utf-8"))]


class ExampleStore:
    def __init__(self, yaml_path: Path, persist_dir: Path, embeddings: Embeddings):
        self.yaml_path = yaml_path
        self.persist_dir = persist_dir
        self.embeddings = embeddings
        self.examples = load_examples(yaml_path)
        self._store: Chroma | None = None

    def static(self, k: int) -> list[Example]:
        """A fixed, hand-picked set (the first k entries) for the static few-shot ablation."""
        return self.examples[:k]

    def similar(self, question: str, k: int) -> list[Example]:
        docs = self._vectorstore().similarity_search(question, k=k)
        return [Example(question=d.page_content, sql=d.metadata["sql"],
                        reason=d.metadata.get("reason", "")) for d in docs]

    def add(self, question: str, sql: str) -> None:
        self.examples.append(Example(question=question, sql=sql))
        dumped = [e.model_dump(exclude_defaults=True) for e in self.examples]
        self.yaml_path.write_text(yaml.safe_dump(dumped, sort_keys=False, width=100),
                                  encoding="utf-8")
        self._store = None  # re-index lazily on next query

    def _vectorstore(self) -> Chroma:
        if self._store is not None:
            return self._store
        payload = "\n".join(e.model_dump_json() for e in self.examples)
        name = "verified_" + hashlib.sha256(payload.encode()).hexdigest()[:16]
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(self.persist_dir))
        existing = {c.name if hasattr(c, "name") else str(c) for c in client.list_collections()}
        for stale in existing - {name}:
            client.delete_collection(stale)
        store = Chroma(client=client, collection_name=name, embedding_function=self.embeddings,
                       collection_metadata={"hnsw:space": "cosine"})
        if name not in existing:
            store.add_texts(
                texts=[e.question for e in self.examples],
                metadatas=[{"sql": e.sql, "reason": e.reason} for e in self.examples],
                ids=[f"ex-{i}" for i in range(len(self.examples))],
            )
        self._store = store
        return store
