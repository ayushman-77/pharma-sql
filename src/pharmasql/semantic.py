"""Semantic layer: business definitions loaded from YAML and rendered for the prompt."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel


class SemanticLayer(BaseModel):
    company: str
    as_of_date: str
    data_coverage: str
    time_conventions: list[str]
    tables: dict[str, str]
    columns: dict[str, str]
    metrics: dict[str, str]
    join_paths: dict[str, str] = {}
    synonyms: dict[str, str]
    rules: list[str]

    @classmethod
    def load(cls, path: Path) -> SemanticLayer:
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))

    def column_description(self, table: str, column: str) -> str | None:
        return self.columns.get(f"{table}.{column}")

    def render_glossary(self) -> str:
        lines = [f"Company: {self.company}", f"Data coverage: {self.data_coverage}", "",
                 "Time conventions:"]
        lines += [f"- {t}" for t in self.time_conventions]
        lines += ["", "Metric definitions:"]
        lines += [f"- {name}: {definition}" for name, definition in self.metrics.items()]
        lines += ["", "Standard join paths:"]
        lines += [f"- {purpose}: {path}" for purpose, path in self.join_paths.items()]
        lines += ["", "Business synonyms:"]
        lines += [f"- {term} -> {meaning}" for term, meaning in self.synonyms.items()]
        lines += ["", "Rules:"]
        lines += [f"- {rule}" for rule in self.rules]
        return "\n".join(lines)
