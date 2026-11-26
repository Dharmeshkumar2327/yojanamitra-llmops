"""Loads the scheme registry (data/sources.yaml)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Scheme:
    id: str
    name: str
    ministry: str = ""
    category: str = ""
    aliases: tuple[str, ...] = field(default_factory=tuple)
    urls: tuple[str, ...] = field(default_factory=tuple)


def load_schemes(path: str | Path | None = None) -> list[Scheme]:
    if path is None:
        path = Path(__file__).resolve().parents[3] / "data" / "sources.yaml"
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    schemes = []
    for s in raw.get("schemes", []):
        schemes.append(
            Scheme(
                id=s["id"],
                name=s["name"],
                ministry=s.get("ministry", ""),
                category=s.get("category", ""),
                aliases=tuple(a.lower() for a in s.get("aliases", [])),
                urls=tuple(s.get("urls", [])),
            )
        )
    return schemes
