"""Prompt registry: Langfuse prompt management in production, local YAML fallback.

Every answer records `prompt_version`, so evals and traces can be compared per version.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)
_VAR = re.compile(r"\{\{\s*(\w+)\s*\}\}")


def render(template: str, variables: dict[str, Any]) -> str:
    return _VAR.sub(lambda m: str(variables.get(m.group(1), m.group(0))), template)


@dataclass
class PromptBundle:
    name: str
    version: int
    system: str
    user_template: str
    source: str  # "langfuse" | "local"
    langfuse_prompt: Any = None  # linked to generations in Langfuse

    def compile(self, **variables: Any) -> list[dict]:
        return [
            {"role": "system", "content": render(self.system, variables)},
            {"role": "user", "content": render(self.user_template, variables)},
        ]


def load_local(
    prompts_dir: Path, version: int | None = None, name: str = "yojanamitra-rag"
) -> PromptBundle:
    files = sorted(prompts_dir.glob("rag_v*.yaml"), key=lambda p: int(p.stem.split("_v")[1]))
    if not files:
        raise FileNotFoundError(f"no prompt files in {prompts_dir}")
    chosen = files[-1]
    if version is not None:
        matches = [f for f in files if f.stem == f"rag_v{version}"]
        if matches:
            chosen = matches[0]
    data = yaml.safe_load(chosen.read_text(encoding="utf-8"))
    return PromptBundle(name, int(data["version"]), data["system"], data["user_template"], "local")


def load_prompt(version: int | None = None) -> PromptBundle:
    """Langfuse (label=production, or a specific version) -> fallback to local YAML."""
    from yojanamitra.config import PROMPTS_DIR, get_settings

    s = get_settings()
    if s.langfuse_enabled:
        try:
            from yojanamitra.observability.tracing import get_langfuse

            lf = get_langfuse()
            if version is not None:
                p = lf.get_prompt(s.prompt_name, version=version, type="chat", cache_ttl_seconds=60)
            else:
                p = lf.get_prompt(
                    s.prompt_name, label=s.prompt_label, type="chat", cache_ttl_seconds=60
                )
            msgs = {m["role"]: m["content"] for m in p.prompt}
            return PromptBundle(
                s.prompt_name, p.version, msgs["system"], msgs["user"], "langfuse", p
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("Langfuse prompt fetch failed (%s) - using local prompt", exc)
    return load_local(
        PROMPTS_DIR, version if version is not None else s.local_prompt_version, s.prompt_name
    )
