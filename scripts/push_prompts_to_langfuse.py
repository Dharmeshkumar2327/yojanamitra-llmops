"""Push local prompt versions (prompts/rag_v*.yaml) to Langfuse prompt management.

    python scripts/push_prompts_to_langfuse.py                 # push all, label newest 'production'
    python scripts/push_prompts_to_langfuse.py --production 2  # roll back: make v2 production

Langfuse assigns its own version numbers in push order, so push in order v1, v2, v3
on a fresh project to keep numbers aligned with the YAML files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--production", type=int, default=None, help="YAML version to label production")
    args = ap.parse_args()

    from yojanamitra.config import get_settings
    from yojanamitra.observability.tracing import get_langfuse

    s = get_settings()
    if not s.langfuse_enabled:
        raise SystemExit("Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in .env first")
    lf = get_langfuse()

    files = sorted((ROOT / "prompts").glob("rag_v*.yaml"), key=lambda p: int(p.stem.split("_v")[1]))
    newest = int(files[-1].stem.split("_v")[1])
    prod = args.production or newest

    for f in files:
        data = yaml.safe_load(f.read_text(encoding="utf-8"))
        v = int(data["version"])
        labels = ["production"] if v == prod else ["staging"]
        lf.create_prompt(
            name=s.prompt_name,
            type="chat",
            prompt=[
                {"role": "system", "content": data["system"]},
                {"role": "user", "content": data["user_template"]},
            ],
            labels=labels,
            config={"model": s.llm_model, "temperature": s.llm_temperature, "yaml_version": v},
            commit_message=f"rag_v{v}.yaml",
        )
        print(f"pushed rag_v{v}.yaml labels={labels}")
    lf.flush()


if __name__ == "__main__":
    main()
