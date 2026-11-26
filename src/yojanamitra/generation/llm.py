"""LLM client (Groq via the OpenAI-compatible API) with retries, token + cost accounting."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from functools import lru_cache

from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


@dataclass
class LLMResult:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float

    def json(self) -> dict:
        """Parse a JSON answer (tolerates ```json fences)."""
        t = self.text.strip()
        if t.startswith("```"):
            t = t.strip("`")
            t = t[4:] if t.lower().startswith("json") else t
        start, end = t.find("{"), t.rfind("}")
        return json.loads(t[start : end + 1]) if start != -1 else {}


class LLMClient:
    def __init__(self, api_key: str, base_url: str, price_in: float, price_out: float):
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=60, max_retries=0)
        self.price_in = price_in
        self.price_out = price_out

    def cost(self, tin: int, tout: int) -> float:
        return (tin * self.price_in + tout * self.price_out) / 1_000_000

    @retry(
        retry=retry_if_exception_type((RateLimitError, APIConnectionError, APITimeoutError)),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        stop=stop_after_attempt(6),
        reraise=True,
    )
    def chat(
        self,
        messages: list[dict],
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 700,
        json_mode: bool = False,
    ) -> LLMResult:
        t0 = time.perf_counter()
        kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
        resp = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        latency = (time.perf_counter() - t0) * 1000
        usage = resp.usage
        tin = usage.prompt_tokens if usage else 0
        tout = usage.completion_tokens if usage else 0
        return LLMResult(
            text=resp.choices[0].message.content or "",
            model=model,
            input_tokens=tin,
            output_tokens=tout,
            latency_ms=latency,
            cost_usd=self.cost(tin, tout),
        )


@lru_cache
def get_llm() -> LLMClient:
    from yojanamitra.config import get_settings

    s = get_settings()
    if not s.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not set - copy .env.example to .env and add your key")
    return LLMClient(s.groq_api_key, s.llm_base_url, s.price_input_per_m, s.price_output_per_m)
