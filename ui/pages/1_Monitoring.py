"""Monitoring dashboard: traffic, cost, latency, quality, feedback, gaps, eval history.

Reads the request log (DATABASE_URL - SQLite locally, shared Postgres in production)
and evals/results/run_*.json.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine

ROOT = Path(__file__).resolve().parents[2]
DB_URL = os.getenv("DATABASE_URL", f"sqlite:///{ROOT / 'data' / 'requests.db'}")

st.set_page_config(page_title="YojanaMitra - Monitoring", page_icon="📊", layout="wide")
st.title("📊 YojanaMitra · LLMOps Monitoring")


@st.cache_data(ttl=60)
def load_requests() -> pd.DataFrame:
    try:
        df = pd.read_sql("SELECT * FROM requests", create_engine(DB_URL))
    except Exception:  # noqa: BLE001
        return pd.DataFrame()
    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df["day"] = df["ts"].dt.date
    return df


@st.cache_data(ttl=300)
def load_eval_runs() -> pd.DataFrame:
    rows = []
    for f in sorted((ROOT / "evals" / "results").glob("run_*.json")):
        run = json.loads(f.read_text(encoding="utf-8"))
        rows.append(
            {
                "run": f.stem.replace("run_", ""),
                "prompt_version": run.get("prompt_version"),
                "subset": run.get("subset"),
                **{k: v for k, v in run["summary"].items() if k != "by_type"},
            }
        )
    return pd.DataFrame(rows)


df = load_requests()
tab_live, tab_quality, tab_evals = st.tabs(["Live traffic", "Quality & gaps", "Eval history"])

with tab_live:
    if df.empty:
        st.info("No requests logged yet. Ask a few questions in the chat first.")
    else:
        answered = df[~df["refused"].astype(bool)]
        c = st.columns(6)
        c[0].metric("Requests", len(df))
        c[1].metric("Cache hit rate", f"{df['cache_hit'].astype(bool).mean():.0%}")
        c[2].metric("p95 latency", f"{df['latency_ms'].quantile(0.95) / 1000:.1f}s")
        c[3].metric("Total cost", f"${df['cost_usd'].sum():.4f}")
        cost_1k = 1000 * answered["cost_usd"].mean() if len(answered) else 0
        c[4].metric("Cost / 1k answers", f"${cost_1k:.3f}")
        fb = df["feedback"].dropna()
        c[5].metric("👎 rate", f"{(fb == -1).mean():.0%}" if len(fb) else "—")

        daily = df.groupby("day").agg(
            queries=("request_id", "count"),
            cost_usd=("cost_usd", "sum"),
            p95_latency_s=("latency_ms", lambda s: s.quantile(0.95) / 1000),
        )
        left, right = st.columns(2)
        left.subheader("Queries per day")
        left.bar_chart(daily["queries"])
        right.subheader("Cost per day (USD)")
        right.line_chart(daily["cost_usd"])
        st.subheader("Latency breakdown (avg ms)")
        st.bar_chart(answered[["retrieval_ms", "llm_ms"]].mean())
        st.subheader("Language mix")
        st.bar_chart(df["language"].value_counts())

with tab_quality:
    if df.empty:
        st.info("No data yet.")
    else:
        st.subheader("Refusals by reason")
        st.bar_chart(df[df["refused"].astype(bool)]["refusal_reason"].value_counts())
        st.subheader("🕳️ Knowledge gaps - questions we could not answer")
        st.caption("These tell you which scheme documents to add next.")
        gaps = df[df["refusal_reason"].isin(["low_confidence", "llm_not_found"])]
        st.dataframe(
            gaps[["ts", "question", "schemes", "confidence"]].sort_values("ts", ascending=False),
            use_container_width=True,
            hide_index=True,
        )
        st.subheader("👎 Negative feedback")
        neg = df[df["feedback"] == -1]
        st.dataframe(
            neg[["ts", "question", "answer", "feedback_comment", "prompt_version"]],
            use_container_width=True,
            hide_index=True,
        )
        st.subheader("Confidence distribution (answered)")
        st.bar_chart(pd.cut(df["confidence"], bins=10).astype(str).value_counts().sort_index())
        st.subheader("Top schemes asked about")
        schemes = df["schemes"].fillna("").str.split(",").explode()
        st.bar_chart(schemes[schemes != ""].value_counts().head(15))

with tab_evals:
    runs = load_eval_runs()
    if runs.empty:
        st.info("No eval runs yet - run `python -m evals.run_evals`.")
    else:
        metrics = [
            "faithfulness",
            "correctness",
            "answer_relevancy",
            "hallucination_rate",
            "refusal_accuracy",
            "retrieval_hit_rate",
        ]
        st.line_chart(runs.set_index("run")[[m for m in metrics if m in runs]])
        st.dataframe(runs, use_container_width=True, hide_index=True)
