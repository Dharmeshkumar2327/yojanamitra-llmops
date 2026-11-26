"""YojanaMitra chat UI (Streamlit).

Two modes:
  * API mode   (default): talks to the FastAPI backend at API_URL
  * Local mode (YOJANAMITRA_LOCAL=1): runs the pipeline in-process - used for the
    all-in-one Hugging Face Space (no separate backend, no cold starts)

    streamlit run ui/app.py
"""

from __future__ import annotations

import os
import uuid

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
LOCAL = os.getenv("YOJANAMITRA_LOCAL", "0") == "1"

st.set_page_config(
    page_title="YojanaMitra - Govt Scheme Assistant", page_icon="🇮🇳", layout="centered"
)


# ------------------------------------------------------------------ backend calls
@st.cache_resource(show_spinner="Loading models (first time only)...")
def _local_pipeline():
    from yojanamitra.generation.factory import build_pipeline

    return build_pipeline()


def ask(question: str, scheme_ids: list[str] | None) -> dict:
    if LOCAL:
        r = _local_pipeline().answer(
            question, session_id=st.session_state.session_id, scheme_ids=scheme_ids
        )
        return {
            "request_id": r.request_id,
            "answer": r.answer,
            "citations": [c.__dict__ for c in r.citations],
            "refused": r.refused,
            "cache_hit": r.cache_hit,
            "latency_ms": r.latency_ms,
            "cost_usd": r.cost_usd,
            "prompt_version": r.prompt_version,
            "pii_masked": r.pii_found,
        }
    resp = requests.post(
        f"{API_URL}/ask",
        json={
            "question": question,
            "session_id": st.session_state.session_id,
            "scheme_ids": scheme_ids,
        },
        timeout=120,
    )
    if resp.status_code == 429:
        return {
            "answer": "Too many requests - please wait a minute.",
            "citations": [],
            "request_id": None,
        }
    resp.raise_for_status()
    return resp.json()


def send_feedback(request_id: str, rating: int) -> None:
    if LOCAL:
        from yojanamitra.monitoring.request_log import get_request_log

        found, trace_id = get_request_log().set_feedback(request_id, rating, None)
        if found:
            _local_pipeline().tracer.score(trace_id, "user_feedback", rating)
        return
    requests.post(
        f"{API_URL}/feedback", json={"request_id": request_id, "rating": rating}, timeout=10
    )


@st.cache_data(ttl=3600)
def scheme_options() -> dict[str, str]:
    try:
        if LOCAL:
            from yojanamitra.ingestion.registry import load_schemes

            return {s.name: s.id for s in load_schemes()}
        return {s["name"]: s["id"] for s in requests.get(f"{API_URL}/schemes", timeout=60).json()}
    except Exception:  # noqa: BLE001
        return {}


# ------------------------------------------------------------------ UI
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:16]
    st.session_state.messages = []
    st.session_state.voted = set()

with st.sidebar:
    st.header("🇮🇳 YojanaMitra")
    st.caption("Answers come only from official scheme documents, with sources.")
    options = scheme_options()
    picked = st.multiselect("Limit to schemes (optional)", list(options.keys()))
    scheme_filter = [options[p] for p in picked] or None
    st.markdown("**Try asking:**")
    examples = [
        "Who is eligible for PM-KISAN?",
        "What does PMSBY cover and what is the premium?",
        "सुकन्या समृद्धि खाते में न्यूनतम कितनी राशि जमा करनी होती है?",
        "I am 35, can I join Atal Pension Yojana?",
        "How much loan can I get under Mudra Shishu?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.pending = ex
    st.divider()
    st.caption(
        "⚠️ Informational only. Final eligibility is decided by the implementing authority. "
        "Never share Aadhaar or bank details here."
    )
    if not LOCAL:
        st.caption(f"Backend: {API_URL} (first request after idle may take ~1 min)")

st.title("YojanaMitra · योजनामित्र")
st.caption("Ask about Indian Government schemes in English or Hindi")

for i, m in enumerate(st.session_state.messages):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            meta = m.get("meta", {})
            if meta.get("citations"):
                with st.expander(f"Sources ({len(meta['citations'])})"):
                    for c in meta["citations"]:
                        st.markdown(
                            f"**[{c['id']}] {c['scheme_name']} › {c['section']}**  \n"
                            f"{c['snippet']}…  \n[{c['source_url']}]({c['source_url']})"
                        )
            info = []
            if meta.get("cache_hit"):
                info.append("⚡ cached")
            if meta.get("pii_masked"):
                info.append("🔒 personal data masked: " + ", ".join(meta["pii_masked"]))
            if meta.get("latency_ms") is not None:
                info.append(f"{meta['latency_ms'] / 1000:.1f}s")
            if meta.get("prompt_version"):
                info.append(f"prompt v{meta['prompt_version']}")
            if info:
                st.caption(" · ".join(info))
            rid = meta.get("request_id")
            if rid and rid not in st.session_state.voted:
                c1, c2, _ = st.columns([1, 1, 8])
                if c1.button("👍", key=f"up{i}"):
                    send_feedback(rid, 1)
                    st.session_state.voted.add(rid)
                    st.rerun()
                if c2.button("👎", key=f"down{i}"):
                    send_feedback(rid, -1)
                    st.session_state.voted.add(rid)
                    st.rerun()

question = st.chat_input("Type your question…") or st.session_state.pop("pending", None)
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.spinner("Searching official documents…"):
        try:
            data = ask(question, scheme_filter)
        except Exception as exc:  # noqa: BLE001
            data = {
                "answer": f"Sorry, the service is unavailable right now ({exc.__class__.__name__}).",
                "citations": [],
            }
    st.session_state.messages.append({"role": "assistant", "content": data["answer"], "meta": data})
    st.rerun()
