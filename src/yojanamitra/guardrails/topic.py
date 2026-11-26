"""Input guardrails: language detection, greetings, off-topic and prompt-injection checks.

Cheap heuristics run first; the small LLM classifier is only called when the
heuristics cannot decide (saves cost + latency).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

DEVANAGARI = re.compile(r"[ऀ-ॿ]")

GREETINGS = {
    "hi",
    "hello",
    "hey",
    "namaste",
    "namaskar",
    "good morning",
    "good evening",
    "नमस्ते",
    "नमस्कार",
    "हेलो",
    "thanks",
    "thank you",
    "धन्यवाद",
    "shukriya",
}

# words that strongly indicate a scheme question
ON_TOPIC_HINTS = [
    "scheme",
    "yojana",
    "yojna",
    "subsidy",
    "pension",
    "loan",
    "insurance",
    "bima",
    "eligib",
    "apply",
    "application",
    "benefit",
    "installment",
    "instalment",
    "kisan",
    "farmer",
    "awas",
    "house",
    "ration",
    "card",
    "document",
    "certificate",
    "government",
    "sarkari",
    "csc",
    "beneficiary",
    "status",
    "premium",
    "account",
    "scholarship",
    "widow",
    "girl child",
    "योजना",
    "सब्सिडी",
    "पेंशन",
    "लोन",
    "ऋण",
    "बीमा",
    "पात्र",
    "आवेदन",
    "लाभ",
    "किसान",
    "आवास",
    "सरकार",
    "दस्तावेज",
    "किस्त",
    "छात्रवृत्ति",
]

INJECTION_PATTERNS = [
    r"ignore (all |the )?(previous|above|prior) (instructions|rules)",
    r"system prompt",
    r"you are now",
    r"act as (?!a citizen)",
    r"jailbreak",
    r"developer mode",
    r"reveal (your|the) (prompt|instructions)",
]
_INJECTION = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)


@dataclass
class TopicDecision:
    on_topic: bool
    reason: str  # greeting | injection | hint | router | classifier | default


def detect_language(text: str) -> str:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "English"
    ratio = sum(bool(DEVANAGARI.match(c)) for c in letters) / len(letters)
    return "Hindi" if ratio > 0.3 else "English"


def is_greeting(text: str) -> bool:
    t = re.sub(r"[^\w\sऀ-ॿ]", "", text.lower()).strip()
    return t in GREETINGS or (len(t.split()) <= 3 and any(t.startswith(g) for g in GREETINGS))


def is_injection(text: str) -> bool:
    return bool(_INJECTION.search(text))


def heuristic_topic(text: str, routed_schemes: list[str]) -> TopicDecision | None:
    if is_injection(text):
        return TopicDecision(False, "injection")
    if routed_schemes:
        return TopicDecision(True, "router")
    t = text.lower()
    if any(h in t for h in ON_TOPIC_HINTS):
        return TopicDecision(True, "hint")
    return None  # undecided -> ask the classifier


CLASSIFIER_PROMPT = """You are a strict classifier for an Indian Government schemes help desk.
Decide if the user message is about Indian government schemes, welfare benefits, subsidies,
pensions, govt insurance/loans, eligibility, documents or application process.
Reply with JSON only: {"on_topic": true|false}

Message: """


def classify_with_llm(text: str, llm, model: str) -> TopicDecision:
    try:
        res = llm.chat(
            [{"role": "user", "content": CLASSIFIER_PROMPT + text}],
            model=model,
            temperature=0,
            max_tokens=20,
            json_mode=True,
        )
        return TopicDecision(bool(res.json().get("on_topic", True)), "classifier")
    except Exception:  # noqa: BLE001 - fail open: retrieval confidence still protects us
        return TopicDecision(True, "default")


GREETING_REPLY = {
    "English": "Namaste! I am YojanaMitra. Ask me about Indian Government schemes - "
    "benefits, eligibility, documents or how to apply. Example: 'Who is eligible for PM-KISAN?'",
    "Hindi": "नमस्ते! मैं योजनामित्र हूँ। सरकारी योजनाओं के लाभ, पात्रता, दस्तावेज़ या आवेदन प्रक्रिया "
    "के बारे में पूछें। उदाहरण: 'पीएम किसान के लिए कौन पात्र है?'",
}
OFF_TOPIC_REPLY = {
    "English": "I can only help with Indian Government schemes (benefits, eligibility, documents, "
    "how to apply). Please ask a scheme-related question.",
    "Hindi": "मैं केवल सरकारी योजनाओं (लाभ, पात्रता, दस्तावेज़, आवेदन) से जुड़े प्रश्नों में मदद कर सकता हूँ।",
}
NOT_FOUND_REPLY = {
    "English": "I could not find this in the official scheme documents I have. Please check "
    "https://www.myscheme.gov.in or visit your nearest Common Service Centre (CSC).",
    "Hindi": "मुझे यह जानकारी उपलब्ध आधिकारिक योजना दस्तावेज़ों में नहीं मिली। कृपया "
    "https://www.myscheme.gov.in देखें या नज़दीकी कॉमन सर्विस सेंटर (CSC) पर जाएँ।",
}
