"""Section-aware chunker.

Govt scheme documents follow a predictable structure (Benefits, Eligibility, Exclusions,
Documents Required, How to Apply, FAQ). We split on those headings first, then window
long sections by words with overlap. Each chunk carries its section name so the
retriever can cite "PM-KISAN > Eligibility".
"""

from __future__ import annotations

import re
import uuid
from dataclasses import asdict, dataclass

# canonical section -> keywords (English + Hindi).
# ORDER MATTERS: more specific sections first ("not eligible" must hit Exclusions
# before Eligibility; "eligibility details" must not hit Overview).
SECTION_KEYWORDS: dict[str, list[str]] = {
    "Exclusions": ["exclusion", "not eligible", "ineligible", "अपात्र", "बहिष्करण"],
    "Eligibility": ["eligibility", "eligible", "who can apply", "criteria", "पात्रता", "पात्र"],
    "Documents Required": ["documents required", "required documents", "documents", "आवश्यक दस्तावेज", "दस्तावेज"],
    "How to Apply": ["how to apply", "application process", "apply online", "registration", "आवेदन", "पंजीकरण"],
    "Benefits": ["benefit", "assistance", "financial support", "लाभ", "सहायता राशि"],
    "FAQ": ["faq", "frequently asked", "प्रश्न"],
    "Contact": ["helpline", "contact", "grievance", "संपर्क", "हेल्पलाइन"],
    "Overview": ["overview", "introduction", "about the scheme", "objective", "परिचय", "उद्देश्य"],
}

_MD_HEADING = re.compile(r"^#{1,4}\s+(.+)$")
_NUMBERED_HEADING = re.compile(r"^(\d{1,2}(\.\d{1,2})*\.?|[ivxIVX]+\.)\s+([A-Zऀ-ॿ][^.!?]{2,80})$")


@dataclass
class Chunk:
    id: str
    text: str  # text that is embedded (with context header)
    raw_text: str  # text shown to the LLM
    scheme_id: str
    scheme_name: str
    section: str
    source_id: str
    source_url: str
    ministry: str
    chunk_index: int
    doc_hash: str

    def to_dict(self) -> dict:
        return asdict(self)


def canonical_section(heading: str) -> str | None:
    h = heading.lower().strip(" :#-")
    for section, kws in SECTION_KEYWORDS.items():
        if any(kw in h for kw in kws):
            return section
    return None


def detect_heading(line: str) -> str | None:
    """Return heading text if the line looks like a heading, else None."""
    s = line.strip()
    if not s or len(s) > 90:
        return None
    m = _MD_HEADING.match(s)
    if m:
        return m.group(1).strip()
    m = _NUMBERED_HEADING.match(s)
    if m and canonical_section(m.group(3)):
        return m.group(3).strip()
    if s.endswith(":") and len(s.split()) <= 6 and canonical_section(s):
        return s.rstrip(":")
    if s.isupper() and 1 <= len(s.split()) <= 6 and canonical_section(s):
        return s.title()
    return None


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split document text into (section_name, body) pairs."""
    sections: list[tuple[str, list[str]]] = [("Overview", [])]
    for line in text.splitlines():
        heading = detect_heading(line)
        if heading:
            name = canonical_section(heading) or heading[:60]
            sections.append((name, []))
        else:
            sections[-1][1].append(line)
    out = []
    for name, lines in sections:
        body = "\n".join(lines).strip()
        if body:
            out.append((name, body))
    return out


def window_words(text: str, max_words: int, overlap: int) -> list[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text]
    step = max(1, max_words - overlap)
    windows = []
    for start in range(0, len(words), step):
        windows.append(" ".join(words[start : start + max_words]))
        if start + max_words >= len(words):
            break
    return windows


def chunk_document(
    text: str,
    *,
    scheme_id: str,
    scheme_name: str,
    source_id: str,
    source_url: str,
    ministry: str,
    doc_hash: str,
    max_words: int = 220,
    overlap: int = 40,
    min_words: int = 8,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    idx = 0
    for section, body in split_sections(text):
        for piece in window_words(body, max_words, overlap):
            if len(piece.split()) < min_words:
                continue
            header = f"Scheme: {scheme_name} | Section: {section}"
            chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{idx}:{doc_hash}"))
            chunks.append(
                Chunk(
                    id=chunk_id,
                    text=f"{header}\n{piece}",
                    raw_text=piece,
                    scheme_id=scheme_id,
                    scheme_name=scheme_name,
                    section=section,
                    source_id=source_id,
                    source_url=source_url,
                    ministry=ministry,
                    chunk_index=idx,
                    doc_hash=doc_hash,
                )
            )
            idx += 1
    return chunks
