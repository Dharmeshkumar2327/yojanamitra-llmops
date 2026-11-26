"""Text cleaning for govt PDFs/HTML (English + Hindi)."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

_MULTI_SPACE = re.compile(r"[ \t ]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_PAGE_NUM = re.compile(r"^\s*(page\s*)?\d+\s*(of\s*\d+)?\s*$", re.IGNORECASE)


def remove_repeated_lines(pages: list[str], min_ratio: float = 0.5) -> list[str]:
    """Drop header/footer lines that appear on more than `min_ratio` of pages."""
    if len(pages) < 3:
        return pages
    counts: Counter[str] = Counter()
    for page in pages:
        counts.update({ln.strip() for ln in page.splitlines() if ln.strip()})
    threshold = max(2, int(len(pages) * min_ratio))
    repeated = {ln for ln, c in counts.items() if c >= threshold and len(ln) < 120}
    return ["\n".join(ln for ln in p.splitlines() if ln.strip() not in repeated) for p in pages]


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)  # important for Devanagari
    text = text.replace("\r", "\n").replace("​", "")
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    lines = []
    for ln in text.splitlines():
        ln = _MULTI_SPACE.sub(" ", ln).strip()
        if _PAGE_NUM.match(ln):
            continue
        lines.append(ln)
    text = "\n".join(lines)
    return _MULTI_NEWLINE.sub("\n\n", text).strip()


def clean_pages(pages: list[str]) -> str:
    return clean_text("\n\n".join(remove_repeated_lines(pages)))
