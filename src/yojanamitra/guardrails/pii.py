"""PII masking for Indian identifiers - applied BEFORE text reaches the LLM, logs or traces.

Aadhaar numbers are validated with the Verhoeff checksum (UIDAI uses it), so random
12-digit numbers like amounts are not masked by mistake.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Verhoeff tables
_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]


def verhoeff_valid(number: str) -> bool:
    c = 0
    for i, ch in enumerate(reversed(number)):
        c = _D[c][_P[i % 8][int(ch)]]
    return c == 0


AADHAAR_RE = re.compile(r"(?<!\d)([2-9]\d{3})[\s-]?(\d{4})[\s-]?(\d{4})(?!\d)")
PAN_RE = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?91[\s-]?|0)?[6-9]\d{9}(?!\d)")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
IFSC_RE = re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b")
BANK_AC_RE = re.compile(r"(?i)(account\s*(?:no\.?|number)?\s*[:\-]?\s*)(\d{9,18})")


@dataclass
class PIIResult:
    text: str
    found: list[str] = field(default_factory=list)

    @property
    def has_pii(self) -> bool:
        return bool(self.found)


def mask_pii(text: str) -> PIIResult:
    found: list[str] = []

    def _aadhaar(m: re.Match) -> str:
        digits = "".join(m.groups())
        if verhoeff_valid(digits):
            found.append("AADHAAR")
            return "[AADHAAR]"
        return m.group(0)

    out = AADHAAR_RE.sub(_aadhaar, text)
    for name, pattern in (("PAN", PAN_RE), ("IFSC", IFSC_RE), ("EMAIL", EMAIL_RE)):
        if pattern.search(out):
            found.append(name)
            out = pattern.sub(f"[{name}]", out)
    if BANK_AC_RE.search(out):
        found.append("BANK_ACCOUNT")
        out = BANK_AC_RE.sub(lambda m: m.group(1) + "[BANK_ACCOUNT]", out)
    if PHONE_RE.search(out):
        found.append("PHONE")
        out = PHONE_RE.sub("[PHONE]", out)
    return PIIResult(out, found)
