"""Scheme router: maps a question to scheme ids using aliases (English/Hindi/abbr).

If the user names a scheme, retrieval is filtered to it -> fewer wrong-scheme answers.
"""

from __future__ import annotations

import re

from yojanamitra.ingestion.registry import Scheme


class SchemeRouter:
    def __init__(self, schemes: list[Scheme]):
        self.schemes = {s.id: s for s in schemes}
        self._patterns: list[tuple[re.Pattern, str]] = []
        for s in schemes:
            names = set(s.aliases) | {s.id.replace("_", " ")}
            for alias in sorted(names, key=len, reverse=True):
                # word boundaries for latin text; plain match for Devanagari
                if re.search(r"[a-z0-9]", alias):
                    pat = re.compile(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", re.IGNORECASE)
                else:
                    pat = re.compile(re.escape(alias))
                self._patterns.append((pat, s.id))

    def route(self, question: str) -> list[str]:
        q = question.lower()
        found: list[str] = []
        for pat, sid in self._patterns:
            if sid not in found and pat.search(q):
                found.append(sid)
        return found

    def name(self, scheme_id: str) -> str:
        s = self.schemes.get(scheme_id)
        return s.name if s else scheme_id
