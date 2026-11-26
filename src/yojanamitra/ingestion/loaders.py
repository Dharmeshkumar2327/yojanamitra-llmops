"""Document loaders: local files in data/raw first, then remote URLs.

Returns RawDocument objects whose text keeps headings as markdown ('## ...') so the
section-aware chunker can split on them.
"""

from __future__ import annotations

import hashlib
import io
import logging
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup
from pypdf import PdfReader

from yojanamitra.ingestion.registry import Scheme

log = logging.getLogger(__name__)

SUPPORTED_EXT = {".pdf", ".html", ".htm", ".txt", ".md"}
USER_AGENT = "Mozilla/5.0 (YojanaMitra research bot; +https://github.com/Dharmeshkumar2327)"


@dataclass
class RawDocument:
    source_id: str  # e.g. pm_kisan__pm_kisan_guidelines.pdf
    scheme_id: str
    scheme_name: str
    ministry: str
    source_url: str
    pages: list[str]  # one string per page (HTML = 1 page)

    @property
    def text(self) -> str:
        return "\n\n".join(self.pages)

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------- parsers
def parse_pdf_bytes(data: bytes) -> list[str]:
    reader = PdfReader(io.BytesIO(data))
    return [(page.extract_text() or "") for page in reader.pages]


def parse_html(html: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "svg", "form"]):
        tag.decompose()
    lines: list[str] = []
    for el in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "td", "th"]):
        text = el.get_text(" ", strip=True)
        if not text:
            continue
        if el.name in {"h1", "h2", "h3", "h4"}:
            lines.append(f"\n## {text}")
        elif el.name == "li":
            lines.append(f"- {text}")
        else:
            lines.append(text)
    return ["\n".join(lines)]


def parse_file(path: Path) -> list[str]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return parse_pdf_bytes(path.read_bytes())
    if ext in {".html", ".htm"}:
        return parse_html(path.read_text(encoding="utf-8", errors="ignore"))
    return [path.read_text(encoding="utf-8", errors="ignore")]


# ---------------------------------------------------------------- loaders
def local_files_for(scheme: Scheme, raw_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in raw_dir.glob(f"{scheme.id}*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXT
    )


def fetch_url(url: str, timeout: float = 30.0) -> list[str]:
    import httpx  # imported lazily so pure modules/tests don't need it

    resp = httpx.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout, follow_redirects=True)
    resp.raise_for_status()
    ctype = resp.headers.get("content-type", "")
    if "pdf" in ctype or url.lower().endswith(".pdf"):
        return parse_pdf_bytes(resp.content)
    return parse_html(resp.text)


def load_scheme_documents(scheme: Scheme, raw_dir: Path, allow_fetch: bool = True) -> list[RawDocument]:
    docs: list[RawDocument] = []
    for path in local_files_for(scheme, raw_dir):
        try:
            pages = parse_file(path)
            docs.append(
                RawDocument(
                    source_id=f"{scheme.id}__{path.name}",
                    scheme_id=scheme.id,
                    scheme_name=scheme.name,
                    ministry=scheme.ministry,
                    source_url=scheme.urls[0] if scheme.urls else f"file://{path.name}",
                    pages=pages,
                )
            )
            log.info("loaded %s (%d pages)", path.name, len(pages))
        except Exception as exc:  # noqa: BLE001
            log.warning("failed to parse %s: %s", path, exc)

    if docs or not allow_fetch:
        return docs

    for url in scheme.urls:
        try:
            pages = fetch_url(url)
            if sum(len(p) for p in pages) < 300:
                log.warning("%s returned very little text (JS site?) - save it as PDF in data/raw", url)
            docs.append(
                RawDocument(
                    source_id=f"{scheme.id}__{hashlib.md5(url.encode()).hexdigest()[:8]}",
                    scheme_id=scheme.id,
                    scheme_name=scheme.name,
                    ministry=scheme.ministry,
                    source_url=url,
                    pages=pages,
                )
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("fetch failed for %s: %s", url, exc)
    return docs
