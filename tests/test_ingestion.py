from yojanamitra.ingestion.chunker import canonical_section, chunk_document, detect_heading, split_sections, window_words
from yojanamitra.ingestion.cleaner import clean_text, remove_repeated_lines
from yojanamitra.ingestion.loaders import parse_html
from yojanamitra.ingestion.registry import load_schemes

DOC = """## Benefits
Eligible farmer families get Rs 6000 per year in three equal instalments of Rs 2000 each paid directly.
## Eligibility
All landholding farmer families having cultivable land in their names are eligible under the scheme.
EXCLUSIONS
Institutional land holders and income tax payers are not eligible for any benefit under the scheme rules.
"""


def test_canonical_section_order():
    assert canonical_section("Eligibility Details") == "Eligibility"
    assert canonical_section("Who is not eligible") == "Exclusions"
    assert canonical_section("पात्रता") == "Eligibility"
    assert canonical_section("How to Apply") == "How to Apply"


def test_detect_heading():
    assert detect_heading("## Benefits") == "Benefits"
    assert detect_heading("EXCLUSIONS") == "Exclusions"
    assert detect_heading("Documents Required:") == "Documents Required"
    assert detect_heading("This is a normal sentence about the scheme benefits.") is None


def test_split_sections():
    names = [n for n, _ in split_sections(DOC)]
    assert names == ["Benefits", "Eligibility", "Exclusions"]


def test_window_words_overlap():
    text = " ".join(str(i) for i in range(100))
    w = window_words(text, max_words=40, overlap=10)
    assert len(w) == 3
    assert w[1].split()[0] == "30"


def test_chunk_document_metadata_and_ids():
    kw = dict(scheme_id="pm_kisan", scheme_name="PM-KISAN", source_id="pm_kisan__x.pdf",
              source_url="https://pmkisan.gov.in/", ministry="Agri", doc_hash="abc")
    a = chunk_document(DOC, **kw)
    b = chunk_document(DOC, **kw)
    assert [c.id for c in a] == [c.id for c in b]  # deterministic ids
    assert {c.section for c in a} == {"Benefits", "Eligibility", "Exclusions"}
    assert a[0].text.startswith("Scheme: PM-KISAN | Section: Benefits")


def test_clean_text_removes_page_numbers_and_hyphenation():
    out = clean_text("Agri-\nculture   scheme\nPage 3 of 10\n\n\n\nEnd")
    assert "Agriculture scheme" in out
    assert "Page 3" not in out
    assert "\n\n\n" not in out


def test_remove_repeated_lines():
    pages = [f"Ministry Header\ncontent {i}\nwww.site.gov.in" for i in range(5)]
    out = remove_repeated_lines(pages)
    assert all("Ministry Header" not in p for p in out)
    assert "content 3" in out[3]


def test_parse_html_keeps_headings():
    html = "<html><nav>menu</nav><h2>Eligibility</h2><p>Farmers</p><ul><li>Own land</li></ul><script>x</script></html>"
    text = parse_html(html)[0]
    assert "## Eligibility" in text and "- Own land" in text
    assert "menu" not in text and "x" not in text.split()


def test_registry_loads():
    schemes = load_schemes()
    ids = {s.id for s in schemes}
    assert {"pm_kisan", "pm_jay", "sukanya_samriddhi"} <= ids
    assert len(ids) == len(schemes)  # no duplicate ids
