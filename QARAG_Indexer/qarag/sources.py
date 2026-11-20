"""
sources.py - read an article from anywhere and return it as plain text.

Design: the rest of the pipeline never needs to know WHERE a document came
from. It asks `load_document(location)` for a RawDocument and gets plain text
back, whether `location` was:

    https://en.wikipedia.org/wiki/Scott_Derrickson      (web page)
    https://arxiv.org/pdf/2210.03629                     (online PDF)
    F:\\articles\\report.pdf                              (local PDF)
    F:\\articles\\notes.docx | .txt | .md | .html          (other local files)

How it is organised:
  * `Source` is an abstract base class: anything that can `load()` a RawDocument.
  * `UrlSource` downloads bytes over HTTP; `FileSource` reads bytes from disk.
  * Both hand the bytes to one shared set of FORMAT EXTRACTORS (HTML, PDF,
    DOCX, plain text), chosen by file extension or the web server's
    Content-Type. So adding a new format means adding one extractor function
    and one line in the EXTRACTORS table - nothing else changes.
  * `expand_locations()` turns user input (several paths, folders, or a .txt
    file listing sources) into a flat list of locations to index.
"""
from __future__ import annotations

import io
import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import unquote, urlparse

import httpx

from qarag.models import RawDocument

# File types we know how to read. Used both for local files and for
# deciding what an online link points to.
SUPPORTED_EXTENSIONS = {".txt", ".md", ".html", ".htm", ".pdf", ".docx"}


# ===========================================================================
# 1. FORMAT EXTRACTORS: bytes in -> (title, segments) out
#    A "segment" is (page_number_or_None, text).
#
#    SECTION HEADINGS: every extractor writes headings it can recognise as
#    Markdown-style lines ("# Heading" = level 1, "## Sub-heading" = level 2,
#    ...). The chunker turns these lines into a section path such as
#    "Career > Film" attached to every following sentence, so each Q&A pair
#    can cite the sub-sections its answer comes from.
#      HTML : <h1>..<h4> tags          DOCX : "Heading 1..4" / "Title" styles
#      MD   : native "#" lines         TXT  : short standalone title-like lines
#      PDF  : no reliable headings -> citations use page numbers instead
#
#    TABLES (HTML incl. Wikipedia infoboxes, DOCX, Markdown) become one line
#    per row, "<label> - <field>: <value>", see linearize_table(). PDF tables
#    arrive as plain text from the PDF's text layer.
# ===========================================================================
def _clean_text(text: str) -> str:
    """Normalise whitespace but keep paragraph breaks (blank lines), because
    the chunker uses line breaks as natural sentence boundaries."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)          # runs of spaces/tabs -> one space
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)  # 3+ newlines -> one blank line
    return text.strip()


# ---------------------------------------------------------------------------
# TABLES: turned into one readable line per row, so their values become
# sentences the generator can question and cite. Without this, facts that live
# only in tables (e.g. a Wikipedia infobox's birth date and nationality, or a
# results table) would never reach the Q&A pairs.
#
#   key-value rows (infobox, 2-column table) -> "<label> - Born: July 16, 1966."
#   grid with a header row                   -> "<label> - Model: X; Accuracy: 92%."
#   a single-cell row inside a table         -> starts a sub-group of the label
# ---------------------------------------------------------------------------
MAX_TABLE_ROWS = 200        # very long tables are usually data dumps, not prose facts
MAX_ROW_CHARS = 600


def _cell(text: str) -> str:
    return " ".join((text or "").split()).strip(" |")


def linearize_table(rows: list[list[tuple[str, bool]]], label: str) -> list[str]:
    """rows: each row is a list of (cell_text, is_header_cell).
    Returns one sentence per data row, each starting with `label` so it is
    understandable on its own (the generator sees rows out of their grid)."""
    # Clean cell text; drop rows with no text at all. Empty cells are KEPT for
    # now so values stay aligned with their column headers.
    rows = [[(_cell(t), h) for t, h in row] for row in rows]
    rows = [row for row in rows if any(t for t, _ in row)][:MAX_TABLE_ROWS]
    if not rows:
        return []

    # Header row: the first row if all its cells are header cells (HTML <th>),
    # or if the caller marked them so (DOCX/Markdown first rows).
    header: list[str] | None = None
    if len(rows) > 1 and len(rows[0]) > 1 and all(h for _, h in rows[0]):
        header = [t for t, _ in rows[0]]
        rows = rows[1:]

    def non_empty(row):
        return [c for c in row if c[0]]

    lines: list[str] = []
    group = ""                                  # sub-heading inside a table (e.g. infobox section)
    for full_row in rows:
        if header and len(header) == len(full_row):
            # Grid row: pair each value with its column header, skipping empty cells.
            prefix = f"{label} ({group})" if group else label
            line = f"{prefix} - " + "; ".join(f"{h}: {t}" if h else t
                                               for h, (t, _) in zip(header, full_row) if t)
            line = line[:MAX_ROW_CHARS].rstrip(" ;,")
            lines.append(line if line.endswith((".", "!", "?")) else line + ".")
            continue
        row = non_empty(full_row)
        texts = [t for t, _ in row]
        prefix = f"{label} ({group})" if group else label
        if len(row) == 1:
            # A lone header cell is a sub-heading row (e.g. "Personal information"
            # inside an infobox); a lone data cell is kept as a value.
            if row[0][1]:
                group = texts[0] if texts[0].lower() != label.lower() else ""
                continue
            line = f"{prefix} - {texts[0]}"
        elif len(texts) == 2:
            line = f"{prefix} - {texts[0]}: {texts[1]}"
        else:
            line = f"{prefix} - " + "; ".join(texts)
        line = line[:MAX_ROW_CHARS].rstrip(" ;,")
        lines.append(line if line.endswith((".", "!", "?")) else line + ".")
    return lines


def _markdown_tables_to_lines(text: str, title: str) -> str:
    """Replace Markdown pipe tables ("| a | b |" + "|---|---|") with linearized rows.
    The nearest heading above the table is used as its label."""
    out, block, label = [], [], title
    def flush():
        if not block:
            return
        rows, header_done = [], False
        for line in block:
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                header_done = True               # the |---|---| separator line
                if rows:
                    rows[0] = [(t, True) for t, _ in rows[0]]
                continue
            rows.append([(c, False) for c in cells])
        out.extend(linearize_table(rows, label) if header_done or len(rows) > 1 else block)
        block.clear()
    for line in text.split("\n"):
        if line.strip().startswith("|") and line.strip().endswith("|"):
            block.append(line)
            continue
        flush()
        heading = re.match(r"^#{1,6}\s+(.+)$", line.strip())
        if heading:
            label = heading.group(1).strip()
        out.append(line)
    flush()
    return "\n".join(out)


def _decode(data: bytes) -> str:
    """Decode text bytes as UTF-8, falling back to Windows-1252 (common for
    older Windows text files)."""
    try:
        return data.decode("utf-8-sig")          # -sig also strips a BOM if present
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace")


def extract_markdown(data: bytes, fallback_title: str) -> tuple[str, list]:
    """.md files: headings are already written as "# ..." lines."""
    text = _clean_text(_decode(data))
    # Front matter ("---" / "author: ..." / "---") is metadata, not content: remove it.
    text = re.sub(r"^---\s*\n.*?\n---\s*\n", "", text, count=1, flags=re.DOTALL)
    match = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
    title = match.group(1).strip() if match else fallback_title
    text = _markdown_tables_to_lines(text, title)    # pipe tables -> one line per row
    return title, [(None, text)]


def _looks_like_heading(line: str) -> bool:
    """A plain-text line that is probably a heading: short, does not end like
    a sentence, and is not a list item. Used only for .txt files."""
    line = line.strip()
    return (0 < len(line) <= 80 and not re.search(r"[.!?:;,]$", line)
            and not re.match(r"^[-*•\d]", line) and len(line.split()) <= 12)


def extract_text(data: bytes, fallback_title: str) -> tuple[str, list]:
    """.txt files: a short title-like line standing alone between blank lines
    is treated as a heading (so plain-text reports also get sections)."""
    text = _clean_text(_decode(data))
    paragraphs = text.split("\n\n")
    title = fallback_title
    out = []
    for i, para in enumerate(paragraphs):
        if "\n" not in para and _looks_like_heading(para):
            level = 1 if i == 0 else 2             # first line = document title
            if i == 0:
                title = para.strip()
            out.append("#" * level + " " + para.strip())
        else:
            out.append(para)
    return title, [(None, "\n\n".join(out))]


def extract_html(data: bytes, fallback_title: str) -> tuple[str, list]:
    """Web pages: keep the readable article text, drop menus, scripts, footers."""
    from bs4 import BeautifulSoup  # imported here so the package loads even if bs4 is missing

    soup = BeautifulSoup(data, "html.parser")

    # Title: prefer the first <h1> (usually the article headline), else <title>.
    title = fallback_title
    if soup.find("h1") and soup.find("h1").get_text(strip=True):
        title = soup.find("h1").get_text(" ", strip=True)
    elif soup.title and soup.title.get_text(strip=True):
        title = soup.title.get_text(" ", strip=True)

    # Remove elements that are never part of the article body. Tables and
    # figures are KEPT: tables are linearized below, figure captions are text.
    for tag in soup(["script", "style", "noscript", "nav", "header", "footer",
                     "aside", "form", "button", "svg", "img", "video", "audio"]):
        tag.decompose()
    # Wikipedia-specific clutter: reference markers [1], edit links, navigation
    # boxes, maintenance notices, the references list. Infoboxes are kept.
    for css in ["sup.reference", "span.mw-editsection", "div.navbox", "table.navbox",
                "div.reflist", "ol.references", "div.hatnote", "div.toc", "div#toc",
                "table.ambox", "table.metadata", "div.sidebar", "table.sidebar",
                "span.mw-cite-backlink", "style"]:
        for tag in soup.select(css):
            tag.decompose()

    # Prefer the main content container if the page marks one.
    body = (soup.find("article") or soup.find("main")
            or soup.find("div", id="mw-content-text") or soup.body or soup)

    # Collect text block by block, in document order, so paragraphs and
    # headings stay on separate lines; headings become "#"-lines (section names),
    # tables become one line per row, figure captions are kept as text.
    blocks = []
    current_heading = title
    for el in body.find_all(["h1", "h2", "h3", "h4", "p", "li", "blockquote", "pre",
                             "table", "figcaption"]):
        # Text inside a table is handled by the table itself; nested tables by the outer one.
        if el.find_parent("table") is not None:
            continue
        if el.name == "table":
            caption = el.find("caption")
            label = caption.get_text(" ", strip=True) if caption else current_heading
            rows = []
            for tr in el.find_all("tr"):
                if tr.find_parent("table") is not el:
                    continue                      # row of a nested table
                rows.append([(c.get_text(" ", strip=True), c.name == "th")
                             for c in tr.find_all(["th", "td"], recursive=False)])
            blocks.extend(linearize_table(rows, label or title))
            continue
        txt = el.get_text(" ", strip=True)
        if not txt:
            continue
        if el.name in ("h1", "h2", "h3", "h4"):
            level = int(el.name[1])
            blocks.append("#" * level + " " + txt)
            current_heading = txt
        elif el.name == "figcaption":
            blocks.append(f"Figure caption: {txt}" + ("" if txt.endswith(".") else "."))
        else:
            blocks.append(txt)
    text = "\n\n".join(blocks) if blocks else body.get_text("\n", strip=True)
    return title, [(None, _clean_text(text))]


def extract_pdf(data: bytes, fallback_title: str) -> tuple[str, list]:
    """PDFs: one segment per page, so Q&A pairs can cite page numbers.
    Note: scanned PDFs (images of text) have no text layer and will come back
    empty - they would need OCR, which is out of scope for Stage 0."""
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    title = fallback_title
    try:
        if reader.metadata and reader.metadata.title and reader.metadata.title.strip():
            title = reader.metadata.title.strip()
    except Exception:
        pass  # broken metadata is common in PDFs; just keep the fallback title
    segments = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = _clean_text(page.extract_text() or "")
        if text:
            segments.append((page_number, text))
    return title, segments


def extract_docx(data: bytes, fallback_title: str) -> tuple[str, list]:
    """Word documents: paragraphs in order. Word files have no reliable page
    numbers (pages depend on the printer layout), so page is None."""
    import docx  # the "python-docx" package

    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = docx.Document(io.BytesIO(data))
    title = (document.core_properties.title or "").strip() or fallback_title
    paragraphs = []
    current_heading = title
    # Walk the body in document order so tables appear where they are in the text.
    for child in document.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "tbl":
            table = Table(child, document)
            rows = []
            for r, row in enumerate(table.rows):
                cells, last = [], None
                for cell in row.cells:
                    text = cell.text.strip()
                    if text and text != last:     # merged cells repeat their text; keep once
                        cells.append(text)
                    last = text
                # Word has no header-cell marker; treat the first row of a
                # multi-column table as its header (the usual convention).
                rows.append([(t, r == 0 and len(table.rows) > 1 and len(cells) > 2) for t in cells])
            paragraphs.extend(linearize_table(rows, current_heading))
            continue
        if tag != "p":
            continue
        p = Paragraph(child, document)
        text = p.text.strip()
        if not text:
            continue
        style = (p.style.name if p.style is not None else "") or ""
        # Word's built-in heading styles become section headings.
        match = re.match(r"Heading (\d)", style)
        if style == "Title":
            paragraphs.append("# " + text)
            current_heading = text
        elif match:
            paragraphs.append("#" * min(int(match.group(1)) + 1, 6) + " " + text)
            current_heading = text
        else:
            paragraphs.append(text)
    return title, [(None, _clean_text("\n\n".join(paragraphs)))]


# ===========================================================================
# 1b. DOCUMENT METADATA (P4a): bytes in -> dict of descriptive fields out
#
#    Stored per document in SQLite (documents table) and partly copied into
#    the Chroma metadata (doc_type, language, year) so searches can be
#    filtered. Every field is OPTIONAL: metadata is often missing or wrong
#    (especially in PDFs), so nothing in the pipeline depends on it.
#
#    field        HTML / Wikipedia                     PDF               DOCX
#    -----------  -----------------------------------  ----------------  ----------------
#    authors      citation_author, author, JSON-LD     /Author           core author
#    published    citation_publication_date, JSON-LD   /CreationDate     core created
#                 datePublished, article:published_time
#    modified     JSON-LD dateModified, Last-Modified  /ModDate          core modified
#    language     <html lang>, og:locale               (text heuristic)  core language
#    doi          citation_doi, dc.identifier, text    text of page 1-2  text
#    publisher    citation_publisher, og:site_name     -                 -
#    description  meta description / og:description   /Subject          core subject
#    keywords     meta keywords                        /Keywords         core keywords
#    doc_type     "wikipedia article" | "scholarly article" | "web page" | "pdf" | ...
# ===========================================================================
_DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s\"<>]+)", re.IGNORECASE)
_ARABIC_RE = re.compile(r"[؀-ۿ]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_EN_WORDS = {"the", "and", "of", "to", "in", "is", "was", "for", "that", "with", "as", "by", "on"}


def _guess_language(text: str) -> Optional[str]:
    """Cheap fallback when the file does not declare its language: Arabic
    script -> 'ar'; Latin text with many common English words -> 'en';
    otherwise unknown (None) rather than a wrong guess."""
    sample = text[:20000]
    arabic, latin = len(_ARABIC_RE.findall(sample)), len(_LATIN_RE.findall(sample))
    if arabic > latin:
        return "ar"
    words = re.findall(r"[a-z]+", sample.lower())
    if words and sum(w in _EN_WORDS for w in words) / len(words) > 0.08:
        return "en"
    return None


def _norm_date(value) -> Optional[str]:
    """Dates come in many shapes ("2016-10-13T12:00:00Z", "D:20210304...",
    "2021/03/04", datetime objects). Keep the ISO date part YYYY[-MM[-DD]]."""
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()[:10]
    text = str(value).strip()
    if text.startswith("D:"):                                  # PDF date format D:YYYYMMDD...
        digits = re.sub(r"\D", "", text[2:])
        parts = [digits[0:4], digits[4:6], digits[6:8]]
        return "-".join(p for p in parts if p) or None
    match = re.search(r"(\d{4})(?:[-/.](\d{1,2}))?(?:[-/.](\d{1,2}))?", text)
    if not match:
        return None
    y, m, d = match.groups()
    return y + (f"-{int(m):02d}" if m else "") + (f"-{int(d):02d}" if m and d else "")


def _split_authors(value) -> list[str]:
    """'A; B', 'A and B' or a list -> clean list of names (max 50)."""
    items = value if isinstance(value, list) else re.split(r";|\band\b|\n", str(value or ""))
    names = [" ".join(str(x).split()).strip(" ,") for x in items]
    return list(dict.fromkeys(n for n in names if n and len(n) < 200))[:50]


def _find_doi(text: str) -> Optional[str]:
    match = _DOI_RE.search(text[:30000])
    return match.group(1).rstrip(".,;)]") if match else None


def metadata_html(data: bytes, url: Optional[str]) -> dict:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(data, "html.parser")
    meta: dict = {}

    def tags(*names: str) -> list[str]:
        """Content of every <meta name=...> or <meta property=...> with one of these names."""
        wanted = {n.lower() for n in names}
        out = []
        for tag in soup.find_all("meta"):
            key = (tag.get("name") or tag.get("property") or "").lower()
            if key in wanted and tag.get("content"):
                out.append(tag["content"].strip())
        return out

    # JSON-LD blocks (schema.org): Wikipedia and many news/journal sites use them.
    ld: dict = {}
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data_ld = json.loads(script.string or "")
        except (ValueError, TypeError):
            continue
        for item in (data_ld if isinstance(data_ld, list) else [data_ld]):
            if isinstance(item, dict):
                for key, value in item.items():
                    ld.setdefault(key, value)

    def ld_names(value) -> list[str]:
        items = value if isinstance(value, list) else [value]
        return [i.get("name") if isinstance(i, dict) else str(i) for i in items if i]

    authors = tags("citation_author", "dc.creator", "author", "article:author")
    if not authors and ld.get("author"):
        authors = ld_names(ld["author"])
    meta["authors"] = _split_authors(authors) if authors else []
    meta["published"] = _norm_date((tags("citation_publication_date", "citation_date", "dc.date",
                                         "article:published_time", "date") or [ld.get("datePublished")])[0])
    meta["modified"] = _norm_date((tags("article:modified_time", "last-modified") or [ld.get("dateModified")])[0])
    html_tag = soup.find("html")
    lang = (html_tag.get("lang") if html_tag else None) or (tags("og:locale") or [None])[0]
    meta["language"] = lang.split("-")[0].split("_")[0].lower() if lang else None
    doi = (tags("citation_doi", "dc.identifier", "prism.doi") or [None])[0]
    meta["doi"] = _find_doi(doi) if doi else None
    publisher = tags("citation_publisher", "citation_journal_title", "og:site_name")
    if not publisher and isinstance(ld.get("publisher"), dict):
        publisher = [ld["publisher"].get("name")]
    meta["publisher"] = publisher[0] if publisher and publisher[0] else None
    meta["description"] = ((tags("description", "og:description") or [None])[0] or None)
    meta["keywords"] = _split_authors((tags("keywords", "citation_keywords") or [""])[0].replace(",", ";"))

    host = (urlparse(url).netloc.lower() if url else "")
    if "wikipedia.org" in host:
        meta["doc_type"] = "wikipedia article"
        meta["publisher"] = "Wikipedia"
        # Wikipedia is cited by its contributors and the REVISION date (the
        # version that was indexed), not the date the article was first created.
        meta["authors"] = ["Wikipedia contributors"]
        meta["created"] = _norm_date(ld.get("datePublished"))
        meta["published"] = meta["modified"] or meta["published"]
    elif tags("citation_title", "citation_doi", "citation_journal_title"):
        meta["doc_type"] = "scholarly article"
    else:
        meta["doc_type"] = "web page"
    if not meta["doi"]:
        meta["doi"] = _find_doi(soup.get_text(" ", strip=True)) if meta["doc_type"] == "scholarly article" else None
    return meta


def metadata_pdf(data: bytes) -> dict:
    from pypdf import PdfReader

    meta: dict = {"doc_type": "pdf"}
    try:
        reader = PdfReader(io.BytesIO(data))
        info = reader.metadata or {}
        meta["authors"] = _split_authors(info.get("/Author")) if info.get("/Author") else []
        meta["published"] = _norm_date(info.get("/CreationDate"))
        meta["modified"] = _norm_date(info.get("/ModDate"))
        meta["description"] = str(info.get("/Subject")).strip() if info.get("/Subject") else None
        meta["keywords"] = _split_authors(str(info.get("/Keywords") or "").replace(",", ";"))
        meta["pages"] = len(reader.pages)
        first_pages = " ".join((p.extract_text() or "") for p in reader.pages[:2])
        meta["doi"] = _find_doi(first_pages)
        meta["language"] = _guess_language(first_pages)
    except Exception:  # noqa: BLE001 - broken PDF metadata must never stop indexing
        pass
    return meta


def metadata_docx(data: bytes) -> dict:
    import docx

    meta: dict = {"doc_type": "word document"}
    try:
        document = docx.Document(io.BytesIO(data))
        core = document.core_properties
        meta["authors"] = _split_authors(core.author) if core.author else []
        meta["published"] = _norm_date(core.created)
        meta["modified"] = _norm_date(core.modified)
        meta["language"] = (core.language or "").split("-")[0].lower() or None
        meta["description"] = core.subject or None
        meta["keywords"] = _split_authors((core.keywords or "").replace(",", ";"))
        text = "\n".join(p.text for p in document.paragraphs[:200])
        meta["doi"] = _find_doi(text)
        meta["language"] = meta["language"] or _guess_language(text)
    except Exception:  # noqa: BLE001
        pass
    return meta


def metadata_text(data: bytes, fmt: str) -> dict:
    """.md / .txt: optional YAML-style front matter at the top of Markdown
    ("---" / "author: ..." / "date: ..." / "---"), plus text heuristics."""
    text = _decode(data)
    meta: dict = {"doc_type": "markdown" if fmt == ".md" else "text"}
    front = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, flags=re.DOTALL) if fmt == ".md" else None
    if front:
        fields = dict(re.findall(r"^(\w[\w-]*)\s*:\s*(.+)$", front.group(1), flags=re.MULTILINE))
        fields = {k.lower(): v.strip().strip("\"'") for k, v in fields.items()}
        meta["authors"] = _split_authors(fields.get("author") or fields.get("authors") or "")
        meta["published"] = _norm_date(fields.get("date") or fields.get("published"))
        meta["language"] = fields.get("lang") or fields.get("language")
        meta["description"] = fields.get("description")
    meta["doi"] = _find_doi(text)
    meta["language"] = meta.get("language") or _guess_language(text)
    return meta


def extract_metadata(fmt: str, data: bytes, url: Optional[str] = None) -> dict:
    """Metadata for one document in any supported format (never raises)."""
    try:
        if fmt in (".html", ".htm"):
            meta = metadata_html(data, url)
        elif fmt == ".pdf":
            meta = metadata_pdf(data)
        elif fmt == ".docx":
            meta = metadata_docx(data)
        else:
            meta = metadata_text(data, fmt)
    except Exception:  # noqa: BLE001
        meta = {}
    meta.setdefault("authors", [])
    for key in ("published", "modified", "language", "doi", "publisher", "description", "doc_type"):
        meta.setdefault(key, None)
    if meta.get("description"):
        meta["description"] = " ".join(str(meta["description"]).split())[:500]
    meta["size_bytes"] = len(data)
    return meta


# Extension -> extractor function. Add a new format by adding a line here.
EXTRACTORS: dict[str, Callable[[bytes, str], tuple[str, list]]] = {
    ".txt": extract_text,
    ".md": extract_markdown,
    ".html": extract_html,
    ".htm": extract_html,
    ".pdf": extract_pdf,
    ".docx": extract_docx,
}


# ===========================================================================
# 2. SOURCES: where the bytes come from
# ===========================================================================
class Source(ABC):
    """Anything that can produce a RawDocument. Subclasses only differ in how
    they obtain the raw bytes and decide the format."""

    def __init__(self, location: str):
        self.location = location

    @abstractmethod
    def load(self) -> RawDocument:
        ...


class FileSource(Source):
    """A document on the local disk (any of SUPPORTED_EXTENSIONS)."""

    def load(self) -> RawDocument:
        path = Path(self.location).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        ext = path.suffix.lower()
        if ext not in EXTRACTORS:
            raise ValueError(f"Unsupported file type '{ext}' for {path.name}. "
                             f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        data = path.read_bytes()
        title, segments = EXTRACTORS[ext](data, path.stem)
        metadata = extract_metadata(ext, data)
        metadata["file_name"] = path.name
        if not metadata.get("modified"):
            # No date inside the file: fall back to the file's last-modified time on disk.
            from datetime import datetime
            metadata["modified"] = datetime.fromtimestamp(path.stat().st_mtime).date().isoformat()
        # The absolute path is the document's identity in the database, so the
        # same file is recognised again later even if given with a relative path.
        return RawDocument(source=str(path), kind="file", title=title, segments=segments,
                           metadata=metadata)


class UrlSource(Source):
    """A document on the web: an HTML page or a directly linked file (PDF, DOCX, TXT)."""

    def __init__(self, location: str, user_agent: str, timeout: float = 60.0):
        super().__init__(location)
        self.user_agent = user_agent
        self.timeout = timeout

    def _detect_format(self, url: str, content_type: str) -> str:
        """Decide which extractor to use: trust the server's Content-Type first,
        then the URL's extension; default to HTML (most links are web pages)."""
        ct = content_type.lower()
        if "pdf" in ct:
            return ".pdf"
        if "officedocument.wordprocessingml" in ct:
            return ".docx"
        if "text/plain" in ct:
            return ".txt"
        if "markdown" in ct:
            return ".md"
        if "html" in ct:
            return ".html"
        ext = Path(urlparse(url).path).suffix.lower()
        return ext if ext in EXTRACTORS else ".html"

    def load(self) -> RawDocument:
        headers = {
            "User-Agent": self.user_agent,
            # Ordinary browser-style headers; some sites refuse requests without them.
            "Accept": "text/html,application/xhtml+xml,application/pdf;q=0.9,*/*;q=0.8",
            "Accept-Language": "en;q=0.9",
        }
        # follow_redirects: many links (e.g. arxiv.org/pdf/...) redirect before serving the file.
        response = httpx.get(self.location, headers=headers, timeout=self.timeout,
                             follow_redirects=True)
        if response.status_code == 403:
            # 403 = the site refused us. Explain the two usual fixes instead of a bare error.
            raise PermissionError(
                f"403 Forbidden from {urlparse(self.location).netloc}. The site rejected the request. "
                "For Wikipedia, set QARAG_USER_AGENT in .env with your contact, e.g. "
                "QARAG_USER_AGENT=qarag-research/0.1 (PhD research; mailto:you@example.com). "
                "For any site, you can also save the page in your browser (Ctrl+S, 'Webpage, HTML only') "
                "and index the saved .html file instead.")
        response.raise_for_status()   # turns 404/500 etc. into a clear Python error
        fmt = self._detect_format(str(response.url), response.headers.get("content-type", ""))
        # Fallback title from the last part of the URL, e.g. ".../Scott_Derrickson" -> "Scott Derrickson".
        last_part = unquote(Path(urlparse(str(response.url)).path).stem) or urlparse(self.location).netloc
        fallback = last_part.replace("_", " ")
        title, segments = EXTRACTORS[fmt](response.content, fallback)
        metadata = extract_metadata(fmt, response.content, str(response.url))
        metadata["file_name"] = unquote(Path(urlparse(str(response.url)).path).name) or None
        metadata["final_url"] = str(response.url)          # after redirects
        if not metadata.get("modified") and response.headers.get("last-modified"):
            from email.utils import parsedate_to_datetime
            try:
                metadata["modified"] = parsedate_to_datetime(response.headers["last-modified"]).date().isoformat()
            except (TypeError, ValueError):
                pass
        return RawDocument(source=self.location, kind="url", title=title, segments=segments,
                           metadata=metadata)


def is_url(location: str) -> bool:
    return location.lower().startswith(("http://", "https://"))


def load_document(location: str, user_agent: str) -> RawDocument:
    """The single function the rest of the code calls: give it a URL or a path."""
    source: Source = UrlSource(location, user_agent) if is_url(location) else FileSource(location)
    document = source.load()
    if not document.full_text.strip():
        raise ValueError(f"No text could be extracted from {location} "
                         "(empty page, scanned PDF without text layer, or blocked download)")
    return document


# ===========================================================================
# 3. INPUT EXPANSION: turn what the user typed into a list of locations
# ===========================================================================
def expand_locations(items: list[str], from_list: Optional[str] = None) -> list[str]:
    """
    Accepts any mix of:
      * URLs                     -> kept as they are
      * file paths               -> kept (made absolute)
      * folder paths             -> every supported file inside, recursively
      * from_list (a .txt file)  -> one URL or path per line; lines starting with # are ignored
    Duplicates are removed while keeping the original order.
    """
    raw: list[str] = list(items)
    if from_list:
        for line in Path(from_list).read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                raw.append(line)

    result: list[str] = []
    for item in raw:
        if is_url(item):
            result.append(item)
            continue
        path = Path(item).expanduser()
        if path.is_dir():
            # Sorted so the indexing order is the same every run.
            for file in sorted(path.rglob("*")):
                if file.is_file() and file.suffix.lower() in SUPPORTED_EXTENSIONS:
                    result.append(str(file.resolve()))
        else:
            result.append(str(path.resolve()))

    seen: set[str] = set()
    unique = []
    for loc in result:
        if loc not in seen:
            seen.add(loc)
            unique.append(loc)
    return unique
