"""
models.py - the data structures passed between modules.

Using small, explicit classes (instead of loose dicts) documents exactly what
each stage produces and consumes, and lets pydantic reject malformed model
output automatically.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# --------------------------------------------------------------------------
# Output of sources.py: an article as plain text, before any processing.
# --------------------------------------------------------------------------
@dataclass
class RawDocument:
    source: str                     # the URL or the absolute local path, exactly as indexed
    kind: str                       # "url" or "file"
    title: str                      # best-effort title (HTML <title>, PDF metadata, or file name)
    # The text, split into segments. For PDFs each segment is one page, so we
    # can later cite page numbers; other formats have a single segment with
    # page = None.
    segments: list[tuple[Optional[int], str]] = field(default_factory=list)
    # Descriptive metadata (P4a): authors, published, modified, language,
    # doc_type, doi, publisher, description, keywords, file_name, size_bytes ...
    # Every key is optional - see sources.extract_metadata().
    metadata: dict = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return "\n\n".join(text for _, text in self.segments)


# --------------------------------------------------------------------------
# Output of chunker.py
# --------------------------------------------------------------------------
@dataclass
class Sentence:
    idx: int                        # position in the whole document, starting at 0
    text: str
    page: Optional[int] = None      # page number (PDFs only), used for citations
    section: Optional[str] = None   # heading path, e.g. "Career > Film" (None if no headings)


@dataclass
class Chunk:
    chunk_index: int                # 0, 1, 2 ... within the document
    # Sentences shown BEFORE the core (end of the previous chunk), only as
    # background. The model must NOT write Q&A pairs about these alone.
    context: list[Sentence]
    # Sentences the model must cover with Q&A pairs.
    core: list[Sentence]
    # Sentences shown AFTER the core (start of the next chunk): read-only, but
    # the model may use and cite them to complete an answer about a core
    # sentence (documents often explain a point in the following paragraph).
    lookahead: list[Sentence] = field(default_factory=list)

    @property
    def core_ids(self) -> set[int]:
        return {s.idx for s in self.core}


# --------------------------------------------------------------------------
# What the generator model must return for each Q&A pair (validated by pydantic).
# --------------------------------------------------------------------------
class GeneratedPair(BaseModel):
    question: str = Field(min_length=5)
    answer: str = Field(min_length=1)
    # Ids of the sentences ([S12] -> 12) that support the answer.
    sentence_ids: list[int] = Field(min_length=1)

    @field_validator("question", "answer")
    @classmethod
    def _strip(cls, value: str) -> str:
        # Collapse internal whitespace/newlines so stored text is clean.
        return " ".join(value.split())


class GeneratedPairs(BaseModel):
    """The whole JSON object the model returns: {"pairs": [...]}."""
    pairs: list[GeneratedPair] = Field(default_factory=list)


# --------------------------------------------------------------------------
# What the consolidation (merge) model returns for each group of candidates.
# --------------------------------------------------------------------------
class MergedGroup(BaseModel):
    members: list[int] = Field(min_length=1)     # candidate numbers (1-based) combined in this group
    question: str = Field(min_length=5)
    answer: str = Field(min_length=1)
    sentence_ids: list[int] = Field(default_factory=list)

    @field_validator("question", "answer")
    @classmethod
    def _strip(cls, value: str) -> str:
        return " ".join(value.split())


# --------------------------------------------------------------------------
# A candidate pair: raw output of the generator for one chunk, BEFORE the
# document-level consolidation step merges repeated facts.
# --------------------------------------------------------------------------
@dataclass
class Candidate:
    cand_id: str
    doc_id: str
    chunk_index: int
    question: str
    answer: str
    sentence_ids: list[int]
    q_embedding: Optional[list[float]] = None    # cached vector of the question (for grouping)


# --------------------------------------------------------------------------
# A final stored Q&A pair as read back from SQLite (used for embedding and search).
# --------------------------------------------------------------------------
@dataclass
class QARecord:
    pair_id: str
    doc_id: str
    chunk_index: int                 # first chunk the fact appeared in
    question: str
    answer: str
    sentence_ids: list[int]          # ALL supporting sentences, from every section
    source_pages: list[int]          # pages of those sentences (PDFs)
    sections: list[str]              # every sub-section the answer draws on
    merged_from: list[str]           # candidate ids combined into this pair (1 = not merged)
    source: str
    title: str
    # Document metadata (P4a), used for richer citations and search filters.
    authors: list[str] = field(default_factory=list)
    published: Optional[str] = None  # YYYY[-MM[-DD]]
    doc_type: Optional[str] = None
    language: Optional[str] = None

    @property
    def year(self) -> Optional[int]:
        return int(self.published[:4]) if self.published and self.published[:4].isdigit() else None

    def citation(self) -> str:
        """Short reference for answers: 'Title (Author et al., 2016)'."""
        bits = []
        if self.authors:
            bits.append(self.authors[0] + (" et al." if len(self.authors) > 1 else ""))
        if self.year:
            bits.append(str(self.year))
        return f"{self.title} ({', '.join(bits)})" if bits else self.title
