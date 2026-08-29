"""
Shared machinery for both chunking strategies.

Both strategies treat tables *identically* (atomic, header repeated when a table
must be split). That is deliberate: it holds table handling constant so the
fixed-vs-semantic comparison measures prose segmentation, not table parsing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

MAX_TABLE_CHARS = 2400


def split_table(md: str) -> list[str]:
    if len(md) <= MAX_TABLE_CHARS:
        return [md]
    lines = md.split("\n")

    # `table_to_markdown` emits a plain prose join for layout tables, which is
    # still stored as a table block. Row-splitting one of those produced ZERO
    # parts, so an oversized layout table vanished from both strategies silently.
    if not any(l.startswith("|") for l in lines):
        return [md[i:i + MAX_TABLE_CHARS] for i in range(0, len(md), MAX_TABLE_CHARS)]

    header = "\n".join(lines[:2])                      # header + separator
    parts: list[str] = []
    cur: list[str] = []
    length = len(header)
    for row in lines[2:]:
        if length + len(row) > MAX_TABLE_CHARS and cur:
            parts.append(f"{header}\n" + "\n".join(cur))
            cur, length = [], len(header)
        cur.append(row)
        length += len(row) + 1
    if cur:
        parts.append(f"{header}\n" + "\n".join(cur))

    if len(parts) > 1:
        parts = [f"{p}\n(table continued: part {i + 1} of {len(parts)})" for i, p in enumerate(parts)]
    return parts


@dataclass
class Run:
    type: str
    section: str
    text: str


def group_runs(blocks: Iterable) -> list[Run]:
    """Consecutive text blocks in the same section become one prose run."""
    runs: list[Run] = []
    cur: Run | None = None
    for b in blocks:
        if b.type == "table":
            cur = None
            runs.append(Run("table", b.section, b.text))
            continue
        if cur is None or cur.section != b.section:
            cur = Run("text", b.section, b.text)
            runs.append(cur)
        else:
            cur.text += " " + b.text
    return runs


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def make_chunk(doc: dict, *, text: str, section: str, is_table: bool, strategy: str, i: int) -> Chunk:
    return Chunk(
        id=f"{doc['docId']}::{strategy}::{i}",
        text=text,
        metadata={
            "docId": doc["docId"],
            "ticker": doc["ticker"],
            "company": doc["company"],
            "form": doc["form"],
            "filingDate": doc["filingDate"],
            "periodOfReport": doc.get("periodOfReport"),
            "section": section,
            "sourceUrl": doc["sourceUrl"],
            "isTable": bool(is_table),
            "strategy": strategy,
            "chunkIndex": i,
            "chars": len(text),
        },
    )


_CHANGE_COL = re.compile(r"change|growth|vs\.?|%", re.I)
_YEAR_START = re.compile(r"^\(?(19|20)\d{2}")
_YEAR_ANY = re.compile(r"(19|20)\d{2}")
_UNITS = re.compile(r"^\(?in (millions|thousands|billions)", re.I)


def table_caption(text: str, meta: dict) -> str:
    """
    A synthesized natural-language caption for a table chunk.

    Tables lose to prose in BOTH retrievers, for different reasons: a grid of
    numbers embeds nowhere near a natural-language question, and BM25 rewards
    prose that repeats the query's words while a table repeats nothing. Measured
    on this corpus, every passage surviving reranking for "Microsoft's total
    revenue for fiscal year 2026" was prose - the table holding the figure never
    reached the reranker.

    The specific gap is vocabulary: the table says `2026`, the question says
    "fiscal year 2026". So restate the table's own structure as a sentence. This
    invents nothing; every term comes from its row labels and column headers.
    """
    lines = [l for l in text.split("\n") if l.startswith("|")]
    if len(lines) < 2:
        return ""

    def cells(line: str) -> list[str]:
        return [c.strip() for c in line.split("|")[1:-1]]

    headings = [h for h in cells(lines[0]) if h]
    periods = [h for h in headings if _YEAR_ANY.search(h)]

    labels: list[str] = []
    for line in lines[2:]:
        c = cells(line)
        if not c:
            continue
        s = re.sub(r"\(\d+\)$", "", c[0]).strip()
        if s and 2 < len(s) < 60 and not re.match(r"^[$\d(.,%-]", s) and not _UNITS.match(s):
            labels.append(s)

    uniq = list(dict.fromkeys(labels))[:14]
    if not uniq and not periods:
        return ""

    parts = [f"Financial table from {meta['company']} ({meta['ticker']}) {meta['form']}."]

    # A change/growth column must be announced in words. Keeping only year-like
    # headings dropped "PercentageChange", so a table literally reporting
    # `| Revenue | $331,839 | $281,724 | 18% |` had a caption that never mentioned
    # growth - and a "which company grew fastest" question filled its Microsoft
    # slots with SEGMENT prose instead.
    change_cols = [h for h in headings if _CHANGE_COL.search(h) and not _YEAR_START.match(h)]
    if change_cols:
        parts.append(
            "Reports year-over-year change: revenue growth, percentage change and "
            f"increase or decrease versus the prior period ({'; '.join(change_cols)})."
        )

    if periods:
        years = list(dict.fromkeys(y.group(0) for h in periods for y in _YEAR_ANY.finditer(h)))
        parts.append("Covers fiscal year " + ", fiscal year ".join(years) + ".")
        parts.append("Column headings: " + "; ".join(periods) + ".")
    if uniq:
        parts.append("Reported line items: " + ", ".join(uniq) + ".")
    return " ".join(parts)


def embed_text(c: Chunk) -> str:
    """
    The indexed representation - used by BOTH the embeddings and the BM25 index,
    which must see the same text or hybrid fusion degrades.

    Provenance is prefixed because financial tables are lexically impoverished:
    `| Revenue | $331,839 | $281,724 |` names no company, no fiscal year and no
    document. The company NAME is included alongside the ticker on purpose -
    people ask about "Microsoft", not "MSFT".
    """
    m = c.metadata
    head = f"[{m['ticker']} {m['company']} {m['form']} {m['filingDate']} | {m['section']}]"
    caption = table_caption(c.text, m) if m.get("isTable") else ""
    return f"{head}\n{caption}\n{c.text}" if caption else f"{head}\n{c.text}"
