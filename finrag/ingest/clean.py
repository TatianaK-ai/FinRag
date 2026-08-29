"""
Turn a raw SEC filing into ordered, labelled blocks.

SEC filings are inline-XBRL documents: 1.5-8 MB of HTML in which a large fraction
of the "text" is machine-readable tagging (context refs, unit refs, `us-gaap:`
concept names). Naively taking the text yields runs like

    0000320193us-gaap:CommonStockIncludingAdditionalPaidInCapitalMember2022-09-24

which poison both embeddings and BM25. Stripping <ix:header> is the single
highest-value cleaning step in this pipeline (847 kB removed across six filings).
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from typing import Literal

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

# Filings are XHTML served as HTML; the html parser handles them correctly and the
# warning fires on every document.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

IX_HEADER = re.compile(r"<ix:header[\s\S]*?</ix:header>", re.I)
IX_HIDDEN = re.compile(r"<ix:hidden[\s\S]*?</ix:hidden>", re.I)

BOILERPLATE = [
    re.compile(r"^table of contents$", re.I),
    re.compile(r"^\d{1,4}$"),                                  # bare page numbers
    re.compile(r"\|\s*(20\d\d\s+)?form\s+10-[kq]\s*\|", re.I),  # running footers
    re.compile(r"^page\s+\d+(\s+of\s+\d+)?$", re.I),
    re.compile(r"^\(?(unaudited|in millions|in thousands)\)?$", re.I),
]

# Titles run long: Microsoft's Item 5 heading is 116 characters. A 90-character
# cap silently failed to match them, so those sections fell back to a bare
# "Item 5" from a cross-reference and lost their title entirely.
ITEM_RE = re.compile(r"^item\s+(\d{1,2}[a-z]?)\s*[.:—-]?\s*(.{0,160})$", re.I)
PART_RE = re.compile(r"^part\s+([ivx]+)\s*[.:—-]?\s*$", re.I)

LEAD_GLYPH = re.compile(r"^[$(]+$")
TRAIL_GLYPH = re.compile(r"^[)%]+$")
# A lone dash is the filing's "nil" marker (e.g. a 0% change). It is a VALUE, not
# punctuation - treating it as a prefix silently welds it onto the next figure and
# corrupts two cells at once.
NIL_CELL = re.compile(r"^[—–-]+$")


def _norm(s: str) -> str:
    return (
        s.replace(" ", " ")
        .replace("‘", "'").replace("’", "'")
        .replace("“", '"').replace("”", '"')
        .replace("–", "-").replace("—", "-")
        .replace("\t", " ")
    ).strip()


def _collapse_ws(s: str) -> str:
    return re.sub(r"[ \t]+", " ", s).strip()


def compact_row(cells: list[str]) -> list[str]:
    """
    SEC tables are ragged *per row*, not per column: the currency symbol lives in
    its own cell but is emitted only on the first and last row of a block, and
    spacer cells appear inconsistently. Column-wise cleanup therefore fails.

    Compacting each row independently - dropping empties, gluing `$`/`(` onto the
    figure that follows and `)`/`%` onto the one before - collapses every data row
    to the same arity.
    """
    out: list[str] = []
    lead = ""
    for raw in cells:
        c = raw.strip()
        if not c:
            continue
        if NIL_CELL.match(c):
            out.append(lead + c)
            lead = ""
            continue
        if LEAD_GLYPH.match(c):
            lead += c
            continue
        if TRAIL_GLYPH.match(c):
            if out:
                out[-1] += c
            else:
                out.append(c)
            continue
        out.append(lead + c)
        lead = ""
    if lead and out:
        out[-1] += lead
    return out


_FIGURE = re.compile(r"^[($]?[\d,]+\.?\d*[)%]?$")


def looks_like_header(cells: list[str]) -> bool:
    """
    Does row 0 hold column headings, or is it already data?

    Not every filing table has a header. Apple's term-debt maturity schedule
    starts straight at `| 2026 | $12,393 |`, and treating that as headings both
    loses a row of data and makes the synthesized caption claim the table "covers
    fiscal year 2026" when it runs to Thereafter. Reported figures are the tell.
    """
    if not cells:
        return True
    if cells[0] == "":
        return True                                     # classic SEC layout
    return not any(_FIGURE.match(c) and re.search(r"[,$]", c) for c in cells)


def normalise_grid(rows: list[list[str]]) -> list[list[str]]:
    if len(rows) < 2:
        return rows
    first = compact_row(rows[0])
    headerless = not looks_like_header(first)
    header = [] if headerless else first
    data = [compact_row(r) for r in (rows if headerless else rows[1:])]
    data = [r for r in data if r]
    if not data:
        return rows

    width = max(len(r) for r in data)
    grid = [(r + [""] * (width - len(r)))[:width] for r in data]

    # Re-seat header labels right-aligned over the value columns; column 0 is the
    # row-label column.
    out_header = [""] * width
    labels = header[-width:] if header else []
    offset = max(0, width - len(labels))
    for i, v in enumerate(labels):
        out_header[offset + i] = v
    return [out_header, *grid]


def table_to_markdown(table) -> str:
    """<table> -> GitHub-flavoured markdown, so numbers keep row/column meaning."""
    rows: list[list[str]] = []
    for tr in table.find_all("tr"):
        cells: list[str] = []
        for td in tr.find_all(["th", "td"], recursive=False):
            text = _collapse_ws(_norm(td.get_text()))
            try:
                span = max(1, int(td.get("colspan", 1)))
            except (TypeError, ValueError):
                span = 1
            cells.append(text)
            cells.extend([""] * (span - 1))
        if any(c for c in cells):
            rows.append(cells)

    normalised = normalise_grid(rows)
    # Keep row 0 even when entirely empty: for a headerless table `normalise_grid`
    # deliberately emits a blank heading row, and dropping it here would let the
    # first data row slide back into the header position.
    folded = [r for i, r in enumerate(normalised) if i == 0 or any(c for c in r)]
    if len(folded) < 2 or len(folded[0]) < 2:
        return " ".join(c for c in (x for row in folded for x in row) if c)

    width = len(folded[0])

    def esc(c: str) -> str:
        return c.replace("|", "\\|")[:200]

    head = "| " + " | ".join(esc(c) for c in folded[0]) + " |"
    sep = "| " + " | ".join(["---"] * width) + " |"
    body = [
        "| " + " | ".join(esc(r[i] if i < len(r) else "") for i in range(width)) + " |"
        for r in folded[1:]
    ]
    return "\n".join([head, sep, *body])


@dataclass
class Block:
    type: Literal["text", "table"]
    text: str
    section: str


def clean_filing(html: str) -> tuple[list[Block], dict]:
    stripped = IX_HIDDEN.sub("", IX_HEADER.sub("", html))
    soup = BeautifulSoup(stripped, "lxml")

    for tag in soup.find_all(["script", "style", "noscript"]):
        tag.decompose()
    for tag in soup.select('[style*="display:none"], [style*="display: none"]'):
        tag.decompose()

    # Pull tables out first, leaving a token behind so ordering survives.
    tables: list[str] = []
    for tbl in soup.find_all("table"):
        md = table_to_markdown(tbl)
        if not md or len(md) < 40:
            tbl.decompose()                              # layout scaffolding
            continue
        tables.append(md)
        tbl.replace_with(f"\n\n@@TABLE{len(tables) - 1}@@\n\n")

    # Force block-level boundaries into the text stream.
    for tag in soup.find_all(["p", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "br"]):
        tag.insert_after("\n")

    body = soup.body or soup
    lines = [_norm(l) for l in body.get_text().split("\n")]

    blocks: list[Block] = []
    section = "Front matter"
    part = ""
    item_titles: dict[str, str] = {}
    buffer: list[str] = []
    dropped = 0

    def flush() -> None:
        nonlocal buffer
        text = _collapse_ws(" ".join(buffer))
        buffer = []
        if len(text) >= 80:
            blocks.append(Block("text", text, section))

    table_token = re.compile(r"^@@TABLE(\d+)@@$")

    for line in lines:
        if not line:
            if buffer:
                flush()
            continue

        m = table_token.match(line)
        if m:
            flush()
            blocks.append(Block("table", tables[int(m.group(1))], section))
            continue

        if any(rx.search(line) for rx in BOILERPLATE):
            dropped += 1
            continue

        pm = PART_RE.match(line)
        if pm:
            flush()
            part = f"Part {pm.group(1).upper()}"
            continue

        im = ITEM_RE.match(line)
        if im and len(line) < 200:
            flush()
            num = im.group(1).upper()
            title = im.group(2).rstrip(".").strip()
            if title:
                item_titles[num] = title
            # A bare "Item 7" with no title is a running page header or a
            # cross-reference, not a new section. Microsoft's 10-K emits one a few
            # blocks after the real heading; taking it at face value stripped the
            # title from every block that followed - including the summary
            # financials table.
            known = title or item_titles.get(num, "")
            section = f"Item {num}" + (f". {known}" if known else "") + (f" ({part})" if part else "")
            continue

        buffer.append(line)

    flush()

    chars = sum(len(b.text) for b in blocks)
    stats = {
        "htmlBytes": len(html),
        "strippedBytes": len(stripped),
        "xbrlBytesRemoved": len(html) - len(stripped),
        "blocks": len(blocks),
        "tableBlocks": sum(1 for b in blocks if b.type == "table"),
        "textChars": chars,
        "boilerplateLinesDropped": dropped,
        "sections": len({b.section for b in blocks}),
    }
    return blocks, stats
