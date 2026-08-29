"""
Semantic chunking: cut where the subject changes, not at a character budget.

Embed each sentence in the context of its neighbours, measure cosine distance
between consecutive sentences, and cut where the distance exceeds the Nth
percentile *for that document* - i.e. where the topic actually shifts.
"""
from __future__ import annotations

import re

from ..lib.embed import cosine, embed_all
from .common import Chunk, group_runs, make_chunk, split_table

BUFFER_SIZE = 1        # sentences of context on each side when embedding
PERCENTILE = 90        # distance percentile that counts as a topic shift
MIN_CHARS = 350
MAX_CHARS = 2000

# Must not break on "$1.2 billion", "U.S.", "Inc.", "No. 1", "Note 12." or decimals.
ABBREV = re.compile(
    r"\b(?:U\.S|U\.K|Inc|Corp|Co|Ltd|LLC|L\.P|No|Nos|Ref|Art|Sec|St|Mr|Ms|Dr|vs"
    r"|approx|e\.g|i\.e|etc|Jr|Sr|Fig|Note)\.$", re.I)


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    out: list[str] = []
    for p in parts:
        prev = out[-1] if out else None
        glue = prev is not None and (
            ABBREV.search(prev.strip())          # abbreviation, not a sentence end
            or re.search(r"\d\.$", prev.strip())  # "12." - a decimal or list marker
            or re.match(r"^[a-z)]", p)            # continuation of the same sentence
        )
        if glue:
            out[-1] = f"{prev} {p}"
        else:
            out.append(p)
    return [s.strip() for s in out if s.strip()]


def _percentile(values: list[float], p: int) -> float:
    if not values:
        return float("inf")
    s = sorted(values)
    return s[min(len(s) - 1, int(p / 100 * len(s)))]


def chunk_semantic(doc: dict) -> list[Chunk]:
    runs = group_runs(doc["blocks"])

    # One embedding pass for the whole document keeps the batch efficient.
    spans: list[dict | None] = []
    flat: list[str] = []
    for ri, run in enumerate(runs):
        if run.type == "table":
            spans.append(None)
            continue
        sents = split_sentences(run.text)
        if len(sents) <= 1:
            spans.append(None)
            continue
        start = len(flat)
        for i in range(len(sents)):
            lo = max(0, i - BUFFER_SIZE)
            hi = min(len(sents), i + BUFFER_SIZE + 1)
            flat.append(" ".join(sents[lo:hi]))
        spans.append({"start": start, "end": len(flat), "sents": sents, "runIndex": ri})

    vectors = embed_all(flat, label=f"semantic:{doc['docId']}") if flat else []

    chunks: list[Chunk] = []
    i = 0
    for ri, run in enumerate(runs):
        if run.type == "table":
            for part in split_table(run.text):
                chunks.append(make_chunk(doc, text=part, section=run.section,
                                         is_table=True, strategy="semantic", i=i))
                i += 1
            continue

        span = next((s for s in spans if s and s["runIndex"] == ri), None)
        if span is None:
            text = run.text.strip()
            if len(text) >= 60:
                chunks.append(make_chunk(doc, text=text, section=run.section,
                                         is_table=False, strategy="semantic", i=i))
                i += 1
            continue

        vecs = vectors[span["start"]:span["end"]]
        dist = [1 - cosine(vecs[k], vecs[k + 1]) for k in range(len(vecs) - 1)]
        threshold = _percentile(dist, PERCENTILE)

        buf: list[str] = []
        length = 0

        def flush() -> None:
            nonlocal buf, length, i
            text = " ".join(buf).strip()
            buf, length = [], 0
            if len(text) >= 60:
                chunks.append(make_chunk(doc, text=text, section=run.section,
                                         is_table=False, strategy="semantic", i=i))
                i += 1

        for k, s in enumerate(span["sents"]):
            buf.append(s)
            length += len(s) + 1
            shift = k < len(dist) and dist[k] > threshold
            if (shift and length >= MIN_CHARS) or length >= MAX_CHARS:
                flush()
        if buf:
            flush()

    return chunks
