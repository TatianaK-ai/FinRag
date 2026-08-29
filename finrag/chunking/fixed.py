"""Fixed-size chunking: recursive character splitting."""
from __future__ import annotations

import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .common import Chunk, group_runs, make_chunk, split_table

# ~1200 chars is roughly 300 tokens. Paired deliberately with
# text-embedding-3-small @1536d: small enough that one chunk carries a single
# claim, large enough that a financial sentence keeps its qualifying clause.
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
)

_LEADING_PUNCT = re.compile(r"^\s*[.;,]\s*")


def chunk_fixed(doc: dict) -> list[Chunk]:
    chunks: list[Chunk] = []
    i = 0
    for run in group_runs(doc["blocks"]):
        if run.type == "table":
            for part in split_table(run.text):
                chunks.append(make_chunk(doc, text=part, section=run.section,
                                         is_table=True, strategy="fixed", i=i))
                i += 1
            continue
        for piece in _splitter.split_text(run.text):
            # The recursive splitter keeps the separator it split on, so pieces
            # often open with a stray ". " - strip it or every chunk starts
            # mid-clause.
            text = _LEADING_PUNCT.sub("", piece).strip()
            if len(text) < 60:
                continue
            chunks.append(make_chunk(doc, text=text, section=run.section,
                                     is_table=False, strategy="fixed", i=i))
            i += 1
    return chunks
