from __future__ import annotations

import json
import re
from pathlib import Path

from .models import Chunk, Document


def load_documents(path: Path) -> list[Document]:
    """Load small research corpora from JSON with strict field validation."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Corpus must be a JSON list")
    documents: list[Document] = []
    for row in raw:
        required = {"doc_id", "title", "period", "text"}
        if not isinstance(row, dict) or not required.issubset(row):
            raise ValueError(f"Every document needs fields: {sorted(required)}")
        documents.append(Document(**{key: str(row[key]) for key in required}))
    return documents


def chunk_documents(documents: list[Document], max_words: int = 90) -> list[Chunk]:
    """Create sentence-aware chunks with stable, auditable identifiers."""
    if max_words < 20:
        raise ValueError("max_words must be at least 20")
    chunks: list[Chunk] = []
    for doc in documents:
        sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", doc.text) if part]
        groups: list[str] = []
        current: list[str] = []
        size = 0
        for sentence in sentences:
            words = len(sentence.split())
            if current and size + words > max_words:
                groups.append(" ".join(current))
                current, size = [], 0
            current.append(sentence)
            size += words
        if current:
            groups.append(" ".join(current))
        for position, text in enumerate(groups):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}#c{position}",
                    doc_id=doc.doc_id,
                    title=doc.title,
                    period=doc.period,
                    text=text,
                    position=position,
                )
            )
    return chunks

