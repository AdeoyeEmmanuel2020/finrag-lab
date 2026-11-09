from __future__ import annotations

import math
import re
from collections import Counter

from .models import Chunk, SearchResult

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9%$.-]+")


def tokenize(text: str) -> list[str]:
    return [token.lower().strip(".-") for token in TOKEN_PATTERN.findall(text) if token.strip(".-")]


class BM25Index:
    """Dependency-free BM25 baseline suitable for reproducible experiments."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        if not chunks:
            raise ValueError("Cannot build an index without chunks")
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.tokens = [tokenize(chunk.text) for chunk in chunks]
        self.term_frequencies = [Counter(tokens) for tokens in self.tokens]
        self.avg_length = sum(map(len, self.tokens)) / len(self.tokens)
        self.document_frequency: Counter[str] = Counter()
        for tokens in self.tokens:
            self.document_frequency.update(set(tokens))

    def _idf(self, term: str) -> float:
        n = len(self.chunks)
        frequency = self.document_frequency[term]
        return math.log(1 + (n - frequency + 0.5) / (frequency + 0.5))

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        terms = tokenize(query)
        if not terms or top_k < 1:
            return []
        scored: list[SearchResult] = []
        for chunk, frequencies, tokens in zip(self.chunks, self.term_frequencies, self.tokens):
            score = 0.0
            for term in terms:
                tf = frequencies[term]
                if not tf:
                    continue
                normalizer = tf + self.k1 * (1 - self.b + self.b * len(tokens) / self.avg_length)
                score += self._idf(term) * (tf * (self.k1 + 1)) / normalizer
            if score > 0:
                scored.append(SearchResult(chunk=chunk, score=score))
        return sorted(scored, key=lambda item: (-item.score, item.chunk.chunk_id))[:top_k]

