from __future__ import annotations

import re

from .models import Answer
from .retrieval import Retriever, tokenize

STOPWORDS = {"a", "an", "did", "does", "how", "in", "is", "its", "of", "the", "to", "was", "what"}


class FinRAGPipeline:
    """Retrieval and extractive answer generation with explicit abstention."""

    def __init__(self, retriever: Retriever, min_score: float = 0.20):
        self.retriever = retriever
        self.min_score = min_score

    def ask(self, question: str, top_k: int = 3) -> Answer:
        question = question.strip()
        evidence = self.retriever.search(question, top_k=top_k)
        query_terms = {term for term in tokenize(question) if term not in STOPWORDS}
        top_terms = set(tokenize(evidence[0].chunk.text)) if evidence else set()
        weak_coverage = len(query_terms) >= 2 and len(query_terms.intersection(top_terms)) < 2
        if not evidence or evidence[0].score < self.min_score or weak_coverage:
            return Answer(
                question=question,
                answer="I do not have enough evidence in the indexed documents to answer reliably.",
                citations=(),
                evidence=tuple(evidence),
                support_score=0.0,
            )
        candidates: list[tuple[float, str, str]] = []
        top_result = evidence[0]
        sentences = re.split(r"(?<=[.!?])\s+", top_result.chunk.text)
        for sentence in sentences:
            overlap = len(query_terms.intersection(tokenize(sentence)))
            if overlap:
                candidates.append((overlap * top_result.score, sentence, top_result.chunk.chunk_id))
        if len(sentences) <= 2:
            candidates = [
                (
                    max(1, len(query_terms.intersection(tokenize(sentence)))) * top_result.score,
                    sentence,
                    top_result.chunk.chunk_id,
                )
                for sentence in sentences
            ]
        candidates.sort(key=lambda item: (-item[0], item[2]))
        selected = candidates[:2]
        answer = " ".join(f"{sentence} [{chunk_id}]" for _, sentence, chunk_id in selected)
        citations = tuple(dict.fromkeys(chunk_id for _, _, chunk_id in selected))
        # Interpretable query coverage, not a probabilistic confidence estimate.
        support_score = len(query_terms.intersection(top_terms)) / max(1, len(query_terms))
        return Answer(question, answer, citations, tuple(evidence), support_score)

