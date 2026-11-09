from finrag.models import Chunk
from finrag.pipeline import FinRAGPipeline
from finrag.retrieval import BM25Index


def test_answer_contains_auditable_citation():
    chunk = Chunk("filing#c0", "filing", "Filing", "FY25", "Revenue increased to $10 million.", 0)
    answer = FinRAGPipeline(BM25Index([chunk])).ask("What was revenue?")
    assert "filing#c0" in answer.citations
    assert "[filing#c0]" in answer.answer


def test_pipeline_abstains_without_evidence():
    chunk = Chunk("filing#c0", "filing", "Filing", "FY25", "Revenue increased.", 0)
    answer = FinRAGPipeline(BM25Index([chunk])).ask("What was employee satisfaction?")
    assert answer.citations == ()
    assert answer.support_score == 0.0

