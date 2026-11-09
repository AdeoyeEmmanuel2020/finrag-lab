from finrag.models import Chunk
from finrag.retrieval import BM25Index


def chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(chunk_id, "doc", "Title", "FY25", text, 0)


def test_relevant_chunk_ranks_first():
    index = BM25Index([chunk("revenue", "Revenue grew by 18 percent."), chunk("risk", "Cyber risk increased.")])
    assert index.search("revenue growth")[0].chunk.chunk_id == "revenue"


def test_empty_query_returns_no_results():
    index = BM25Index([chunk("one", "Useful evidence")])
    assert index.search("") == []

