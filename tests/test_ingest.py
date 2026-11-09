import pytest

from finrag.ingest import chunk_documents
from finrag.models import Document


def test_chunk_ids_are_stable():
    document = Document("report", "Report", "FY25", "One short sentence. Another sentence.")
    chunks = chunk_documents([document], max_words=20)
    assert [chunk.chunk_id for chunk in chunks] == ["report#c0"]


def test_rejects_tiny_chunk_size():
    with pytest.raises(ValueError):
        chunk_documents([], max_words=10)

