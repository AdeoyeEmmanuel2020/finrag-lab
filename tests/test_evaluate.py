import json

from finrag.evaluate import EvalResult, evaluate, save_result
from finrag.models import Chunk
from finrag.pipeline import FinRAGPipeline
from finrag.retrieval import BM25Index


def test_evaluation_separates_retrieval_from_abstention(tmp_path):
    dataset = tmp_path / "eval.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "question": "What was revenue?",
                    "relevant_chunks": ["report#c0"],
                    "reference_answer": "Revenue was $10 million.",
                },
                {
                    "question": "What was the Bitcoin balance?",
                    "relevant_chunks": [],
                    "reference_answer": "",
                    "should_abstain": True,
                },
            ]
        )
    )
    chunk = Chunk("report#c0", "report", "Report", "FY25", "Revenue was $10 million.", 0)
    result = evaluate(FinRAGPipeline(BM25Index([chunk])), dataset)
    assert result.answerable_examples == 1
    assert result.retrieval_recall_at_k == 1.0
    assert result.mean_reciprocal_rank == 1.0


def test_experiment_record_contains_provenance(tmp_path):
    dataset = tmp_path / "eval.json"
    dataset.write_text("[]")
    output = tmp_path / "results.jsonl"
    result = EvalResult(1.0, 1.0, 1.0, 1.0, 1.0, 1, 1)
    save_result(result, output, dataset)
    record = json.loads(output.read_text())
    assert len(record["dataset_sha256"]) == 64
    assert "git_sha" in record
    assert "timestamp_utc" in record
