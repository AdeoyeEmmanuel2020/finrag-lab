from __future__ import annotations

import json
import hashlib
import subprocess
from datetime import datetime, timezone
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

from .pipeline import FinRAGPipeline
from .retrieval import tokenize


@dataclass(frozen=True)
class EvalResult:
    retrieval_recall_at_k: float
    mean_reciprocal_rank: float
    citation_precision: float
    answer_token_recall: float
    abstention_accuracy: float
    examples: int
    answerable_examples: int


def evaluate(pipeline: FinRAGPipeline, dataset_path: Path, top_k: int = 3) -> EvalResult:
    cases = json.loads(dataset_path.read_text(encoding="utf-8"))
    retrieval_hits, reciprocal_ranks, precisions, recalls, abstentions = [], [], [], [], []
    answerable = 0
    for case in cases:
        answer = pipeline.ask(case["question"], top_k=top_k)
        expected = set(case.get("relevant_chunks", []))
        ranked = [item.chunk.chunk_id for item in answer.evidence]
        expected_abstain = bool(case.get("should_abstain", False))
        did_abstain = not answer.citations
        if expected:
            answerable += 1
            matching_ranks = [
                rank for rank, chunk_id in enumerate(ranked, start=1) if chunk_id in expected
            ]
            retrieval_hits.append(float(bool(matching_ranks)))
            reciprocal_ranks.append(1 / min(matching_ranks) if matching_ranks else 0.0)
        precisions.append(
            len(set(answer.citations).intersection(expected)) / len(answer.citations)
            if answer.citations else float(expected_abstain)
        )
        gold_tokens = set(tokenize(case.get("reference_answer", "")))
        predicted_tokens = set(tokenize(answer.answer))
        recalls.append(
            len(gold_tokens.intersection(predicted_tokens)) / len(gold_tokens)
            if gold_tokens
            else 1.0
        )
        abstentions.append(float(did_abstain == expected_abstain))
    return EvalResult(
        mean(retrieval_hits) if retrieval_hits else 0.0,
        mean(reciprocal_ranks) if reciprocal_ranks else 0.0,
        mean(precisions),
        mean(recalls),
        mean(abstentions),
        len(cases),
        answerable,
    )


def save_result(result: EvalResult, path: Path, dataset_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        git_sha = "unknown"
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha,
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "metrics": asdict(result),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")

