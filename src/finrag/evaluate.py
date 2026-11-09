from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

from .pipeline import FinRAGPipeline
from .retrieval import tokenize


@dataclass(frozen=True)
class EvalResult:
    retrieval_hit_rate: float
    citation_precision: float
    answer_token_recall: float
    abstention_accuracy: float
    examples: int


def evaluate(pipeline: FinRAGPipeline, dataset_path: Path, top_k: int = 3) -> EvalResult:
    cases = json.loads(dataset_path.read_text(encoding="utf-8"))
    hit_rates, precisions, recalls, abstentions = [], [], [], []
    for case in cases:
        answer = pipeline.ask(case["question"], top_k=top_k)
        expected = set(case.get("relevant_chunks", []))
        retrieved = {item.chunk.chunk_id for item in answer.evidence}
        expected_abstain = bool(case.get("should_abstain", False))
        did_abstain = not answer.citations
        hit_rates.append(float(bool(expected.intersection(retrieved))) if expected else float(did_abstain))
        precisions.append(
            len(set(answer.citations).intersection(expected)) / len(answer.citations)
            if answer.citations else float(expected_abstain)
        )
        gold_tokens = set(tokenize(case.get("reference_answer", "")))
        predicted_tokens = set(tokenize(answer.answer))
        recalls.append(len(gold_tokens.intersection(predicted_tokens)) / len(gold_tokens) if gold_tokens else 1.0)
        abstentions.append(float(did_abstain == expected_abstain))
    return EvalResult(mean(hit_rates), mean(precisions), mean(recalls), mean(abstentions), len(cases))


def save_result(result: EvalResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(result), sort_keys=True) + "\n")

