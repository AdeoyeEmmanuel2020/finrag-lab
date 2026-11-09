from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluate import evaluate, save_result
from .ingest import chunk_documents, load_documents
from .pipeline import FinRAGPipeline
from .retrieval import BM25Index


def build_pipeline(corpus: Path, max_words: int = 90) -> FinRAGPipeline:
    chunks = chunk_documents(load_documents(corpus), max_words=max_words)
    return FinRAGPipeline(BM25Index(chunks))


def main() -> None:
    parser = argparse.ArgumentParser(description="Citation-grounded financial RAG baseline")
    parser.add_argument("--corpus", type=Path, default=Path("data/sample_filings.json"))
    subparsers = parser.add_subparsers(dest="command", required=True)
    ask = subparsers.add_parser("ask", help="Ask a grounded question")
    ask.add_argument("question")
    ask.add_argument("--top-k", type=int, default=3)
    run_eval = subparsers.add_parser("evaluate", help="Run the reproducible benchmark")
    run_eval.add_argument("--dataset", type=Path, default=Path("data/eval_questions.json"))
    run_eval.add_argument("--output", type=Path, default=Path("experiments/results.jsonl"))
    args = parser.parse_args()
    pipeline = build_pipeline(args.corpus)
    if args.command == "ask":
        print(json.dumps(pipeline.ask(args.question, args.top_k).to_dict(), indent=2))
    else:
        result = evaluate(pipeline, args.dataset)
        save_result(result, args.output, args.dataset)
        print(json.dumps(result.__dict__, indent=2))


if __name__ == "__main__":
    main()

