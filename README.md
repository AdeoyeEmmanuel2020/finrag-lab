# FinRAG Lab

**An evaluation-first, citation-grounded RAG baseline for financial research.**

[![CI](https://github.com/AdeoyeEmmanuel2020/finrag-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/AdeoyeEmmanuel2020/finrag-lab/actions)

FinRAG Lab answers questions over financial reports while exposing its evidence, confidence, and
failure behaviour. It is deliberately small enough to audit end to end: ingestion, sentence-aware
chunking, BM25 retrieval, extractive answering, abstention, evaluation, REST API, tests, and CI.

This is a research baseline, not an investment product. The included company reports are synthetic.

## Why this project

Many RAG demos optimise for a convincing answer. This project optimises for a **measurable answer**:
every claim carries a stable chunk citation, unsupported questions trigger abstention, and every
experiment can be rerun without a paid model or external service.

## Architecture

```text
Synthetic filings -> validated documents -> stable chunks -> BM25 index
                                                        |-> evidence + citations
Question ---------------------------------------------->|-> extractive answer
                                                        |-> evaluation metrics
```

The zero-dependency core makes the baseline deterministic. An LLM or dense retriever can be added
behind a narrow interface and evaluated against exactly the same benchmark.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,api]'

finrag ask "What concentration risk did NovaPay report?"
finrag evaluate
pytest -q
uvicorn finrag.api:app --reload
```

Example response:

```json
{
  "answer": "The two largest banking partners processed 44% of total payment volume. [nova-2025-annual#c0]",
  "citations": ["nova-2025-annual#c0"],
  "confidence": 0.71
}
```

## Evaluation

The checked-in benchmark tests retrieval, citation quality, answer coverage, and abstention. Run:

```bash
PYTHONPATH=src python -m finrag.cli evaluate
```

Results are appended as JSON Lines under `experiments/` so runs remain machine-readable. See
[Experiment 001](docs/EXPERIMENT.md) for the hypothesis, protocol, and limitations.

Current deterministic baseline (5 labelled questions):

| Metric | Score |
|---|---:|
| Retrieval hit rate | 100% |
| Citation precision | 100% |
| Abstention accuracy | 100% |
| Answer token recall | 79.6% |

## API

```bash
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"How did NovaPay revenue change?","top_k":3}'
```

Interactive OpenAPI documentation is available at `/docs` when the service is running.

## Research and engineering roadmap

- Add SEC/SGX filing connectors with provenance and temporal filters.
- Compare BM25, dense, and hybrid retrieval with confidence intervals.
- Add a provider-neutral LLM answerer with claim-level entailment checks.
- Build an error-analysis dashboard and human-evaluation protocol.
- Add experiment metadata (git SHA, seed, dataset version, latency, and cost).

## Working with contributors

[The contributor workflow](docs/MENTORING.md) defines review expectations, reproducibility checks,
and scoped work packages for junior engineers or research interns.

## Responsible use

Outputs may be incomplete or wrong and must not be treated as financial advice. Production use would
require access control, prompt-injection defences, PII review, source licensing, monitoring, and
human oversight.

## License

MIT
