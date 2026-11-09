# Architecture and engineering decisions

## System boundary

FinRAG Lab is a research baseline for answering questions over a small, trusted financial corpus.
It is not presented as a production RAG platform or a trained financial model. The current answerer
is extractive: it selects source sentences and adds stable citations rather than generating prose.

## Components

| Component | Responsibility | Current implementation | Extension point |
|---|---|---|---|
| Ingestion | Validate source records | Strict JSON loader | SEC/SGX adapters |
| Chunking | Preserve provenance and stable IDs | Sentence-aware word budget | Structure-aware filing parser |
| Retrieval | Rank evidence for a question | Dependency-free BM25 | `Retriever` protocol |
| Answering | Return only supported claims | Extractive sentence selection | Provider-neutral LLM answerer |
| Guardrail | Reject weakly supported questions | Score and term-coverage gate | Calibrated selective QA model |
| Evaluation | Detect retrieval and answer regressions | Labelled offline benchmark | Human review and confidence intervals |
| Serving | Expose the same measured pipeline | FastAPI `/query` endpoint | Auth, rate limits, tracing |

## Decisions

### Start with a lexical baseline

BM25 is inexpensive, deterministic, and interpretable. It creates a meaningful control condition
for later dense or hybrid retrieval experiments. Adding embeddings before measuring this baseline
would make it harder to attribute improvements.

### Use stable chunk citations

Chunk IDs combine the source document ID and chunk position. Answers cite those IDs and API callers
receive the full retrieved evidence. This makes a failure auditable without access to hidden model
state.

### Do not call retrieval score “confidence”

Raw BM25 scores are corpus-dependent and are not probabilities. The API therefore reports an
interpretable `support_score`: the fraction of meaningful query terms found in the top evidence.
It remains a heuristic and is not claimed to be calibrated.

### Share one pipeline between serving and evaluation

The CLI, API, and evaluator construct the same `FinRAGPipeline`. This reduces training-serving or
evaluation-serving skew and makes CI results relevant to the demo behaviour.

## Known failure modes

- Synonyms and multi-hop questions can defeat lexical retrieval.
- Sentence extraction can return incomplete context or awkward answers.
- Term coverage is vulnerable to entity-only overlap and requires calibration on a larger dataset.
- Stable positional IDs change if a source document is re-chunked; a content hash should be added
  before operating a mutable corpus.
- The five-question synthetic benchmark is a smoke test, not evidence of external validity.
- The in-memory index is appropriate for the demo corpus but not for large filing collections.

## Production gaps

Before real financial use, the system needs licensed source connectors, document versioning,
content-hash provenance, prompt-injection controls for generative components, access control,
rate limiting, observability, PII handling, human evaluation, and calibrated abstention thresholds.
