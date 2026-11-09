# Experiment 001 — lexical retrieval baseline

## Research question

Can a transparent lexical retriever provide reliable evidence selection for a small,
domain-specific financial corpus before introducing embedding or generative-model complexity?

## Hypothesis

BM25 should retrieve the correct filing passage for at least 80% of the benchmark questions and
the pipeline should abstain on questions unsupported by the corpus.

## Protocol

- Corpus: three synthetic company reports, deliberately free of licensing and privacy concerns.
- Unit of retrieval: sentence-aware chunks of at most 90 words.
- Retriever: BM25 (`k1=1.5`, `b=0.75`), top-k = 3.
- Answerer: deterministic extractive sentence selection.
- Metrics: retrieval hit rate, citation precision, token recall, abstention accuracy.
- Reproduction: `make evaluate` from a clean environment.

## Limitations and next experiment

The benchmark is intentionally small and synthetic, so it proves engineering reproducibility—not
real-world financial QA performance. The next experiment should ingest licensed public filings,
add temporal metadata filters, compare BM25 against dense retrieval, and use bootstrap confidence
intervals. A human-reviewed error taxonomy should separate retrieval, reasoning, and citation
failures.

