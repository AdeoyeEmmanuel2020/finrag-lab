# Engineering workflow for contributors

This repository models how I would coordinate a small research-engineering team.

1. Every task begins with a testable research question and acceptance criteria.
2. Contributors use short branches and small pull requests; the author includes commands and results.
3. Review checks correctness, experimental leakage, reproducibility, and maintainability.
4. Failed experiments are recorded because negative results narrow the search space.
5. Weekly milestones distinguish research uncertainty from engineering delivery risk.

## Suggested intern work packages

| Work package | Deliverable | Verification |
|---|---|---|
| Dense retrieval | Adapter plus benchmark comparison | Same dataset and metrics as baseline |
| Filing ingestion | Parser with provenance metadata | Golden-file and malformed-input tests |
| Error analysis | Labelled failure report | Second-reviewer agreement sample |
| Web demo | Read-only query interface | API contract and end-to-end tests |

