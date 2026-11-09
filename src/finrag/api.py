from __future__ import annotations

import os
from pathlib import Path

from .cli import build_pipeline

try:
    from fastapi import FastAPI
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Install API dependencies with: pip install -e '.[api]'") from exc

app = FastAPI(title="FinRAG Lab", version="0.1.0")
pipeline = build_pipeline(Path(os.getenv("FINRAG_CORPUS", "data/sample_filings.json")))


class Query(BaseModel):
    question: str = Field(min_length=5, max_length=500)
    top_k: int = Field(default=3, ge=1, le=10)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query")
def query(payload: Query) -> dict:
    return pipeline.ask(payload.question, payload.top_k).to_dict()

