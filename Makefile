.PHONY: install test lint evaluate api

install:
	python -m pip install -e '.[dev,api]'

test:
	pytest -q

lint:
	ruff check src tests

evaluate:
	python -m finrag.cli evaluate

api:
	uvicorn finrag.api:app --reload

