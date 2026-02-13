.PHONY: sync test lint

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

sync:
	python3 -m venv $(VENV)
	$(PIP) install -e .

test:
	$(PYTHON) -m pytest -v

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .
