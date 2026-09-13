# Configuration
#--------------

PYTHON := uv run python

MAIN := src

MYPY_FLAGS := --warn-return-any \
              --warn-unused-ignores \
              --ignore-missing-imports \
              --disallow-untyped-defs \
              --check-untyped-defs


# Rules
#------

install:
	uv sync

run:
	$(PYTHON) -m $(MAIN)

debug:
	$(PYTHON) -m pdb -m $(MAIN)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .mypy_cache

lint:
	uv run flake8 .
	uv run mypy . $(MYPY_FLAGS)

lint-strict:
	uv run flake8 .
	uv run mypy . --strict

.PHONY: install run debug clean lint lint-strict