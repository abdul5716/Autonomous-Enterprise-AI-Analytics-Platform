.PHONY: setup up down nuke health sandbox data ask lint fmt typecheck test check

AS ?= finance.admin

setup:
	uv sync
	@test -f .env || cp .env.example .env

up:
	docker compose up -d
	@echo "Oracle takes ~1 minute on first start. Run 'make health' to check."

down:
	docker compose down

nuke:
	docker compose down -v

health:
	uv run python -m analytics_platform.healthcheck

sandbox:
	docker build -t aeap-sandbox:latest -f infra/sandbox/Dockerfile .

data:
	uv run python -m data_gen
	uv run python -m analytics_platform.tools.rag.ingest

ask:  # make ask Q="Why did revenue drop last quarter?" AS=emea.sales
	uv run python -m analytics_platform.agent "$(Q)" --user $(AS)

lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:
	uv run ruff check --fix .
	uv run ruff format .

typecheck:
	uv run mypy src tests

test:
	uv run pytest

check: lint typecheck test
