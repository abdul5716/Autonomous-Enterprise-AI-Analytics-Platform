.PHONY: setup up down nuke health lint fmt typecheck test check

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
