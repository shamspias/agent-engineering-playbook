.PHONY: all install dev test lint format clean help run examples

# Default target
all: help

# Installation targets
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]" "langgraph-cli[inmem]"

install-llm:
	pip install -e ".[llm]"

# Testing targets
test:
	python -m pytest tests/ -v

test-unit:
	python -m pytest tests/unit_tests/ -v

test-integration:
	python -m pytest tests/integration_tests/ -v

test-cov:
	python -m pytest tests/ --cov=src/agent --cov-report=html --cov-report=term

test-watch:
	python -m pytest-watch tests/

# Linting and formatting
lint:
	python -m ruff check src/ tests/
	python -m mypy src/

format:
	python -m ruff format src/ tests/ examples.py
	python -m ruff check --fix src/ tests/ examples.py

# Running the agent
run:
	langgraph dev

examples:
	python examples.py

# Cleaning
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".coverage" -delete
	rm -rf dist/ build/

# Documentation
docs:
	@echo "Opening documentation..."
	@echo "README: ./README.md"
	@echo "Architecture: ./docs/ARCHITECTURE.md"

# Docker targets
docker-build:
	docker build -t tool-call-agent .

docker-run:
	docker run -p 8000:8000 --env-file .env tool-call-agent

# Help
help:
	@echo "Tool Call Agent - Available Commands"
	@echo "===================================="
	@echo ""
	@echo "Installation:"
	@echo "  make install          - Install package"
	@echo "  make install-dev      - Install with dev dependencies"
	@echo "  make install-llm      - Install with LLM provider support"
	@echo ""
	@echo "Testing:"
	@echo "  make test            - Run all tests"
	@echo "  make test-unit       - Run unit tests only"
	@echo "  make test-integration - Run integration tests only"
	@echo "  make test-cov        - Run tests with coverage report"
	@echo "  make test-watch      - Run tests in watch mode"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint            - Run linters (ruff, mypy)"
	@echo "  make format          - Format code with ruff"
	@echo ""
	@echo "Running:"
	@echo "  make run             - Start LangGraph dev server"
	@echo "  make examples        - Run example usage scripts"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean           - Remove build artifacts and caches"
	@echo "  make docs            - View documentation"
	@echo "  make docker-build    - Build Docker image"
	@echo "  make docker-run      - Run in Docker container"
	@echo ""