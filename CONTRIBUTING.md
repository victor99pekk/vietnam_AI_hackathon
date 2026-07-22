# Contributing

Thanks for your interest in contributing to the Knowledge Graph Generator!

## Setup

```bash
# Clone and set up
git clone https://github.com/vietnam-ai-challenge/kg-generator.git
cd kg-generator

# Using uv (recommended)
uv venv
source .venv/bin/activate
uv pip install -e ".[dev,curation,embeddings]"
python -m spacy download en_core_web_sm

# Run tests
python -m pytest tests/ -v
```

## Code Style

- Python 3.10+ with type hints
- Format with `ruff format src/ tests/`
- Lint with `ruff check src/ tests/`
- Type-check with `mypy src/`
- Run pre-commit hooks: `pre-commit run --all-files`

## Testing

```bash
# All tests
python -m pytest tests/ -v

# Specific test file
python -m pytest tests/test_extract.py -v

# With coverage
python -m pytest tests/ -v --cov=src/kg_generator --cov-report=term
```

## Project Structure

- `src/kg_generator/` — main package (pipeline, CLI, core modules)
- `src/kg_generator/evaluate/` — evaluation suite (data_eval, model_eval, graphgen, plots)
- `configs/` — YAML pipeline configuration presets
- `docs/` — user-facing documentation
- `data/` — sample inputs and curated outputs
- `tests/` — pytest test suite

## Pull Requests

1. Fork the repo and create a feature branch
2. Add tests for new functionality
3. Ensure all tests pass and linting is clean
4. Update documentation if needed
5. Submit a PR with a clear description

## Adding Features

- **New file format** → add loader in `src/kg_generator/ingest/loader.py`
- **New language** → add backend in `src/kg_generator/extract/entities.py` and wire into `config.py`
- **New export format** → add method in `src/kg_generator/export/exporter.py`
- **New quality metric** → add method in `src/kg_generator/evaluate/data_eval/metrics.py`
- **External KB linking** → implement `src/kg_generator/graph/enrich.py` stubs
