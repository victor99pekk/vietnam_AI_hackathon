# UV Setup Guide for Collaborators

This guide is for team members who are new to `uv`. Follow these steps to get the project running on your machine.

---

## What is UV?

`uv` is a fast Python package manager (10–100× faster than pip). It replaces `pip`, `venv`, and `pip-tools` with a single tool. You use it the same way — just type `uv` instead of `pip`.

## One-Time Setup (First Time Only)

### 1. Install UV

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with Homebrew
brew install uv
```

Close and reopen your terminal, then verify:

```bash
uv --version
# Should print something like: uv 0.6.x
```

### 2. Clone the Project

```bash
git clone <repo-url>
cd hackathon
```

### 3. Create a Virtual Environment

```bash
uv venv
```

This creates a `.venv/` folder in the project. You'll see:

```
Using CPython 3.12.x
Creating virtual environment at: .venv
Activate with: source .venv/bin/activate
```

### 4. Activate the Environment

**Every time** you open a new terminal to work on this project, run:

```bash
source .venv/bin/activate
```

You'll know it's active when your prompt shows `(.venv)` at the beginning.

### 5. Install Dependencies

```bash
uv pip install -e "."
```

This reads `pyproject.toml` and installs everything. Add extras for full features:

```bash
# Basic (core pipeline only)
uv pip install -e "."

# With embedding-based entity resolution (recommended)
uv pip install -e ".[embeddings]"

# Everything — embeddings + Vietnamese NLP + Neo4j + LLM support
uv pip install -e ".[all]"
```

### 6. Download the spaCy Model

```bash
python -m spacy download en_core_web_sm
```

> If you get `No module named spacy`, make sure you ran step 4 (`source .venv/bin/activate`).

---

## Everyday Usage

### Activate → Run

```bash
source .venv/bin/activate    # Always do this first
kg-gen quick -i data/sample/
```

### Run with a config file

```bash
kg-gen run -c configs/pipelines/default.yaml
```

### Run tests

```bash
python -m pytest tests/ -v
```

### Add a new dependency

```bash
uv pip install pandas        # Install it
uv pip freeze > requirements.txt  # (optional) snapshot for others
```

Then add the package to `pyproject.toml` under `dependencies` so it's permanent.

---

## Common Problems & Fixes

| Problem | Fix |
|---|---|
| `kg-gen: command not found` | Run `source .venv/bin/activate` |
| `No module named spacy` | Run `source .venv/bin/activate`, then `python -m spacy download en_core_web_sm` |
| `sentence-transformers unavailable` | It's optional — the pipeline still works with string matching. Or install it: `uv pip install -e ".[embeddings]"` |
| Wrong Python version | `uv venv --python 3.12` to specify a version |
| Messed up environment | Delete `.venv/` and start from step 3 |

---

## Quick Reference Card

```bash
# Setup (first time)
uv venv
source .venv/bin/activate
uv pip install -e ".[embeddings]"
python -m spacy download en_core_web_sm

# Every session
source .venv/bin/activate

# Run pipeline
kg-gen quick -i data/sample/ -o output

# Run tests
python -m pytest tests/ -v

# Install new package
uv pip install <name>
```

---

## How to Update When Someone Adds Dependencies

```bash
git pull
source .venv/bin/activate
uv pip install -e ".[embeddings]"     # Re-run install to pick up new deps
```

---

## Deactivate When Done

```bash
deactivate
```

This exits the virtual environment and returns to your system Python.
