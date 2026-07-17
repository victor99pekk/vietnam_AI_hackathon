VENV    := source .venv/bin/activate
dataset ?= small

# ── Dataset configs ──
# Add new datasets here
dataset_conf.small     := configs/debug.yaml
dataset_input.small    := -i data/debugg_sample/

dataset_conf.wikipedia     := configs/pipeline.yaml
dataset_input.wikipedia    := -i data/wikipedia/

.PHONY: help test ingest upload install clean

help:
	@echo "Usage: make <target> [dataset=<name>]"
	@echo ""
	@echo "Targets:"
	@sed -n 's/^##//p' $(MAKEFILE_LIST) | column -t -s ':' | sed 's/^/ /'

## test: Run the test suite
test:
	$(VENV) && python -m pytest tests/ -v

## ingest: Run the pipeline  [dataset=small|wikipedia]
ingest:
	$(VENV) && kg-gen run -c $(dataset_conf.$(dataset)) $(dataset_input.$(dataset)) -o ./output_$(dataset)

## upload: Upload the generated graph to Neo4j  [dataset=small|wikipedia]
upload:
	$(VENV) && kg-gen neo4j-upload -o ./output_$(dataset)

## install: Set up the project and install dependencies
install:
	uv venv; \
	$(VENV) && uv pip install -e ".[embeddings,dev,neo4j]" && \
	python -m spacy download en_core_web_sm

## clean: Remove generated output folders
clean:
	rm -rf output_small/ output_wikipedia/
