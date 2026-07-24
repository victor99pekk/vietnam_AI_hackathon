SHELL   := /bin/bash
VENV    := source .venv/bin/activate
dataset ?= small

# ═══════════════════════════════════════════════════════════
# Config & Variables
# ═══════════════════════════════════════════════════════════

# ── Dataset presets ──
dataset_conf.small      := configs/pipelines/debug.yaml
dataset_input.small     := -i data/debugg_sample/alan_turing.jsonl -i data/debugg_sample/marie_curie.jsonl
dataset_kg.small        := data/samples/sample_kg.json

dataset_conf.wikipedia   := configs/pipelines/default.yaml
dataset_input.wikipedia  := -i data/wikipedia/
dataset_kg.wikipedia     := data/samples/sample_kg.json

# ── Overrides ──
MODEL        ?= Qwen/Qwen2.5-0.5B-Instruct
variant      ?= both
DEVICE       ?= cpu
wiki_count   ?= 100
wiki_lang    ?= en
scrape_seed  ?= data/download_data/seeds/vietnamese_sources.txt
scrape_count ?= 50
scrape_lang  ?= vi
scrape_time  ?= 600
scrape_depth ?= 2

.PHONY: help test install clean
.PHONY: scrape scrape-llm-discover scrape-llm-clean scrape-full
.PHONY: download-wikipedia
.PHONY: ingest upload download new-graph add
.PHONY: neo4j-new-graph neo4j-add neo4j-add-file neo4j-clear neo4j-eval-method1
.PHONY: eval-install eval-install-model eval eval-finetune eval-graphgen eval-full eval-datasets
.PHONY: plots plots-llm

help:
	@echo "Usage: make <target> [dataset=small|wikipedia] [DEVICE=cpu|cuda] [MODEL=...]"
	@echo ""
	@echo "── Setup ────────────────────────────────────────────"
	@echo "   install          Create venv and install all dependencies"
	@echo "   clean            Remove generated output folders"
	@echo ""
	@echo "── Data Acquisition ─────────────────────────────────"
	@echo "   scrape           Scrape web pages into JSONL (Vietnamese by default)"
	@echo "   scrape-llm-discover  Use LLM to extract article URLs from listing pages"
	@echo "   scrape-llm-clean     Score, filter, and LLM-clean scraped pages"
	@echo "   scrape-full      Full scrape → discover → re-scrape → clean"
	@echo "   download-wikipedia   Download Wikipedia articles [wiki_count=100] [wiki_lang=en|vi]"
	@echo ""
	@echo "── Pipeline (NetworkX — in-memory, no external services) ─"
	@echo "   ingest           Run the full KG generation pipeline"
	@echo "   upload           Upload generated graph to Neo4j"
	@echo "   download         Download graph from Neo4j → JSON"
	@echo "   new-graph        Build KG + upload to Neo4j (clears existing)"
	@echo "   add              Add data/add/*.jsonl files to Neo4j"
	@echo ""
	@echo "── Pipeline (Neo4j — direct-to-database, scales beyond RAM) ─"
	@echo "   neo4j-new-graph  Build KG directly in Neo4j (clears existing)"
	@echo "   neo4j-add        Incrementally add data/add/*.jsonl to Neo4j"
	@echo "   neo4j-add-file   Add a single file to Neo4j [FILE=path/to/file.jsonl]"
	@echo "   neo4j-clear      Delete everything in Neo4j"
	@echo "   neo4j-eval-method1  Run structural audit directly against Neo4j"
	@echo ""
	@echo "── Evaluation ───────────────────────────────────────"
	@echo "   eval-install     Install data quality deps (deepeval, sentence-transformers)"
	@echo "   eval-install-model   Install fine-tuning deps (torch, transformers, peft)"
	@echo ""
	@echo "   eval             KG health check + SFT quality scoring → \"Is my graph any good?\""
	@echo "   eval-datasets    Generate QA training pairs → \"Give me training data\""
	@echo "   eval-finetune    Train & benchmark KG vs raw models [variant=kg|raw|both] → \"Prove KG data works\""
	@echo "   eval-graphgen    Subgraph + multi-hop QA generation"
	@echo "   eval-full        Quality → datasets → finetune → benchmark → \"Do everything\""
	@echo ""
	@echo "── Visualization ────────────────────────────────────"
	@echo "   plots            Generate PNG plots from evaluation results"
	@echo "   plots-llm        Generate model comparison plots from ablation results"
	@echo ""
	@echo "── Dev ──────────────────────────────────────────────"
	@echo "   test             Run the test suite"
	@echo ""
	@echo "── Variables ────────────────────────────────────────"
	@echo "   dataset   = small | wikipedia           (default: small)"
	@echo "   variant   = kg | raw | both             (default: both)"
	@echo "   MODEL     = Qwen/Qwen2.5-{0.5B,1.5B,3B,7B}-Instruct"
	@echo "   DEVICE    = cpu | cuda                  (default: cpu)"
	@echo ""
	@echo "── Quick Start ──────────────────────────────────────"
	@echo "   make install                                  # one-time setup"
	@echo "   make test                                     # verify everything works"
	@echo "   make eval                                     # quick KG quality check"
	@echo "   make eval-datasets                            # generate QA pairs for fine-tuning"
	@echo "   make eval-finetune variant=kg                 # prove KG data improves models (CPU)"
	@echo "   make eval-finetune variant=kg DEVICE=cuda     # same, on GPU"
	@echo "   make eval-full                                # run everything end-to-end"

# ═══════════════════════════════════════════════════════════
# Setup
# ═══════════════════════════════════════════════════════════

## install: Set up the project and install dependencies
install:
	uv venv; \
	$(VENV) && uv pip install -e ".[curation,dev,neo4j,data,mongo,embeddings]" && \
	python -m spacy download en_core_web_sm && \
	python -m spacy download en_core_web_lg

## clean: Remove generated output folders
clean:
	rm -rf generated_KGs/

# ═══════════════════════════════════════════════════════════
# Data Acquisition
# ═══════════════════════════════════════════════════════════

## scrape: Scrape web pages into JSONL  [scrape_seed=path] [scrape_count=50] [scrape_lang=vi] [scrape_depth=2]
scrape:
	$(VENV) && python scripts/scraper.py \
		--seed-file $(scrape_seed) \
		--max-pages $(scrape_count) \
		--language $(scrape_lang) \
		--max-time $(scrape_time) \
		--depth $(scrape_depth) \
		--min-unique-chars 120 \
		--discovery auto \
		--output data/scraped/vn_web_$(scrape_count)/

## scrape-llm-discover: Use LLM to extract article URLs from listing pages  [scrape_count=50]
scrape-llm-discover:
	$(VENV) && python scripts/llm_cleaner.py discover \
		data/scraped/vn_web_$(scrape_count)/vn_web_$(scrape_count).jsonl \
		-o data/download_data/seeds/discovered_urls.txt

## scrape-llm-clean: Score, filter, and LLM-clean scraped pages  [scrape_count=50] [llm_min_score=5]
llm_min_score ?= 5
scrape-llm-clean:
	$(VENV) && python scripts/llm_cleaner.py clean \
		data/scraped/vn_web_$(scrape_count)/vn_web_$(scrape_count).jsonl \
		-o data/scraped/vn_web_$(scrape_count)/vn_web_$(scrape_count)_clean.jsonl \
		--min-score $(llm_min_score)

## scrape-full: Scrape + LLM discover article URLs + re-scrape articles + LLM clean
scrape-full: scrape scrape-llm-discover
	$(VENV) && python scripts/scraper.py \
		--seed-file data/download_data/seeds/discovered_urls.txt \
		--max-pages $$(wc -l < data/download_data/seeds/discovered_urls.txt) \
		--language $(scrape_lang) \
		--max-time 0 \
		--depth 0 \
		--discovery exact \
		--output data/scraped/vn_web_articles/ && \
	$(VENV) && python scripts/llm_cleaner.py clean \
		data/scraped/vn_web_articles/vn_web_articles.jsonl \
		-o data/scraped/vn_web_articles/vn_web_articles_clean.jsonl \
		--min-score 3

## download-wikipedia: Download Wikipedia articles as JSONL  [wiki_count=100] [wiki_lang=en]
download-wikipedia:
	@vietnam_flag=""; \
	if [ "$(wiki_lang)" = "vi" ]; then vietnam_flag="--vietnam-only"; fi; \
	$(VENV) && uv pip install -e ".[data]" && python scripts/download_wikipedia.py \
		--language $(wiki_lang) \
		--count $(wiki_count) \
		$$vietnam_flag \
		--output data/wikipedia/wikipedia_$(wiki_lang)_$(wiki_count).jsonl

# ═══════════════════════════════════════════════════════════
# Pipeline (NetworkX — in-memory)
# ═══════════════════════════════════════════════════════════

## ingest: Run the full KG generation pipeline  [dataset=small|wikipedia]
ingest:
	$(VENV) && kg-gen run -v \
		-c $(dataset_conf.$(dataset)) \
		$(dataset_input.$(dataset)) \
		-o ./generated_KGs/output_$(dataset)

## upload: Clear Neo4j, then upload the generated graph  [dataset=small|wikipedia]
upload:
	$(VENV) && kg-gen neo4j-upload -o ./generated_KGs/output_$(dataset) --clear

## download: Download the graph from Neo4j → JSON  [dataset=small|wikipedia]
download:
	$(VENV) && kg-gen neo4j-download -o ./generated_KGs/output_$(dataset)/knowledge_graph.json

## new-graph: Build KG from data and upload to Neo4j (clears existing graph)  [dataset=small|wikipedia]
new-graph: ingest upload

## add: Add all .jsonl files from data/add/ to the existing Neo4j graph  [dataset=small|wikipedia]
add:
	@files=$$(find data/add -name '*.jsonl' -type f 2>/dev/null); \
	if [ -z "$$files" ]; then \
		echo "ERROR: no .jsonl files found in data/add/"; \
		exit 1; \
	fi; \
	echo "Adding files:"; \
	echo "$$files" | sed 's/^/  /'; \
	input_args=$$(echo "$$files" | sed 's/^/-i /' | tr '\n' ' '); \
	$(VENV) && kg-gen run -v \
		-c $(dataset_conf.$(dataset)) \
		$$input_args \
		-o ./generated_KGs/output_$(dataset)_add; \
	$(VENV) && kg-gen neo4j-upload \
		-o ./generated_KGs/output_$(dataset)_add

# ═══════════════════════════════════════════════════════════
# Pipeline (Neo4j — direct-to-database)
# ═══════════════════════════════════════════════════════════

## neo4j-new-graph: Build KG directly in Neo4j (clears existing graph first)  [dataset=small|wikipedia]
neo4j-new-graph:
	$(VENV) && kg-gen run -v \
		-c $(dataset_conf.$(dataset)) \
		$(dataset_input.$(dataset)) \
		-o ./generated_KGs/output_$(dataset) \
		--backend neo4j \
		--clear

## neo4j-add: Incrementally add all .jsonl files from data/add/ to the existing Neo4j graph
neo4j-add:
	@files=$$(find data/add -name '*.jsonl' -type f 2>/dev/null); \
	if [ -z "$$files" ]; then \
		echo "ERROR: no .jsonl files found in data/add/"; \
		exit 1; \
	fi; \
	echo "Adding files:"; \
	echo "$$files" | sed 's/^/  /'; \
	input_args=$$(echo "$$files" | sed 's/^/-i /' | tr '\n' ' '); \
	$(VENV) && kg-gen add-doc -v \
		-c $(dataset_conf.$(dataset)) \
		$$input_args \
		-o ./generated_KGs/output_$(dataset)

## neo4j-add-file: Add a single file to Neo4j (usage: make neo4j-add-file FILE=data/new_article.jsonl)
neo4j-add-file:
	@test -f "$(FILE)" || { echo "ERROR: $(FILE) not found"; exit 1; }
	$(VENV) && kg-gen add-doc -v -c $(dataset_conf.$(dataset)) -i "$(FILE)" -o ./generated_KGs/output_$(dataset)

## neo4j-clear: Delete everything in Neo4j
neo4j-clear:
	$(VENV) && kg-gen neo4j-clear --yes

## neo4j-eval-method1: Run structural audit directly against Neo4j (no JSON download, no RAM limit)
neo4j-eval-method1:
	$(VENV) && python -m kg_generator.evaluate.run_eval --method quality --kg $(dataset_kg.$(dataset)) --neo4j -o output_eval/$(dataset)

# ═══════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════

## eval-install: Install data quality deps (deepeval + sentence-transformers)
eval-install:
	$(VENV) && uv pip install -e ".[eval-data]"

## eval-install-model: Install fine-tuning deps (torch, transformers, peft)
eval-install-model:
	$(VENV) && uv pip install -e ".[eval-model]"

## eval: KG health check + SFT quality scoring  [dataset=small|wikipedia]
eval:
	$(VENV) && python -m kg_generator.evaluate.run_eval --method quality --kg $(dataset_kg.$(dataset))

## eval-finetune: Train & benchmark KG vs raw models  [variant=kg|raw|both] [dataset=small|wikipedia] [MODEL=...] [DEVICE=cpu|cuda]
eval-finetune:
	@case "$(variant)" in \
		benchmark) target="--skip-finetune" ;; \
		kg)        target="-t kg" ;; \
		raw)       target="-t raw" ;; \
		*)         target="-t both" ;; \
	esac; \
	$(VENV) && python -m kg_generator.evaluate.run_eval --method ablation --kg $(dataset_kg.$(dataset)) $$target --model $(MODEL) --device $(DEVICE)

## eval-graphgen: Subgraph + multi-hop QA generation  [dataset=small|wikipedia]
eval-graphgen:
	$(VENV) && python -m kg_generator.evaluate.run_eval --method graphgen --kg $(dataset_kg.$(dataset))

## eval-full: Quality → datasets → finetune → benchmark  [dataset=small|wikipedia] [MODEL=...] [DEVICE=cpu|cuda]
eval-full:
	$(VENV) && python -m kg_generator.evaluate.run_eval --method full --kg $(dataset_kg.$(dataset)) --model $(MODEL) --device $(DEVICE)

## eval-datasets: Generate QA training pairs only (no evaluation, no fine-tuning)  [dataset=small|wikipedia]
eval-datasets:
	$(VENV) && python -m kg_generator.evaluate.run_eval --method ablation --kg $(dataset_kg.$(dataset)) --skip-finetune --skip-benchmark

# ═══════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════

## plots: Generate PNG plots from evaluation results  [dataset=small|wikipedia]
plots:
	$(VENV) && python -m kg_generator.evaluate.plots

## plots-llm: Generate model comparison plots from ablation results (Method 2)
plots-llm:
	$(VENV) && python -m kg_generator.evaluate.plots --method 2 --ablation output_eval/small_data/method2/method2_results.json --output output_eval/small_data/method2

# ═══════════════════════════════════════════════════════════
# Dev
# ═══════════════════════════════════════════════════════════

## test: Run the test suite
test:
	$(VENV) && python -m pytest tests/ -v
