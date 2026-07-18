SHELL   := /bin/bash
VENV    := source .venv/bin/activate
dataset ?= small

# ── Dataset configs ──
# Add new datasets here
dataset_conf.small     := configs/debug.yaml
dataset_input.small    := -i data/debugg_sample/alan_turing.jsonl -i data/debugg_sample/marie_curie.jsonl
dataset_kg.small       := generated_KGs/output_debug/knowledge_graph.json

dataset_conf.wikipedia     := configs/pipeline.yaml
dataset_input.wikipedia    := -i data/wikipedia/
dataset_kg.wikipedia       := generated_KGs/output/knowledge_graph.json

# ── Eval overrides ──
MODEL   ?= Qwen/Qwen2.5-0.5B-Instruct
model   ?= all
DEVICE  ?= cpu
wiki_count ?= 100
wiki_lang  ?= en
scrape_seed  ?= data/download_data/seeds/vietnamese_sources.txt
scrape_count ?= 50
scrape_lang  ?= vi
scrape_time  ?= 600
scrape_depth ?= 2

.PHONY: help test ingest upload download download-wikipedia scrape scrape-standalone new-graph add plots LLM_plots install clean \
        neo4j-new-graph neo4j-add neo4j-add-file neo4j-clear \
        neo4j-eval-method1 \
        eval-install eval-install-model eval-method1 eval-method2 eval-graphgen eval-all \
        eval-local eval-datasets

help:
	@echo "Usage: make <target> [dataset=<name>] [MODEL=<model>]"
	@echo ""
	@echo "── Pipeline (networkx — in-memory, no external services) ─"
	@echo "   test             Run the test suite"
	@echo "   ingest           Run the KG generation pipeline"
	@echo "   new-graph        Build KG from data and upload to Neo4j [dataset=small|wikipedia]"
	@echo "   add              Add all .jsonl files from data/add/ to the Neo4j graph"
	@echo "   upload           Replace Neo4j contents with the generated graph"
	@echo "   download         Download graph from Neo4j → knowledge_graph.json"
	@echo "   download-wikipedia  Download Wikipedia articles as JSONL [wiki_count=100] [wiki_lang=en]"
	@echo "   install          Set up the project and install dependencies"
	@echo "   clean            Remove generated output folders"
	@echo ""
	@echo "── Pipeline (neo4j — on-disk, incremental, scales beyond RAM) ─"
	@echo "   neo4j-new-graph  Build KG directly in Neo4j (clears existing) [dataset=small|wikipedia]"
	@echo "   neo4j-add        Incrementally add files from data/add/ to Neo4j"
	@echo "   neo4j-add-file   Add a single file to Neo4j [FILE=path/to/file.jsonl]"
	@echo "   neo4j-clear      Delete everything in Neo4j"
	@echo "   neo4j-eval-method1  Run structural audit directly against Neo4j [dataset=small|wikipedia]"
	@echo ""
	@echo "── Visualization ────────────────────────────────────────"
	@echo "   plots            Generate PNG plots from evaluation results"
	@echo "   LLM_plots        Generate model comparison plots from ablation results"
	@echo ""
	@echo "── Evaluation ──────────────────────────────────────────"
	@echo ""
	@echo "   data_eval — SFT data quality check (fast, ~seconds, runs locally)"
	@echo "     → audits graph health (orphans, density, schema, dups)"
	@echo "     → generates & scores SFT training pairs via deepeval + DeepSeek"
	@echo ""
	@echo "   model_eval — Fine-tuning ablation study (CPU or GPU)"
	@echo "     → trains Model B (KG-structured QA pairs)"
	@echo "     → trains Model C (raw-text QA pairs)"
	@echo "     → benchmarks both vs. base Qwen2.5 (Model A)"
	@echo "     → CPU: default, small model (0.5B). GPU: optimized with large model (7B)"
	@echo ""
	@echo "   eval-install           Install data_eval deps (deepeval, sentence-transformers)"
	@echo "   eval-install-model     Install model_eval deps (torch, transformers, peft — for fine-tuning)"
	@echo "   eval-method1           Run data_eval only"
	@echo "   eval-method2           Run model_eval [model=b|c|all] (CPU by default, DEVICE=cuda for GPU)"
	@echo "   eval-method2-gpu       Run model_eval with GPU optimization (7B models, mixed precision, faster)"
	@echo "   eval-graphgen          Build audited k-hop subgraphs and multi-hop QA"
	@echo "   eval-local             Run data_eval + QA dataset generation (CPU, fast)"
	@echo "   eval-datasets          Generate QA datasets only (no fine-tuning, no benchmark)"
	@echo "   eval-all               Run data_eval + model_eval end-to-end"
	@echo ""
	@echo "── Variables ───────────────────────────────────────────"
	@echo "   dataset  = small | wikipedia          (default: small)"
	@echo "   model    = b | c | all                (default: all)"
	@echo "              b = KG-managed (Model B)"
	@echo "              c = raw-text (Model C)"
	@echo "              a = benchmark only (skip fine-tuning)"
	@echo "   MODEL    = Qwen/Qwen2.5-{0.5B,1.5B,3B,7B}-Instruct"
	@echo "             (default: Qwen2.5-0.5B — small enough for CPU)"
	@echo "   DEVICE   = cpu | cuda                 (default: cpu)"
	@echo "   wiki_count = 100                      (articles to download)"
	@echo "   wiki_lang  = en | vi                  (Wikipedia language)"
	@echo "   scrape_count = 50                     (pages to scrape)"
	@echo "   scrape_lang  = vi                     (scraping language)"
	@echo "   scrape_time  = 600                    (max seconds, 0=no limit)"
	@echo "   scrape_depth = 2                      (crawl link depth, 1=landing pages only)"
	@echo "   scrape_seed  = data/download_data/seeds/vietnamese_sources.txt"
	@echo ""
	@echo "── Examples ────────────────────────────────────────────"
	@echo "   # Classic pipeline (networkx in-memory → JSON → upload)"
	@echo "   make scrape                                       # scrape 50 pages from Vietnamese sources"
	@echo "   make scrape scrape_count=100 scrape_time=300       # 100 pages or 5 min"
	@echo "   make new-graph dataset=wikipedia                  # build KG + upload to Neo4j"
	@echo "   make add                                          # add data/add/*.jsonl to Neo4j"
	@echo ""
	@echo "   # Neo4j-native pipeline (direct-to-database, scales beyond RAM)"
	@echo "   make neo4j-new-graph dataset=wikipedia            # build KG directly in Neo4j"
	@echo "   make neo4j-add                                    # incrementally add data/add/*.jsonl"
	@echo "   make neo4j-add-file FILE=data/new_article.jsonl   # add a single file"
	@echo "   make neo4j-eval-method1 dataset=small             # audit graph directly in Neo4j"
	@echo ""
	@echo "   make download-wikipedia wiki_count=500            # download 500 en articles"
	@echo "   make eval-method1                                 # quick KG health check"
	@echo "   make eval-local                                  # full local eval: audit + QA datasets"
	@echo "   make eval-datasets                               # generate QA datasets for Colab"
	@echo "   make eval-method2 model=b                        # fine-tune KG model (CPU)"
	@echo "   make eval-method2 model=b DEVICE=cuda             # fine-tune KG model (GPU)"
	@echo "   make eval-method2 model=b MODEL=Qwen/Qwen2.5-1.5B-Instruct"

## test: Run the test suite
test:
	$(VENV) && python -m pytest tests/ -v

## ingest: Run the pipeline  [dataset=small|wikipedia]
ingest:
	$(VENV) && uv pip install -e ".[curation,neo4j,data,mongo,embeddings]" && \
	python -m spacy download en_core_web_lg && \
	kg-gen run -v -c $(dataset_conf.$(dataset)) $(dataset_input.$(dataset)) -o ./generated_KGs/output_$(dataset)

## upload: Clear Neo4j, then upload the generated graph  [dataset=small|wikipedia]
upload:
	$(VENV) && kg-gen neo4j-upload -o ./generated_KGs/output_$(dataset) --clear

## download: Download the graph from Neo4j (e.g., for Colab evaluation)  [dataset=small|wikipedia]
download:
	$(VENV) && kg-gen neo4j-download -o ./generated_KGs/output_$(dataset)/knowledge_graph.json

## scrape: Scrape Vietnamese web sources into JSONL  [scrape_seed=path] [scrape_count=50] [scrape_lang=vi] [scrape_time=600] [scrape_depth=2]
scrape:
	$(VENV) && python data/download_data/scraper.py \
		--seed-file $(scrape_seed) \
		--max-pages $(scrape_count) \
		--language $(scrape_lang) \
		--max-time $(scrape_time) \
		--depth $(scrape_depth) \
		--min-unique-chars 120 \
		--discovery auto \
		--output data/scraped/vn_web_$(scrape_count)/

## scrape-standalone: Run scraper directly (alias for scrape — same behavior)
scrape-standalone: scrape

## scrape-llm-discover: Use LLM to extract article URLs from listing pages  [scrape_count=50]
scrape-llm-discover:
	$(VENV) && python data/download_data/llm_cleaner.py discover \
		data/scraped/vn_web_$(scrape_count)/vn_web_$(scrape_count).jsonl \
		-o data/download_data/seeds/discovered_urls.txt

## scrape-llm-clean: Score, filter, and LLM-clean scraped pages  [scrape_count=50] [llm_min_score=5]
llm_min_score ?= 5
scrape-llm-clean:
	$(VENV) && python data/download_data/llm_cleaner.py clean \
		data/scraped/vn_web_$(scrape_count)/vn_web_$(scrape_count).jsonl \
		-o data/scraped/vn_web_$(scrape_count)/vn_web_$(scrape_count)_clean.jsonl \
		--min-score $(llm_min_score)

## scrape-full: Scrape + LLM discover article URLs + re-scrape articles + LLM clean
scrape-full: scrape scrape-llm-discover
	$(VENV) && python data/download_data/scraper.py \
		--seed-file data/download_data/seeds/discovered_urls.txt \
		--max-pages $$(wc -l < data/download_data/seeds/discovered_urls.txt) \
		--language $(scrape_lang) \
		--max-time 0 \
		--depth 0 \
		--discovery exact \
		--output data/scraped/vn_web_articles/ && \
	$(VENV) && python data/download_data/llm_cleaner.py clean \
		data/scraped/vn_web_articles/vn_web_articles.jsonl \
		-o data/scraped/vn_web_articles/vn_web_articles_clean.jsonl \
		--min-score 3

## download-wikipedia: Download Wikipedia articles as JSONL  [wiki_count=100] [wiki_lang=en]
download-wikipedia:
	@vietnam_flag=""; \
	if [ "$(wiki_lang)" = "vi" ]; then vietnam_flag="--vietnam-only"; fi; \
	$(VENV) && uv pip install -e ".[data]" && python data/download_data/wikipedia.py \
		--language $(wiki_lang) \
		--count $(wiki_count) \
		$$vietnam_flag \
		--output data/wikipedia/wikipedia_$(wiki_lang)_$(wiki_count).jsonl

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
# Neo4j-native pipeline (direct-to-database, scales beyond RAM)
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
	$(VENV) && python evaluation/run_eval.py --method 1 --kg $(dataset_kg.$(dataset)) --neo4j -o output_eval/$(dataset)

## plots: Generate visual plots from evaluation results  [dataset=small|wikipedia]
plots:
	$(VENV) && python -m generate_plots

## LLM_plots: Generate model comparison plots from ablation results (Method 2)
LLM_plots:
	$(VENV) && python -m generate_plots --method 2 --ablation output_eval/small_data/method2/method2_results.json --output output_eval/small_data/method2

## install: Set up the project and install dependencies
install:
	uv venv; \
	$(VENV) && uv pip install -e ".[curation,dev,neo4j,data,mongo,embeddings]" && \
	python -m spacy download en_core_web_sm && \
	python -m spacy download en_core_web_lg

## clean: Remove generated output folders
clean:
	rm -rf generated_KGs/

# ── Evaluation ────────────────────────────────────────────────

## eval-install: Install data_eval deps (deepeval + sentence-transformers)
eval-install:
	$(VENV) && uv pip install -e ".[eval-data]"

## eval-install-model: Install model_eval deps (transformers, peft, accelerate — for fine-tuning)
eval-install-model:
	$(VENV) && uv pip install -e ".[eval-model]"

## eval-method1: Run Method 1 — SFT data quality assessment  [dataset=small|wikipedia]
eval-method1:
	$(VENV) && python evaluation/run_eval.py --method 1 --kg $(dataset_kg.$(dataset))

## eval-method2: Run Method 2  [model=a|b|c|all] [dataset=small|wikipedia] [MODEL=...] [DEVICE=cpu|cuda]
eval-method2:
	@case "$(model)" in \
		a) target="--skip-finetune" ;; \
		b) target="-t kg" ;; \
		c) target="-t raw" ;; \
		*) target="-t both" ;; \
	esac; \
	$(VENV) && python evaluation/run_eval.py --method 2 --kg $(dataset_kg.$(dataset)) $$target --model $(MODEL) --device $(DEVICE)

## eval-method2-gpu: Run Method 2 with GPU optimization (7B models, mixed precision)  [model=a|b|c|all] [dataset=small|wikipedia]
eval-method2-gpu:
	@case "$(model)" in \
		a) target="--skip-finetune" ;; \
		b) target="-t kg" ;; \
		c) target="-t raw" ;; \
		*) target="-t both" ;; \
	esac; \
	$(VENV) && python evaluation/run_eval.py --method 2 --kg $(dataset_kg.$(dataset)) $$target --model $(MODEL) --gpu

## eval-graphgen: Run GraphGen-style subgraph + QA generation [dataset=small|wikipedia]
eval-graphgen:
	$(VENV) && python evaluation/run_eval.py --method graphgen --kg $(dataset_kg.$(dataset))

## eval-all: Run both evaluation methods end-to-end  [dataset=small|wikipedia] [MODEL=...]
eval-all:
	$(VENV) && python evaluation/run_eval.py --method all --kg $(dataset_kg.$(dataset)) --model $(MODEL)

## eval-local: Run Method 1 + generate QA datasets — everything you can do on a Mac CPU  [dataset=small|wikipedia]
eval-local:
	$(VENV) && python evaluation/run_eval.py --method all --kg $(dataset_kg.$(dataset)) --skip-finetune --skip-benchmark

## eval-datasets: Generate QA datasets only (no evaluation, no fine-tuning) — for Colab  [dataset=small|wikipedia]
eval-datasets:
	$(VENV) && python evaluation/run_eval.py --method 2 --kg $(dataset_kg.$(dataset)) --skip-finetune --skip-benchmark
