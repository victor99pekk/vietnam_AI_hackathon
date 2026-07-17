SHELL   := /bin/bash
VENV    := source .venv/bin/activate
dataset ?= small

# ── Dataset configs ──
# Add new datasets here
dataset_conf.small     := configs/debug.yaml
dataset_input.small    := -i data/debugg_sample/
dataset_kg.small       := generated_KGs/output_debug/knowledge_graph.json

dataset_conf.wikipedia     := configs/pipeline.yaml
dataset_input.wikipedia    := -i data/wikipedia/
dataset_kg.wikipedia       := generated_KGs/output/knowledge_graph.json

# ── Eval overrides ──
MODEL   ?= Qwen/Qwen2.5-0.5B-Instruct
model   ?= all
DEVICE  ?= cpu

.PHONY: help test ingest upload download plots LLM_plots install clean \
        eval-install eval-install-model eval-method1 eval-method2 eval-graphgen eval-all \
        eval-local eval-datasets

help:
	@echo "Usage: make <target> [dataset=<name>] [MODEL=<model>]"
	@echo ""
	@echo "── Pipeline ────────────────────────────────────────────"
	@echo "   test             Run the test suite"
	@echo "   ingest           Run the KG generation pipeline"
	@echo "   upload           Replace Neo4j contents with the generated graph"
	@echo "   download         Download graph from Neo4j → knowledge_graph.json"
	@echo "   plots            Generate PNG plots from evaluation results"
	@echo "   LLM_plots        Generate model comparison plots from ablation results"
	@echo "   install          Set up the project and install dependencies"
	@echo "   clean            Remove generated output folders"
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
	@echo "     → defaults to CPU — set DEVICE=cuda for GPU"
	@echo ""
	@echo "   eval-install           Install data_eval deps (deepeval, sentence-transformers)"
	@echo "   eval-install-model     Install model_eval deps (torch, transformers, peft — for fine-tuning)"
	@echo "   eval-method1           Run data_eval only"
	@echo "   eval-method2           Run model_eval [model=b|c|all] (CPU by default, DEVICE=cuda for GPU)"
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
	@echo ""
	@echo "── Examples ────────────────────────────────────────────"
	@echo "   make eval-method1                              # quick KG health check"
	@echo "   make eval-local                               # full local eval: audit + QA datasets"
	@echo "   make eval-datasets                            # generate QA datasets for Colab"
	@echo "   make eval-method2 model=b                     # fine-tune KG model (CPU)"
	@echo "   make eval-method2 model=b DEVICE=cuda          # fine-tune KG model (GPU)"
	@echo "   make eval-method2 model=b MODEL=Qwen/Qwen2.5-1.5B-Instruct"

## test: Run the test suite
test:
	$(VENV) && python -m pytest tests/ -v

## ingest: Run the pipeline  [dataset=small|wikipedia]
ingest:
	$(VENV) && kg-gen run -c $(dataset_conf.$(dataset)) $(dataset_input.$(dataset)) -o ./generated_KGs/output_$(dataset)

## upload: Clear Neo4j, then upload the generated graph  [dataset=small|wikipedia]
upload:
	$(VENV) && kg-gen neo4j-upload -o ./generated_KGs/output_$(dataset) --clear

## download: Download the graph from Neo4j (e.g., for Colab evaluation)  [dataset=small|wikipedia]
download:
	$(VENV) && kg-gen neo4j-download -o ./generated_KGs/output_$(dataset)/knowledge_graph.json

## plots: Generate visual plots from evaluation results  [dataset=small|wikipedia]
plots:
	$(VENV) && python -m generate_plots

## LLM_plots: Generate model comparison plots from ablation results (Method 2)
LLM_plots:
	$(VENV) && python -m generate_plots --method 2 --ablation output_eval/small_data/method2/method2_results.json --output output_eval/small_data/method2

## install: Set up the project and install dependencies
install:
	uv venv; \
	$(VENV) && uv pip install -e ".[curation,dev,neo4j]" && \
	python -m spacy download en_core_web_sm

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
