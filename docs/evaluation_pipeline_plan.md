# KG → LLM Evaluation Pipeline Plan

## Goal

Prove that a Knowledge Graph (KG) provides a **measurable lift** over brute-force training on raw text, using an A/B/C controlled ablation study with a lightweight, Mac-runnable model.

## The 3-Model Experimental Setup

| Model | Name | Data Source | Purpose |
|-------|------|-------------|---------|
| **Model A** | Baseline | Stock Qwen2.5 (no fine-tuning) | Measures out-of-the-box knowledge |
| **Model B** | KG-Managed | Structured QA pairs generated from the KG (multi-hop, entity-linked) | Measures impact of clean, relational data |
| **Model C** | Unmanaged | Flat QA pairs from the same source documents, no KG curation | Measures "brute-force" baseline |

### Experimental Controls
1. **Same base model** — all three start from the exact same Qwen2.5 checkpoint
2. **Same hyperparameters** — identical LoRA rank, learning rate, epochs, batch size
3. **Token volume control** — Model B and C trained on ~same total tokens
4. **Same eval prompts** — all models evaluated on identical test questions

## Model Choice: Qwen2.5 (Mac-Optimized)

| Model | Size | RAM (4-bit) | Why |
|-------|------|-------------|-----|
| **Qwen2.5-1.5B-Instruct** | ~3GB | ~1.5GB | Fastest iteration, noticeable differences on small datasets |
| **Qwen2.5-3B-Instruct** | ~6GB | ~2.5GB | Best balance of quality vs. Mac resources |
| **Qwen2.5-7B-Instruct** | ~14GB | ~5GB | Best quality, needs 16GB+ Mac |

**Recommendation**: Start with **Qwen2.5-1.5B-Instruct** for fast iteration. The smaller model will show **larger relative improvements** from KG fine-tuning since it has less pre-existing knowledge to mask the effect.

## Architecture

```
src/kg_generator/evaluate/
├── __init__.py              # Updated exports
├── metrics.py               # [existing] Structural KG quality metrics
├── dataset_gen.py           # [NEW] Generate QA pairs from KG & raw text
├── finetune.py              # [NEW] LoRA fine-tuning with Unsloth/MLX
├── benchmark.py             # [NEW] Multi-dimensional evaluation
└── llm_eval_config.yaml     # [NEW] Eval pipeline configuration

scripts/
└── run_llm_eval.py          # [NEW] End-to-end orchestration script
```

## Pipeline Stages

### Stage 1: Dataset Generation (`dataset_gen.py`)

**Model B — KG-Managed QA Generation:**
1. Load the KG (`knowledge_graph.json`)
2. Traverse multi-hop paths (e.g., `[Entity A] → relation → [Entity B] → relation → [Entity C]`)
3. Generate diverse QA types:
   - **Single-hop factual**: "Where was Alan Turing born?"
   - **Multi-hop reasoning**: "Which institution did the person who broke the Enigma code study at?"
   - **Comparison**: "Compare the birthplaces of Turing and Curie."
   - **Negative sampling**: Generate false statements for true/false evaluation
4. Use a simple template-based approach (no external LLM needed for English)
5. Output: `output_eval/kg_qa_train.jsonl`, `output_eval/kg_qa_test.jsonl`

**Model C — Raw Text QA Generation:**
1. Load the original source documents (same data that built the KG)
2. Chunk text into passages
3. Generate flat, single-hop QA pairs from each chunk using simple heuristics
4. Match token count to Model B's dataset
5. Output: `output_eval/raw_qa_train.jsonl`, `output_eval/raw_qa_test.jsonl`

### Stage 2: Fine-Tuning (`finetune.py`)

Uses **Unsloth** (supports Apple Metal/MPS acceleration):
- LoRA fine-tuning with `r=16, alpha=32`
- 4-bit quantization (NF4) for memory efficiency
- ~100-500 training steps (enough for a small KG)
- Both Model B and C use identical config

```python
# Key Unsloth configuration
model = FastLanguageModel.from_pretrained(
    "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model, r=16, target_modules=["q_proj","k_proj","v_proj","o_proj",
                                  "gate_proj","up_proj","down_proj"],
    lora_alpha=32,
)
```

### Stage 3: Evaluation (`benchmark.py`)

**Metrics measured:**

| Metric | How Measured | What It Shows |
|--------|-------------|---------------|
| **Factual Accuracy** | Exact match / F1 on held-out QA pairs | Does KG fine-tuning improve factual recall? |
| **Multi-hop Reasoning** | Accuracy on 2+ hop questions | Can the model chain KG relations? |
| **Hallucination Rate** | % of answers containing entities NOT in the KG | Does KG data reduce fabrication? |
| **Consistency** | Same question rephrased → same answer | Is the model's knowledge stable? |
| **Perplexity** | Loss on a held-out text corpus | General language modeling quality |

**Evaluation flow:**
1. Load each fine-tuned adapter + base model
2. Run test QA pairs through each model
3. Compute metrics
4. Generate comparison report (JSON + Markdown)

## Output

```
output_eval/
├── kg_qa_train.jsonl        # Model B training data
├── kg_qa_test.jsonl         # Model B test data
├── raw_qa_train.jsonl       # Model C training data
├── raw_qa_test.jsonl        # Model C test data
├── model_b_kg/              # Fine-tuned KG model (LoRA adapter)
├── model_c_raw/             # Fine-tuned raw model (LoRA adapter)
├── results.json             # Full evaluation results
└── report.md                # Human-readable comparison report
```

## Success Criteria

- **Model B > Model C on multi-hop reasoning** → KG structure helps logical reasoning
- **Model B > Model C on hallucination** → KG curation reduces fabrication
- **Model B ≈ Model C on single-hop** → Both datasets contain the same facts (control check)
- **Model C > Model B on fluency** → KG data may be too rigid; needs conversational blending

## Quick Start

```bash
# 1. Install Unsloth (Mac compatible)
pip install unsloth

# 2. Generate QA datasets from your KG
python scripts/run_llm_eval.py --stage dataset_gen

# 3. Fine-tune both models
python scripts/run_llm_eval.py --stage finetune

# 4. Evaluate and compare
python scripts/run_llm_eval.py --stage evaluate

# Or run everything end-to-end:
python scripts/run_llm_eval.py --stage all
```

## Future Extensions

- **Vietnamese support**: When the KG switches to Vietnamese (`vi`), integrate VMLU / V-Bench benchmarks
- **LLM-as-Judge**: Use GPT-4o or Qwen3-235B to score open-ended generation quality
- **MLX backend**: Switch from Unsloth to Apple's MLX for even better Mac performance
- **GraphRAG comparison**: Compare KG-fine-tuned model vs. RAG-over-KG baseline
