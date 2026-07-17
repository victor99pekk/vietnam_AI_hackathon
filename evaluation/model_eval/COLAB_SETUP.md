# Running model_eval on Google Colab

model_eval requires a CUDA GPU — it will NOT run on Mac. Google Colab provides a free T4 GPU (16GB VRAM) which is ideal.

## Quickest Start: Use the Notebook

Open [`run_eval_colab.ipynb`](run_eval_colab.ipynb) in Colab, fill in the Configuration cell, then Run All. The notebook handles everything: cloning, dependency install, credential entry, KG download, and both evaluation methods.

## Manual Quick Start (copy-paste into a Colab cell)

```python
# 1. Clone the repo
!git clone https://github.com/your-org/hackathon.git
%cd hackathon

# 2. Install dependencies (Neo4j driver + fine-tuning libraries)
!pip install -e ".[eval-model,neo4j]"

# 3. Set credentials — DeepSeek API key + Neo4j password
import os
os.environ["NEO4J_URI"] = "bolt://your-instance.databases.neo4j.io:7687"
os.environ["NEO4J_USER"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "your-password"
os.environ["DEEPSEEK_API_KEY"] = "sk-..."

# (Optional) Write a .env so kg-gen picks it up
with open(".env", "w") as f:
    f.write(f'NEO4J_URI={os.environ["NEO4J_URI"]}\n')
    f.write(f'NEO4J_USER={os.environ["NEO4J_USER"]}\n')
    f.write(f'NEO4J_PASSWORD={os.environ["NEO4J_PASSWORD"]}\n')
    f.write(f'DEEPSEEK_API_KEY={os.environ["DEEPSEEK_API_KEY"]}\n')

# 4. Download the KG from Neo4j
!kg-gen neo4j-download -o generated_KGs/output/knowledge_graph.json

# 5. Method 1 — SFT Data Quality (CPU only, ~30 sec)
!python evaluation/run_eval.py --method 1 --kg generated_KGs/output/knowledge_graph.json

# 6. Method 2 — Fine-tune both models + benchmark (~25-35 min on T4)
!python evaluation/run_eval.py --method 2 --kg generated_KGs/output/knowledge_graph.json \
    --fine-tune-target both --model Qwen/Qwen2.5-1.5B-Instruct
```

> **Note:** The old `model=a`, `model=b`, `model=c` syntax no longer works. Use `--fine-tune-target kg|raw|both` and `--model` for the base model.

## Expected Times (Colab T4 GPU)

| Step | Time |
|------|------|
| Method 1 — SFT data quality | ~30 seconds |
| Method 2 — train both models + benchmark | ~25-35 minutes |

## Model Sizing Guide

| Model | 4-bit VRAM | T4 fit? | Recommended for |
|-------|-----------|---------|-----------------|
| Qwen2.5-0.5B | ~1 GB | ✅ | Fastest iteration, largest relative improvement |
| Qwen2.5-1.5B | ~2 GB | ✅ | **Default** — fast training, noticeable gains |
| Qwen2.5-3B | ~4 GB | ✅ | Better quality, still fast |
| Qwen2.5-7B | ~6 GB | ✅ | Best quality, needs full T4 memory |

Override the model:
```python
!python evaluation/run_eval.py --method 2 --kg generated_KGs/output/knowledge_graph.json \
    --fine-tune-target both --model Qwen/Qwen2.5-7B-Instruct
```

## Getting Your KG into Colab

### Option A: Download from Neo4j (recommended for large KGs)

If your KG is too large for GitHub or direct upload, use Neo4j as an intermediate store:

**Step 1 — On your local machine (after generating the KG):**
```bash
# Upload the KG to Neo4j (uses .env credentials)
make upload dataset=wikipedia

# Or manually:
kg-gen neo4j-upload -o generated_KGs/output_wikipedia/
```

**Step 2 — In Colab, download the KG from Neo4j:**
```python
# 1. Clone + install
!git clone https://github.com/your-org/hackathon.git
%cd hackathon
!pip install -e ".[neo4j]"

# 2. Set both credentials
import os
os.environ["NEO4J_URI"] = "bolt://your-instance.databases.neo4j.io:7687"
os.environ["NEO4J_USER"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "your-password"
os.environ["DEEPSEEK_API_KEY"] = "sk-..."

# 3. Download the graph
!kg-gen neo4j-download -o generated_KGs/output/knowledge_graph.json
```

> **Tip:** Use [Neo4j AuraDB Free Tier](https://neo4j.com/cloud/platform/aura-graph-database/) —
> no installation needed, accessible from anywhere. Perfect for Mac → Colab transfer.

If you prefer a `.env` file instead of `os.environ`:
```python
# Write .env in Colab (never commit secrets to git!)
with open(".env", "w") as f:
    f.write('NEO4J_URI=bolt://your-instance.databases.neo4j.io:7687\n')
    f.write('NEO4J_USER=neo4j\n')
    f.write('NEO4J_PASSWORD=your-password\n')
    f.write('DEEPSEEK_API_KEY=sk-...\n')

# Now download uses .env automatically
!kg-gen neo4j-download -o generated_KGs/output/knowledge_graph.json
```

### Option B: Direct Upload (small KGs only, <100 MB)

If your KG is small enough to upload via browser:
```python
from google.colab import files
uploaded = files.upload()  # upload knowledge_graph.json
!mkdir -p generated_KGs/output
!mv knowledge_graph.json generated_KGs/output/
```
