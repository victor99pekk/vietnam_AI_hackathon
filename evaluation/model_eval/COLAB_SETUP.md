# Running model_eval on Google Colab

model_eval requires a CUDA GPU — it will NOT run on Mac. Google Colab provides a free T4 GPU (16GB VRAM) which is ideal.

## Quick Start (copy-paste into a Colab cell)

```python
# 1. Clone the repo
!git clone https://github.com/your-org/hackathon.git
%cd hackathon

# 2. Install model_eval dependencies
!pip install -e ".[eval-model]"

# 3. Add your DeepSeek API key (for SFT generation in data_eval)
import os
os.environ["DEEPSEEK_API_KEY"] = "sk-..."

# 4. Run data_eval first (generates QA datasets from your KG)
!python evaluation/run_eval.py --method 1 --kg generated_KGs/output/knowledge_graph.json

# 5. Run model_eval — train Model B (KG-managed)
!python evaluation/run_eval.py --method 2 --kg generated_KGs/output/knowledge_graph.json model=b

# 6. Run model_eval — train Model C (raw-text)
!python evaluation/run_eval.py --method 2 --kg generated_KGs/output/knowledge_graph.json model=c

# 7. Benchmark all three models (A vs B vs C)
#    (after both B and C are trained)
!python evaluation/run_eval.py --method 2 --kg generated_KGs/output/knowledge_graph.json model=a
```

## Expected Times (Colab T4 GPU)

| Step | Time |
|------|------|
| data_eval (Step 1) | ~30 seconds |
| model_eval — train one model (200 steps) | ~10-15 minutes |
| model_eval — benchmark | ~2-5 minutes |

## Model Sizing Guide

| Model | 4-bit VRAM | T4 fit? | Recommended for |
|-------|-----------|---------|-----------------|
| Qwen2.5-0.5B | ~1 GB | ✅ | Fastest iteration, largest relative improvement |
| Qwen2.5-1.5B | ~2 GB | ✅ | **Default** — fast training, noticeable gains |
| Qwen2.5-3B | ~4 GB | ✅ | Better quality, still fast |
| Qwen2.5-7B | ~6 GB | ✅ | Best quality, needs full T4 memory |

Override the model:
```python
!python evaluation/run_eval.py --method 2 model=b --model Qwen/Qwen2.5-7B-Instruct
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
# 1. Clone the repo
!git clone https://github.com/your-org/hackathon.git
%cd hackathon

# 2. Install dependencies including Neo4j driver
!pip install -e ".[neo4j]"

# 3. Set Neo4j credentials (use a Neo4j AuraDB free tier or your own instance)
import os
os.environ["NEO4J_URI"] = "bolt://your-instance.databases.neo4j.io:7687"
os.environ["NEO4J_USER"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "your-password"

# 4. Download the graph
!kg-gen neo4j-download -o generated_KGs/output/knowledge_graph.json
```

> **Tip:** Use [Neo4j AuraDB Free Tier](https://neo4j.com/cloud/platform/aura-graph-database/) —
> no installation needed, accessible from anywhere. Perfect for Mac → Colab transfer.

If you use a `.env` file instead of os.environ:
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
