# KG → LLM Ablation Study Report

**Test samples**: 50
**Base model**: Qwen2.5-0.5B-Instruct

---

## Model Performance

### A_base

| Metric | Score |
|--------|-------|
| factual_accuracy | 0.0266 |
| multi_hop_accuracy | 0.0266 |
| hallucination_rate | 0.9400 |
| consistency_score | 0.5610 |
| avg_response_length | 81.9000 |

### B_kg

| Metric | Score |
|--------|-------|
| factual_accuracy | 0.1814 |
| multi_hop_accuracy | 0.1814 |
| hallucination_rate | 0.0000 |
| consistency_score | 0.8298 |
| avg_response_length | 20.9000 |

### C_raw

| Metric | Score |
|--------|-------|
| factual_accuracy | 0.0420 |
| multi_hop_accuracy | 0.0420 |
| hallucination_rate | 0.3600 |
| consistency_score | 0.7616 |
| avg_response_length | 24.2000 |

---

## Comparison Analysis

- ✅ KG-Managed (B) wins on multi-hop reasoning — the KG structure successfully teaches logical chaining.
- ✅ KG-Managed (B) has lowest hallucination — KG curation reduces fabrication.
- 📊 Overall best factual accuracy: B_kg (score: 0.181)

## Winners by Metric

### factual_accuracy
**Best**: B_kg (0.1814)

| Rank | Model | Score |
|------|-------|-------|
| 1 | B_kg | 0.1814 |
| 2 | C_raw | 0.0420 |
| 3 | A_base | 0.0266 |

### multi_hop_accuracy
**Best**: B_kg (0.1814)

| Rank | Model | Score |
|------|-------|-------|
| 1 | B_kg | 0.1814 |
| 2 | C_raw | 0.0420 |
| 3 | A_base | 0.0266 |

### hallucination_rate
**Best**: B_kg (0.0000)

| Rank | Model | Score |
|------|-------|-------|
| 1 | B_kg | 0.0000 |
| 2 | C_raw | 0.3600 |
| 3 | A_base | 0.9400 |

### consistency_score
**Best**: B_kg (0.8298)

| Rank | Model | Score |
|------|-------|-------|
| 1 | B_kg | 0.8298 |
| 2 | C_raw | 0.7616 |
| 3 | A_base | 0.5610 |
