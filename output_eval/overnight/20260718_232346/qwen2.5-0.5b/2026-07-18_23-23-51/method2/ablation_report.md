# KG → LLM Ablation Study Report

**Test samples**: 200
**Base model**: Qwen/Qwen2.5-0.5B-Instruct

---

## Model Performance

### A_base

| Metric | Score |
|--------|-------|
| factual_accuracy | 0.0472 |
| single_hop_accuracy | 0.0575 |
| true_false_accuracy | 0.0000 |
| hallucination_rate | 1.0000 |
| consistency_score | 0.8204 |
| avg_response_length | 90.4000 |

### B_kg

| Metric | Score |
|--------|-------|
| factual_accuracy | 0.4203 |
| single_hop_accuracy | 0.4150 |
| true_false_accuracy | 0.4444 |
| hallucination_rate | 0.7250 |
| consistency_score | 0.2909 |
| avg_response_length | 6.3000 |

### C_raw

| Metric | Score |
|--------|-------|
| factual_accuracy | 0.1314 |
| single_hop_accuracy | 0.1525 |
| true_false_accuracy | 0.0352 |
| hallucination_rate | 0.9750 |
| consistency_score | 0.2564 |
| avg_response_length | 26.9000 |

---

## Comparison Analysis

- ✅ KG-Managed (B) has lowest hallucination — KG curation reduces fabrication.
- 📊 Overall best factual accuracy: B_kg (score: 0.420)

## Winners by Metric

### factual_accuracy
**Best**: B_kg (0.4203)

| Rank | Model | Score |
|------|-------|-------|
| 1 | B_kg | 0.4203 |
| 2 | C_raw | 0.1314 |
| 3 | A_base | 0.0472 |

### single_hop_accuracy
**Best**: B_kg (0.4150)

| Rank | Model | Score |
|------|-------|-------|
| 1 | B_kg | 0.4150 |
| 2 | C_raw | 0.1525 |
| 3 | A_base | 0.0575 |

### hallucination_rate
**Best**: B_kg (0.7250)

| Rank | Model | Score |
|------|-------|-------|
| 1 | B_kg | 0.7250 |
| 2 | C_raw | 0.9750 |
| 3 | A_base | 1.0000 |

### consistency_score
**Best**: A_base (0.8204)

| Rank | Model | Score |
|------|-------|-------|
| 1 | A_base | 0.8204 |
| 2 | B_kg | 0.2909 |
| 3 | C_raw | 0.2564 |
