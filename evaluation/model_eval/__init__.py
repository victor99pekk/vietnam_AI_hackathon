"""Model Evaluation — does KG-structured data train a better model than raw text?

  dataset_gen.py  → QA pair generation from KG vs. raw text
  finetune.py     → LoRA fine-tuning with Unsloth (Mac-compatible)
  metrics.py      → A/B/C model comparison benchmark
"""

from evaluation.model_eval.dataset_gen import (
    QADatasetGenerator,
    load_kg,
    load_raw_documents,
)
from evaluation.model_eval.finetune import FineTuner, FineTuneConfig
from evaluation.model_eval.metrics import AblationBenchmark

__all__ = [
    "QADatasetGenerator",
    "load_kg",
    "load_raw_documents",
    "FineTuner",
    "FineTuneConfig",
    "AblationBenchmark",
]
