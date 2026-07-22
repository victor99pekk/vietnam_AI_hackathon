"""Model Evaluation — does KG-structured data train a better model than raw text?

  dataset_gen.py  → QA pair generation from KG vs. raw text
  finetune.py     → LoRA fine-tuning with Unsloth (Mac-compatible)
  metrics.py      → A/B/C model comparison benchmark
"""

from kg_generator.evaluate.model_eval.dataset_gen import (
    QADatasetGenerator,
    load_kg,
    load_raw_documents,
)
from kg_generator.evaluate.model_eval.metrics import AblationBenchmark

# Fine-tuning support requires torch + GPU — only export if available
try:
    from kg_generator.evaluate.model_eval.finetune import FineTuner, FineTuneConfig, _TORCH_AVAILABLE
except ImportError:
    _TORCH_AVAILABLE = False
    FineTuner = None        # type: ignore
    FineTuneConfig = None   # type: ignore

__all__ = [
    "QADatasetGenerator",
    "load_kg",
    "load_raw_documents",
    "FineTuner",
    "FineTuneConfig",
    "AblationBenchmark",
    "_TORCH_AVAILABLE",
]

# Fine-tuning and benchmarking are imported by run_method2 only. Keeping those
# optional dependencies lazy lets dataset generation and GraphGen sampling run
# in the lightweight local environment.


def __getattr__(name):
    if name in {"FineTuner", "FineTuneConfig"}:
        from kg_generator.evaluate.model_eval.finetune import FineTuner, FineTuneConfig

        return {"FineTuner": FineTuner, "FineTuneConfig": FineTuneConfig}[name]
    if name == "AblationBenchmark":
        from kg_generator.evaluate.model_eval.metrics import AblationBenchmark

        return AblationBenchmark
    raise AttributeError(name)
