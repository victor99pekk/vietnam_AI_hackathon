"""Stage 5: Quality evaluation — data quality, model ablation, and graph assessment.

Two complementary evaluation types:

  data_eval  — Is my KG generating high-quality SFT training data?
    → structural audit + SFT generation + deepeval quality scoring
    → fact coverage: how many source-document facts are in the KG?

  model_eval — Does KG-structured data produce a better model?
    → dataset generation + LoRA fine-tuning + A/B/C model comparison

Usage:
    python -m kg_generator.evaluate.run_eval --method 1 --kg data/samples/sample_kg.json
    python -m kg_generator.evaluate.run_eval --method 2 --kg data/samples/sample_kg.json
    python -m kg_generator.evaluate.run_eval --method all --kg data/samples/sample_kg.json
"""

from kg_generator.evaluate.data_eval.metrics import QualityEvaluator
from kg_generator.evaluate.data_eval.structural_audit import (
    StructuralAuditor,
    load_kg_for_audit,
)
from kg_generator.evaluate.data_eval.sft_generator import SFTGenerator
from kg_generator.evaluate.data_eval.sft_evaluator import SFTEvaluator
from kg_generator.evaluate.data_eval.coverage import (
    FactExtractor,
    CoverageEvaluator,
    compute_kg_coverage,
)

# dataset_gen (no torch needed) — always import
from kg_generator.evaluate.model_eval.dataset_gen import (
    QADatasetGenerator,
    load_kg,
    load_raw_documents,
)

# Fine-tuning & benchmarking require torch+GPU — only export if available
try:
    from kg_generator.evaluate.model_eval.finetune import FineTuner, FineTuneConfig, _TORCH_AVAILABLE
    from kg_generator.evaluate.model_eval.metrics import AblationBenchmark
    if not _TORCH_AVAILABLE:
        raise ImportError("torch not installed")
except ImportError:
    _TORCH_AVAILABLE = False
    FineTuner = None               # type: ignore
    FineTuneConfig = None          # type: ignore
    AblationBenchmark = None       # type: ignore

__all__ = [
    # data_eval
    "QualityEvaluator",
    "StructuralAuditor",
    "load_kg_for_audit",
    "SFTGenerator",
    "SFTEvaluator",
    "FactExtractor",
    "CoverageEvaluator",
    "compute_kg_coverage",
    # model_eval
    "QADatasetGenerator",
    "load_kg",
    "load_raw_documents",
    "FineTuner",
    "FineTuneConfig",
    "AblationBenchmark",
]

__all__ = [
    "QualityEvaluator",
    "StructuralAuditor",
    "load_kg_for_audit",
    "SFTGenerator",
    "SFTEvaluator",
    "QADatasetGenerator",
    "load_kg",
    "load_raw_documents",
    "FineTuner",
    "FineTuneConfig",
    "AblationBenchmark",
]


def __getattr__(name):
    """Load optional model-training dependencies only when requested."""
    if name in {"FineTuner", "FineTuneConfig", "AblationBenchmark"}:
        import kg_generator.evaluate.model_eval as model_eval

        return getattr(model_eval, name)
    raise AttributeError(name)
