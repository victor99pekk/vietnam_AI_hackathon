"""KG Evaluation Pipeline — standalone package.

Two complementary evaluation types:

  data_eval  — Is my KG generating high-quality SFT training data?
    → structural audit + SFT generation + deepeval quality scoring

  model_eval — Does KG-structured data produce a better model?
    → dataset generation + LoRA fine-tuning + A/B/C model comparison

Usage:
    python evaluation/run_eval.py --method 1 --kg generated_KGs/output_debug/knowledge_graph.json
    python evaluation/run_eval.py --method 2 --kg generated_KGs/output_debug/knowledge_graph.json
    python evaluation/run_eval.py --method all --kg generated_KGs/output_debug/knowledge_graph.json
"""

from evaluation.data_eval.metrics import QualityEvaluator
from evaluation.data_eval import (
    StructuralAuditor,
    load_kg_for_audit,
    SFTGenerator,
    TemplateSFTGenerator,
    SFTEvaluator,
)
__all__ = [
    # data_eval
    "QualityEvaluator",
    "StructuralAuditor",
    "load_kg_for_audit",
    "SFTGenerator",
    "TemplateSFTGenerator",
    "SFTEvaluator",
    "QADatasetGenerator",
    "load_kg",
    "load_raw_documents",
    "FineTuner",
    "FineTuneConfig",
    "AblationBenchmark",
]

# Model-evaluation dependencies (notably PyTorch) are optional and intentionally
# not imported at package initialization. Import them from evaluation.model_eval
# when running Method 2.


def __getattr__(name):
    if name in {
        "QADatasetGenerator",
        "load_kg",
        "load_raw_documents",
        "FineTuner",
        "FineTuneConfig",
        "AblationBenchmark",
    }:
        import evaluation.model_eval as model_eval

        return getattr(model_eval, name)
    raise AttributeError(name)
