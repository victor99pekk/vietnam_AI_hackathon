#!/usr/bin/env python3
"""
KG → LLM Evaluation Pipeline Orchestrator

Runs evaluation modes:
  quality  — SFT Data Quality Assessment
    • Structural audit (graph health)
    • SFT pair generation (LLM-powered)
    • SFT quality evaluation (deepeval or heuristic)

  ablation — Fine-Tuning Ablation Study
    • Dataset generation (KG-structured QA vs. raw-text QA)
    • LoRA fine-tuning (Unsloth or transformers+PEFT)
    • Ablation benchmark (A/B/C model comparison)

  graphgen — Paper-inspired subgraph organization and multi-hop QA synthesis

Usage:
    # Quality check only (fast)
    python scripts/run_eval.py --method quality --kg output/knowledge_graph.json

    # Ablation study only (requires fine-tuning)
    python scripts/run_eval.py --method ablation --kg output/knowledge_graph.json

    # Everything
    python scripts/run_eval.py --method full --kg output/knowledge_graph.json

    # With custom config
    python scripts/run_eval.py --method full --kg output/knowledge_graph.json -c configs/eval_override.yaml
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

# Ensure the project root is on the path so `evaluation` is importable
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from kg_generator.evaluate.data_eval.structural_audit import StructuralAuditor, load_kg_for_audit
from kg_generator.evaluate.data_eval.sft_generator import SFTGenerator
from kg_generator.evaluate.data_eval.sft_evaluator import SFTEvaluator
from kg_generator.evaluate.model_eval.dataset_gen import (
    QADatasetGenerator,
    balance_jsonl_token_volume,
    load_kg,
    load_raw_documents,
    load_raw_documents_from_kg,
)

# Fine-tuning & benchmarking require torch+GPU — gracefully degrade on CPU
try:
    from kg_generator.evaluate.model_eval.finetune import FineTuner, FineTuneConfig, _TORCH_AVAILABLE
    from kg_generator.evaluate.model_eval.metrics import AblationBenchmark
    if not _TORCH_AVAILABLE:
        raise ImportError("torch not installed")
except ImportError:
    _TORCH_AVAILABLE = False
    FineTuner = None          # type: ignore
    FineTuneConfig = None     # type: ignore
    AblationBenchmark = None  # type: ignore

# GPU fine-tuning (requires CUDA)
try:
    from kg_generator.evaluate.model_eval.finetune_gpu import GPUFineTuner, GPUFineTuneConfig, CUDA_AVAILABLE
except (ImportError, RuntimeError):
    GPUFineTuner = None        # type: ignore
    GPUFineTuneConfig = None   # type: ignore
    CUDA_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_eval")


def load_config(config_path: Path | None) -> dict[str, Any]:
    """Load evaluation configuration from YAML, with defaults."""
    default_config_path = (
        Path(__file__).resolve().parent / "eval_config.yaml"
    )

    config: dict[str, Any] = {}
    if default_config_path.exists():
        with open(default_config_path) as f:
            config = yaml.safe_load(f) or {}

    if config_path and config_path.exists():
        with open(config_path) as f:
            override = yaml.safe_load(f) or {}
        # Deep merge (simple two-level)
        for section, values in override.items():
            if section in config and isinstance(config[section], dict):
                config[section].update(values)
            else:
                config[section] = values

    return config


def _auto_plot_training(output_dir: Path) -> None:
    """Auto-generate training metrics plots (best-effort, non-fatal)."""
    try:
        from kg_generator.evaluate.plots.plot_training import plot_model_comparison_training

        logger.info("Auto-generating training metrics plots…")
        plot_model_comparison_training(output_dir)
    except Exception as e:
        logger.warning("Failed to generate training plots: %s", e)


def _auto_plot_method1(output_dir: Path) -> None:
    """Auto-generate plots after Method 1 completes (best-effort, non-fatal)."""
    try:
        from kg_generator.evaluate.plots.plot_structural import plot_structural_audit
        from kg_generator.evaluate.plots.plot_sft_quality import plot_sft_quality
        from kg_generator.evaluate.plots.plot_coverage import plot_coverage

        audit_path = output_dir / "structural_audit.json"
        quality_path = output_dir / "sft_quality_report.json"
        coverage_path = output_dir / "coverage_report.json"

        if audit_path.exists():
            logger.info("Auto-generating structural audit plots…")
            plot_structural_audit(audit_path, output_dir)

        if quality_path.exists():
            logger.info("Auto-generating SFT quality plots…")
            plot_sft_quality(quality_path, output_dir)

        if coverage_path.exists():
            logger.info("Auto-generating coverage plots…")
            plot_coverage(coverage_path, output_dir)

        logger.info("Plots saved → %s", output_dir)
    except ImportError:
        logger.info("matplotlib not installed — skipping plot generation.")
        logger.info("Install with: pip install matplotlib")
    except Exception as exc:
        logger.warning("Plot generation skipped (non-fatal): %s", exc)


# ═══════════════════════════════════════════════════════════════
# Method 1: SFT Data Quality Assessment
# ═══════════════════════════════════════════════════════════════

def run_method1(kg_path: Path, config: dict[str, Any], output_base: Path, neo4j: bool = False) -> dict[str, Any]:
    """Run Method 1: SFT Data Quality Assessment."""
    m1_config = config.get("method1", {})
    output_dir = output_base / "method1"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("METHOD 1: SFT Data Quality Assessment")
    logger.info("=" * 60)

    results: dict[str, Any] = {}

    # ── Neo4j-native path ──────────────────────────────────
    if neo4j:
        import os
        from dotenv import load_dotenv
        load_dotenv()
        from neo4j import GraphDatabase
        from kg_generator.evaluate.data_eval.neo4j_auditor import Neo4jStructuralAuditor

        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "")

        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session() as session:
            logger.info("Connected to Neo4j at %s", uri)

            # Step 1.1: Structural Audit (direct Neo4j queries — no RAM limit)
            logger.info("\n--- Step 1.1: Structural Audit (Neo4j-native) ---")
            audit_config = m1_config.get("structural_audit", {})
            auditor = Neo4jStructuralAuditor(
                session,
                entity_dedup_threshold=audit_config.get("entity_dedup_threshold", 0.85),
            )
            audit_report = auditor.audit()
        driver.close()

        results["structural_audit"] = audit_report

        audit_path = output_dir / "structural_audit.json"
        with open(audit_path, "w") as f:
            json.dump(audit_report, f, indent=2)
        logger.info("Structural audit saved → %s", audit_path)
        logger.info("Overall health score: %.1f/100 — %s",
                     audit_report["overall_health_score"], audit_report["verdict"])

        # Steps 1.2–1.4 require full graph in RAM — skipped in Neo4j mode
        logger.info("Steps 1.2–1.4 (SFT generation, evaluation, coverage) skipped "
                     "in --neo4j mode (require full graph in RAM).")

        # Save combined results
        combined_path = output_dir / "method1_results.json"
        with open(combined_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Method 1 (Neo4j audit) complete → %s", combined_path)

        # ── Auto-generate plots ──────────────────────────
        _auto_plot_method1(output_dir)
        return results

    # ── Original JSON path (unchanged) ────────────────────

    # Load KG
    graph, entities, triples = load_kg_for_audit(kg_path)
    logger.info("Loaded KG: %d nodes, %d edges, %d triples",
                 graph.number_of_nodes(), graph.number_of_edges(), len(triples))

    # Step 1.1: Structural Audit
    logger.info("\n--- Step 1.1: Structural Audit ---")
    audit_config = m1_config.get("structural_audit", {})
    ontology_path = audit_config.get("ontology_path")
    if ontology_path:
        ontology_path = Path(ontology_path)

    auditor = StructuralAuditor(
        ontology_path=ontology_path,
        entity_dedup_threshold=audit_config.get("entity_dedup_threshold", 0.85),
    )
    audit_report = auditor.audit(graph, entities, triples)
    results["structural_audit"] = audit_report

    audit_path = output_dir / "structural_audit.json"
    with open(audit_path, "w") as f:
        json.dump(audit_report, f, indent=2)
    logger.info("Structural audit saved → %s", audit_path)
    logger.info("Overall health score: %.1f/100 — %s",
                 audit_report["overall_health_score"], audit_report["verdict"])

    # Step 1.2: SFT Data Generation
    logger.info("\n--- Step 1.2: SFT Data Generation ---")
    gen_config = m1_config.get("sft_generation", {})

    logger.info("Using LLM-powered SFT generation (%s)", gen_config.get("llm_model"))
    generator = SFTGenerator(
        model=gen_config.get("llm_model", "deepseek-chat"),
        provider=gen_config.get("llm_provider", "deepseek"),
        num_samples=gen_config.get("num_samples", 50),
        hop_distribution=tuple(gen_config.get("hop_distribution", [0.3, 0.4, 0.3])),
        temperature=gen_config.get("temperature", 0.7),
        max_tokens=gen_config.get("max_tokens", 512),
        seed=config.get("common", {}).get("seed", 42),
    )
    sft_path = generator.generate(graph, triples, output_dir)
    results["sft_generation"] = {"path": str(sft_path), "method": "llm"}
    sft_file = sft_path

    # Step 1.3: SFT Quality Evaluation
    logger.info("\n--- Step 1.3: SFT Quality Evaluation ---")
    eval_config = m1_config.get("sft_evaluation", {})

    evaluator = SFTEvaluator(
        faithfulness_threshold=eval_config.get("faithfulness_threshold", 0.7),
        relevancy_threshold=eval_config.get("relevancy_threshold", 0.7),
        judge_model=eval_config.get("judge_model", "deepseek-chat"),
        min_diversity_score=eval_config.get("min_diversity_score", 0.4),
    )
    quality_report = evaluator.evaluate(sft_file)
    results["sft_evaluation"] = quality_report

    quality_path = output_dir / "sft_quality_report.json"
    with open(quality_path, "w") as f:
        json.dump(quality_report, f, indent=2)
    logger.info("SFT quality report saved → %s", quality_path)
    logger.info("Overall SFT quality: %.3f — %s",
                 quality_report.get("overall_score", 0), quality_report.get("verdict", ""))

    # Step 1.4: Source Document Fact Coverage
    logger.info("\n--- Step 1.4: Source Fact Coverage ---")
    coverage_config = m1_config.get("coverage", {})

    if coverage_config.get("enabled", True):
        from kg_generator.evaluate.data_eval.coverage import compute_kg_coverage

        # Resolve source document paths from config
        doc_paths_config = coverage_config.get("document_paths", ["data/"])
        document_paths: list[Path] = []
        for p in doc_paths_config:
            path = Path(p)
            if path.exists():
                document_paths.append(path)
            else:
                logger.warning("Coverage: document path not found: %s", p)

        # Fallback: try to find documents from triples' source_text fields
        if not document_paths:
            source_files = set()
            for t in triples:
                if len(t) > 3 and t[3]:
                    source_files.add(t[3])
            document_paths = [Path(f) for f in source_files if Path(f).exists()]

        # Last resort: use data/ directory
        if not document_paths:
            data_dir = Path("data")
            if data_dir.exists():
                document_paths = [data_dir]
                logger.info("Coverage: falling back to %s", data_dir)

        if document_paths:
            coverage_report = compute_kg_coverage(
                kg_path=kg_path,
                document_paths=document_paths,
                extraction_mode=coverage_config.get("extraction_mode", "heuristic"),
                fact_sample_size=coverage_config.get("fact_sample_size", 30),
                seed=config.get("common", {}).get("seed", 42),
            )
            results["coverage"] = coverage_report

            coverage_path = output_dir / "coverage_report.json"
            with open(coverage_path, "w") as f:
                json.dump(coverage_report, f, indent=2)
            logger.info("Coverage report saved → %s", coverage_path)
            logger.info("Fact coverage: %.1f/100 — %s",
                         coverage_report.get("health_score", 0),
                         coverage_report.get("verdict", ""))
        else:
            logger.warning("No source documents found — skipping coverage evaluation")
            results["coverage"] = {"skipped": True, "reason": "No source documents found"}
    else:
        logger.info("Coverage evaluation disabled in config.")
        results["coverage"] = {"skipped": True, "reason": "Disabled in config"}

    # Save combined results
    combined_path = output_dir / "method1_results.json"
    with open(combined_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Method 1 complete → %s", combined_path)

    # ── Auto-generate plots ──────────────────────────────────
    _auto_plot_method1(output_dir)

    return results


# ═══════════════════════════════════════════════════════════════
# Method 2: Fine-Tuning Ablation Study
# ═══════════════════════════════════════════════════════════════

def run_method2(
    kg_path: Path,
    config: dict[str, Any],
    output_base: Path,
    fine_tune_target: str = "both",
    model_override: str | None = None,
    device: str = "cpu",
    skip_dataset_gen: bool = False,
    skip_benchmark: bool = False,
    skip_finetune: bool = False,
    use_gpu: bool = False,
) -> dict[str, Any]:
    """Run Method 2: Fine-Tuning Ablation Study.
    
    Args:
        use_gpu: If True, use GPU-optimized fine-tuning (7B models, mixed precision).
                Requires CUDA and incompatible with device='cpu'.
    """
    from kg_generator.evaluate.model_eval.finetune import FineTuner, FineTuneConfig
    from kg_generator.evaluate.model_eval.metrics import AblationBenchmark

    m2_config = config.get("method2", {})
    common_config = config.get("common", {})
    
    # GPU mode: use larger models and GPU-optimized defaults
    if use_gpu:
        default_base_model = "Qwen/Qwen2.5-7B-Instruct"
        output_subdir = "method2_gpu"
    else:
        default_base_model = "Qwen/Qwen2.5-0.5B-Instruct"
        output_subdir = "method2"
    
    base_model = model_override or m2_config.get("base_model", default_base_model)
    output_dir = output_base / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("METHOD 2: Fine-Tuning Ablation Study")
    if use_gpu:
        logger.info("MODE: GPU-optimized (7B models, mixed precision)")
    logger.info("=" * 60)

    results: dict[str, Any] = {}
    fine_tune_errors: list[str] = []

    # Step 2.1: Dataset Generation
    if skip_dataset_gen:
        # Resolve existing dataset paths from previous run
        logger.info("\n--- Step 2.1: Dataset Generation (SKIPPED, using existing) ---")
        kg_train_path = output_dir / "test_training_data" / "kg_qa_train.jsonl"
        kg_test_path = output_dir / "test_training_data" / "kg_qa_test.jsonl"
        raw_train_path = output_dir / "test_training_data" / "raw_qa_train.jsonl"
        raw_test_path = output_dir / "test_training_data" / "raw_qa_test.jsonl"

        if not kg_test_path.exists():
            logger.error(
                "Existing datasets not found in %s. "
                "Run without --skip-dataset-gen first.", output_dir
            )
            raise FileNotFoundError("No existing Method 2 datasets found")

        test_data_path = kg_test_path
        results["dataset_gen"] = {
            "kg_train": str(kg_train_path),
            "kg_test": str(kg_test_path),
            "raw_train": str(raw_train_path) if raw_train_path.exists() else None,
            "raw_test": str(raw_test_path) if raw_test_path.exists() else None,
        }
        logger.info("Using existing datasets from %s", output_dir)
    else:
        logger.info("\n--- Step 2.1: Dataset Generation ---")
        gen_config = m2_config.get("dataset_gen", {})

        graph, entities, triples = load_kg(kg_path)
        logger.info("Loaded KG: %d entities, %d triples",
                     len(entities), len(triples))
        if not triples:
            raise ValueError(
                "KG contains no domain triples after excluding NEXT/PART_OF/MENTIONS. "
                "Build it with relation extraction before running Method 2."
            )

        # Generate KG-Managed QA pairs (Model B)
        qa_gen = QADatasetGenerator(
            language=gen_config.get("language", "en"),
            seed=common_config.get("seed", 42),
            max_hops=gen_config.get("max_hops", 3),
            test_split=gen_config.get("test_split", 0.2),
            max_triples=gen_config.get("max_triples", 2000),
            max_paths_per_start=gen_config.get("max_paths_per_start", 20),
            max_pairs=gen_config.get("max_pairs", 10000),
        )

        kg_train_path, kg_test_path = qa_gen.generate_from_kg(
            graph, entities, triples, output_dir,
        )
        results["dataset_gen"] = {
            "kg_train": str(kg_train_path),
            "kg_test": str(kg_test_path),
        }

        # Generate Raw-Text QA pairs (Model C) from source documents
        # Prefer cleaned source chunks embedded in this exact KG export. This
        # prevents Model C from silently reading unrelated files under data/.
        raw_docs = load_raw_documents_from_kg(kg_path)
        raw_paths = [Path(path) for path in gen_config.get("raw_document_paths", [])]
        if not raw_docs and raw_paths:
            raw_docs = load_raw_documents(raw_paths)

        if raw_docs:
            raw_train_path, raw_test_path = qa_gen.generate_from_raw_text(
                raw_docs, output_dir,
            )
            results["dataset_gen"]["raw_train"] = str(raw_train_path)
            results["dataset_gen"]["raw_test"] = str(raw_test_path)

            if gen_config.get("match_token_volume", True):
                balance = balance_jsonl_token_volume(kg_train_path, raw_train_path)
                results["dataset_gen"]["training_balance"] = balance
                logger.info("Balanced KG/raw training token volume: %s", balance)

            # Use the KG test set for both models (fair comparison)
            test_data_path = kg_test_path
        else:
            logger.warning("No source documents found — cannot generate raw-text dataset")
            results["dataset_gen"]["raw_train"] = None
            results["dataset_gen"]["raw_test"] = None
            test_data_path = kg_test_path

        logger.info("Dataset generation complete")

    # Step 2.2: Fine-Tuning
    if skip_finetune:
        logger.info("\n--- Step 2.2: Fine-Tuning (SKIPPED) ---")
        # Still resolve existing adapters for potential benchmark
        kg_adapter_path = output_dir / "model_b_kg" if (output_dir / "model_b_kg").exists() else None
        raw_adapter_path = output_dir / "model_c_raw" if (output_dir / "model_c_raw").exists() else None
        results.setdefault("finetune", {})
        if kg_adapter_path:
            results["finetune"]["kg_adapter"] = str(kg_adapter_path)
        if raw_adapter_path:
            results["finetune"]["raw_adapter"] = str(raw_adapter_path)
    else:
        logger.info("\n--- Step 2.2: Fine-Tuning ---")
        
        # GPU mode requires torch but will be handled by GPUFineTuner
        if use_gpu:
            if not CUDA_AVAILABLE or GPUFineTuner is None:
                logger.error(
                    "GPU fine-tuning requires CUDA. Install with: "
                    "uv pip install -e '.[eval-model]' and ensure CUDA is available"
                )
                raise RuntimeError("CUDA not available or GPU fine-tuning not available")
            logger.info("Using GPU-optimized fine-tuning")
        else:
            if not _TORCH_AVAILABLE:
                logger.error(
                    "Fine-tuning requires torch. Install with: "
                    "uv pip install -e '.[eval-model]'"
                )
                logger.error("Or re-run with --skip-finetune to skip this step.")
                raise RuntimeError("torch not installed — cannot fine-tune")

        lora_config = m2_config.get("lora", {})
        training_config = m2_config.get("training", {})

        logger.info("Base model: %s", base_model)
        logger.info("Fine-tune target: %s", fine_tune_target)

        # GPU-optimized defaults
        if use_gpu:
            from kg_generator.evaluate.model_eval.finetune_gpu import GPUFineTuneConfig
            ft_config = GPUFineTuneConfig(
                base_model=base_model,
                output_dir=output_dir,
                lora_r=lora_config.get("r", 32),  # Larger rank for bigger models
                lora_alpha=lora_config.get("alpha", 64),
                lora_dropout=lora_config.get("dropout", 0.05),
                max_seq_length=training_config.get("max_seq_length", 2048),
                per_device_train_batch_size=training_config.get("per_device_train_batch_size", 16),
                gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 1),
                learning_rate=training_config.get("learning_rate", 1.0e-4),
                max_steps=training_config.get("max_steps", -1),
                num_train_epochs=training_config.get("num_train_epochs", 3.0),
                warmup_steps=training_config.get("warmup_steps", 0),
                warmup_ratio=training_config.get("warmup_ratio", 0.05),
                logging_steps=training_config.get("logging_steps", 10),
                save_steps=training_config.get("save_steps", 100),
                weight_decay=training_config.get("weight_decay", 0.01),
                seed=common_config.get("seed", 42),
            )
            finetuner = GPUFineTuner(ft_config)
        else:
            # CPU-optimized defaults (original)
            ft_config = FineTuneConfig(
                base_model=base_model,
                output_dir=output_dir,
                device=device,
                lora_r=lora_config.get("r", 16),
                lora_alpha=lora_config.get("alpha", 32),
                lora_dropout=lora_config.get("dropout", 0.05),
                max_seq_length=training_config.get("max_seq_length", 2048),
                per_device_train_batch_size=training_config.get("per_device_train_batch_size", 2),
                gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 4),
                learning_rate=training_config.get("learning_rate", 2.0e-4),
                max_steps=training_config.get("max_steps", -1),
                num_train_epochs=training_config.get("num_train_epochs", 3.0),
                warmup_steps=training_config.get("warmup_steps", 0),
                warmup_ratio=training_config.get("warmup_ratio", 0.05),
                logging_steps=training_config.get("logging_steps", 10),
                save_steps=training_config.get("save_steps", 100),
                weight_decay=training_config.get("weight_decay", 0.01),
                seed=common_config.get("seed", 42),
            )
            finetuner = FineTuner(ft_config)
        
        results.setdefault("finetune", {})

        # ── Fine-tune Model B (KG-Managed) ──
        kg_adapter_path = None
        if fine_tune_target in ("kg", "both"):
            logger.info("\n>>> Fine-tuning Model B (KG-Managed)")
            try:
                kg_adapter_path = finetuner.fine_tune(
                    train_data_path=kg_train_path,
                    adapter_name="model_b_kg",
                    # Keep benchmark questions completely untouched by training
                    # and checkpoint selection.
                    eval_data_path=None,
                )
                results["finetune"]["kg_adapter"] = str(kg_adapter_path)
                logger.info("Model B (KG-Managed) fine-tuned → %s", kg_adapter_path)
            except Exception as e:
                logger.error("Model B fine-tuning failed: %s", e)
                results["finetune"]["kg_adapter_error"] = str(e)
                fine_tune_errors.append(f"Model B: {e}")
        else:
            # Look for existing adapter
            existing_kg = output_dir / "model_b_kg"
            if existing_kg.exists():
                kg_adapter_path = existing_kg
                results["finetune"]["kg_adapter"] = str(kg_adapter_path)
                logger.info("Using existing Model B adapter → %s", kg_adapter_path)
            else:
                logger.info("Skipping Model B fine-tuning (--fine-tune-target=%s)", fine_tune_target)

        # ── Fine-tune Model C (Raw-Text) ──
        raw_adapter_path = None
        if fine_tune_target in ("raw", "both"):
            raw_train = results.get("dataset_gen", {}).get("raw_train")
            if raw_train:
                logger.info("\n>>> Fine-tuning Model C (Raw-Text)")
                try:
                    raw_adapter_path = finetuner.fine_tune(
                        train_data_path=Path(raw_train),
                        adapter_name="model_c_raw",
                        eval_data_path=None,
                    )
                    results["finetune"]["raw_adapter"] = str(raw_adapter_path)
                    logger.info("Model C (Raw-Text) fine-tuned → %s", raw_adapter_path)
                except Exception as e:
                    logger.error("Model C fine-tuning failed: %s", e)
                    results["finetune"]["raw_adapter_error"] = str(e)
                    fine_tune_errors.append(f"Model C: {e}")
            else:
                logger.warning("No raw training data available for Model C")
        else:
            existing_raw = output_dir / "model_c_raw"
            if existing_raw.exists():
                raw_adapter_path = existing_raw
                results["finetune"]["raw_adapter"] = str(raw_adapter_path)
                logger.info("Using existing Model C adapter → %s", raw_adapter_path)
            else:
                logger.info("Skipping Model C fine-tuning (--fine-tune-target=%s)", fine_tune_target)

    # Step 2.3: Ablation Benchmark
    if skip_benchmark:
        logger.info("\n--- Step 2.3: Ablation Benchmark (SKIPPED) ---")
    else:
        logger.info("\n--- Step 2.3: Ablation Benchmark ---")
        if not _TORCH_AVAILABLE:
            logger.error(
                "Benchmarking requires torch. Install with: "
                "uv pip install -e '.[eval-model]'"
            )
            logger.error("Or re-run with --skip-benchmark to skip this step.")
            raise RuntimeError("torch not installed — cannot benchmark")

        benchmark_config = m2_config.get("benchmark", {})

        benchmark = AblationBenchmark(
            base_model=base_model,
            max_test_samples=benchmark_config.get("max_test_samples", 200),
            seed=common_config.get("seed", 42),
        )

        # Resolve adapter paths (use existing if fine-tuning was skipped)
        _kg_path = kg_adapter_path or (output_dir / "model_b_kg" if (output_dir / "model_b_kg").exists() else None)
        _raw_path = raw_adapter_path or (output_dir / "model_c_raw" if (output_dir / "model_c_raw").exists() else None)

        ablation_results = benchmark.evaluate(
            test_data_path=test_data_path,
            kg_adapter_path=_kg_path,
            raw_adapter_path=_raw_path,
            output_dir=output_dir,
        )
        results["ablation"] = ablation_results

    # Save combined results
    combined_path = output_dir / "method2_results.json"
    with open(combined_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Method 2 complete → %s", combined_path)

    # Auto-generate training metrics plots (if fine-tuning was run)
    if not skip_finetune:
        _auto_plot_training(output_dir)

    if fine_tune_errors:
        raise RuntimeError("; ".join(fine_tune_errors))

    return results


# ═══════════════════════════════════════════════════════════════
# GraphGen-style subgraph organization and QA generation
# ═══════════════════════════════════════════════════════════════

def run_graphgen(
    kg_path: Path,
    config: dict[str, Any],
    output_base: Path,
    *,
    sample_only: bool = False,
) -> dict[str, Any]:
    """Run the isolated GraphGen-style experiment.

    This does not replace Method 1 or Method 2. It produces stable intermediate
    artifacts so subgraph selection can be inspected before any QA is used for
    training.
    """

    graphgen_config = config.get("graphgen", {})
    output_dir = output_base / "graphgen"
    output_dir.mkdir(parents=True, exist_ok=True)

    nodes, edges = load_graphgen_kg(kg_path)
    logger.info("Loaded GraphGen KG: %d entity nodes, %d knowledge edges", len(nodes), len(edges))
    if not edges:
        raise ValueError(
            "No descriptive knowledge edges found. Expected dictionary triples "
            "with subject, object, and description fields."
        )

    sampler_config = graphgen_config.get("subgraphs", {})
    sampler = GraphGenSubgraphSampler(
        max_depth=sampler_config.get("max_depth", 2),
        max_premise_tokens=sampler_config.get("max_premise_tokens", 256),
        max_extra_edges=sampler_config.get("max_extra_edges", 5),
        edge_sampling=sampler_config.get("edge_sampling", "max_loss"),
        bidirectional=sampler_config.get("bidirectional", True),
        seed=config.get("common", {}).get("seed", 42),
    )
    sampling = sampler.sample(
        nodes,
        edges,
        max_subgraphs=sampler_config.get("max_subgraphs", 50),
    )
    subgraphs_path, sampling_audit_path = sampling.write(output_dir)
    results: dict[str, Any] = {
        "method": "graphgen_style",
        "paper_faithfulness": {
            "implemented": [
                "edge-seeded k-hop expansion",
                "bidirectional traversal",
                "premise token budget",
                "max/min loss or deterministic fallback selection",
                "multi-hop QA generation",
            ],
            "not_implemented": ["trainee-model comprehension assessment"],
        },
        "subgraphs": {
            "path": str(subgraphs_path),
            "count": len(sampling.subgraphs),
            "sampling_audit": str(sampling_audit_path),
            "selection_basis_counts": _count_values(
                sampling.subgraphs, "selection_basis"
            ),
        },
    }

    qa_config = graphgen_config.get("qa_generation", {})
    if not sample_only:
        generator = GraphGenQAGenerator(
            model=qa_config.get("model", "deepseek-v4-pro"),
            provider=qa_config.get("provider", "deepseek"),
            language=qa_config.get("language", "English"),
            temperature=qa_config.get("temperature", 0.2),
            max_tokens=qa_config.get("max_tokens", 512),
        )
        qa_path, qa_audit_path = generator.generate(
            sampling.subgraphs,
            output_dir,
            max_questions=qa_config.get("max_questions", 20),
        )
        with open(qa_path, encoding="utf-8") as handle:
            qa_count = sum(1 for line in handle if line.strip())
        results["qa"] = {
            "path": str(qa_path),
            "count": qa_count,
            "audit": str(qa_audit_path),
        }
    else:
        results["qa"] = {"skipped": True, "reason": "sample_only"}

    results_path = output_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
    logger.info("GraphGen-style experiment complete → %s", results_path)
    return results


def _count_values(items: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        value = str(item.get(key, "unknown"))
        counts[value] = counts.get(value, 0) + 1
    return counts


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="KG → LLM Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_eval.py --method quality --kg output/knowledge_graph.json
  python scripts/run_eval.py --method ablation --kg output/knowledge_graph.json
  python scripts/run_eval.py --method full --kg output/knowledge_graph.json
  python scripts/run_eval.py --method graphgen --kg output/knowledge_graph.json --sample-only
  python scripts/run_eval.py --method quality --kg output/knowledge_graph.json -c my_config.yaml
        """,
    )
    parser.add_argument(
        "--method", "-m",
        choices=["quality", "ablation", "graphgen", "full"],
        default="full",
        help="Which evaluation mode to run: quality, ablation, graphgen, or full (default: full)",
    )
    parser.add_argument(
        "--kg", "-k",
        type=Path,
        default=Path("output/knowledge_graph.json"),
        help="Path to knowledge_graph.json (default: output/knowledge_graph.json)",
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=None,
        help="Path to custom eval config YAML (optional)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("output_eval"),
        help="Output directory (default: output_eval)",
    )
    parser.add_argument(
        "--fine-tune-target", "-t",
        choices=["kg", "raw", "both"],
        default="both",
        help="Which model(s) to fine-tune: kg (Model B), raw (Model C), or both (default: both)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override the base model (e.g., 'unsloth/Qwen2.5-3B-Instruct-bnb-4bit')",
    )
    parser.add_argument(
        "--skip-finetune",
        action="store_true",
        help="Skip fine-tuning entirely in Method 2 (use existing adapters if available)",
    )
    parser.add_argument(
        "--sample-only",
        action="store_true",
        help="For --method graphgen, write subgraphs/audit without calling the QA LLM",
    )
    parser.add_argument(
        "--skip-dataset-gen",
        action="store_true",
        help="Skip dataset generation in Method 2 (use existing datasets from a previous run)",
    )
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Skip the ablation benchmark in Method 2 (run dataset gen + fine-tuning only)",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="cpu",
        help="Device for fine-tuning: cpu (default), cuda, or auto",
    )
    parser.add_argument(
        "--neo4j",
        action="store_true",
        help="Run structural audit directly against Neo4j (no JSON download, no RAM limit). "
             "Uses NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD from .env.",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU-optimized fine-tuning (requires CUDA). "
             "Runs 7B models with mixed precision and larger batches. "
             "Incompatible with --device cpu.",
    )
    args = parser.parse_args()

    # Validate KG path (skip when using --neo4j)
    if not args.neo4j and not args.kg.exists():
        logger.error("KG file not found: %s", args.kg)
        logger.info("Run the pipeline first: kg-gen run -c configs/pipeline.yaml")
        sys.exit(1)

    # Validate GPU flag
    if args.gpu and not CUDA_AVAILABLE:
        logger.error("--gpu requires CUDA (not available on this system)")
        sys.exit(1)
    if args.gpu and args.device == "cpu":
        logger.error("--gpu is incompatible with --device cpu")
        sys.exit(1)

    # Load config
    config = load_config(args.config)

    # Override output directory
    output_base = args.output
    output_base.mkdir(parents=True, exist_ok=True)

    logger.info("KG Evaluation Pipeline")
    logger.info("  KG:       %s", args.kg)
    logger.info("  Method:   %s", args.method)
    logger.info("  Output:   %s", output_base)

    failed = False

    if args.method in ("quality", "full"):
        try:
            run_method1(args.kg, config, output_base, neo4j=args.neo4j)
        except Exception as e:
            logger.error("Quality evaluation failed: %s", e, exc_info=True)
            failed = True

    if args.method in ("ablation", "full"):
        try:
            run_method2(
                args.kg, config, output_base,
                fine_tune_target=args.fine_tune_target,
                model_override=args.model,
                device=args.device,
                skip_dataset_gen=args.skip_dataset_gen,
                skip_benchmark=args.skip_benchmark,
                skip_finetune=args.skip_finetune,
                use_gpu=args.gpu,
            )
        except Exception as e:
            logger.error("Method 2 failed: %s", e, exc_info=True)
            failed = True

    if args.method == "graphgen":
        try:
            run_graphgen(
                args.kg, config, output_base, sample_only=args.sample_only
            )
        except Exception as e:
            logger.error("GraphGen method failed: %s", e, exc_info=True)
            failed = True

    if failed:
        logger.error("Evaluation pipeline completed with failures. Results in: %s", output_base)
        sys.exit(1)
    logger.info("\n✅ Evaluation pipeline complete. Results in: %s", output_base)


if __name__ == "__main__":
    main()
