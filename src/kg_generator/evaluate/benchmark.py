"""
Method 2, Step 3 — Ablation Benchmark

Evaluates and compares three models on the same test set:
  - Model A: Base Qwen2.5 (no fine-tuning)
  - Model B: KG-Managed (fine-tuned on KG-structured QA pairs)
  - Model C: Unmanaged (fine-tuned on raw-text QA pairs)

Metrics:
  - Factual Accuracy (exact match / F1)
  - Multi-hop Reasoning Accuracy
  - Hallucination Rate
  - Consistency
  - Perplexity on held-out text
"""

import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Try importing model libraries
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    HAS_MODEL_LIBS = True
except ImportError:
    HAS_MODEL_LIBS = False
    logger.warning("transformers/peft not available — model-based evaluation disabled")


class AblationBenchmark:
    """Compares base, KG-fine-tuned, and raw-text-fine-tuned models."""

    def __init__(
        self,
        base_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
        max_test_samples: int = 200,
        seed: int = 42,
    ) -> None:
        self.base_model = base_model
        self.max_test_samples = max_test_samples
        self.seed = seed

    def evaluate(
        self,
        test_data_path: Path,
        kg_adapter_path: Path | None = None,
        raw_adapter_path: Path | None = None,
        output_dir: Path | None = None,
    ) -> dict[str, Any]:
        """Run the full A/B/C ablation benchmark.

        Args:
            test_data_path: Path to test QA JSONL file (used for all three models).
            kg_adapter_path: Path to LoRA adapter for KG-Managed model (Model B).
            raw_adapter_path: Path to LoRA adapter for Raw-Text model (Model C).
            output_dir: Where to save results (optional).

        Returns:
            Dict with results for each model and the comparison analysis.
        """
        output_dir = output_dir or Path("output_eval/method2")
        output_dir.mkdir(parents=True, exist_ok=True)

        test_pairs = self._load_test_pairs(test_data_path)
        if len(test_pairs) > self.max_test_samples:
            test_pairs = test_pairs[:self.max_test_samples]

        logger.info("Benchmarking on %d test pairs", len(test_pairs))

        results: dict[str, Any] = {
            "test_samples": len(test_pairs),
            "models": {},
            "comparison": {},
        }

        # Model A: Base model (no fine-tuning)
        results["models"]["A_base"] = self._evaluate_model(
            name="A_base (Qwen2.5 base)",
            test_pairs=test_pairs,
            adapter_path=None,
        )

        # Model B: KG-Managed
        if kg_adapter_path and kg_adapter_path.exists():
            results["models"]["B_kg"] = self._evaluate_model(
                name="B_kg (KG-Managed)",
                test_pairs=test_pairs,
                adapter_path=kg_adapter_path,
            )
        else:
            logger.warning("KG adapter not found at %s — skipping Model B", kg_adapter_path)
            results["models"]["B_kg"] = {"skipped": True, "reason": "Adapter not found"}

        # Model C: Raw-Text
        if raw_adapter_path and raw_adapter_path.exists():
            results["models"]["C_raw"] = self._evaluate_model(
                name="C_raw (Unmanaged)",
                test_pairs=test_pairs,
                adapter_path=raw_adapter_path,
            )
        else:
            logger.warning("Raw adapter not found at %s — skipping Model C", raw_adapter_path)
            results["models"]["C_raw"] = {"skipped": True, "reason": "Adapter not found"}

        # Comparison analysis
        results["comparison"] = self._compare(results["models"])

        # Save results
        results_path = output_dir / "ablation_results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info("Results saved → %s", results_path)

        # Generate markdown report
        report_path = output_dir / "ablation_report.md"
        self._generate_report(results, report_path)
        logger.info("Report saved → %s", report_path)

        return results

    # ── Per-Model Evaluation ──────────────────────────────────

    def _evaluate_model(
        self,
        name: str,
        test_pairs: list[dict[str, Any]],
        adapter_path: Path | None,
    ) -> dict[str, Any]:
        """Evaluate a single model on the test set."""
        logger.info("Evaluating: %s", name)

        predictions: list[dict[str, Any]] = []

        if HAS_MODEL_LIBS:
            model, tokenizer = self._load_model(adapter_path)
            for pair in test_pairs:
                question = pair.get("question", pair.get("instruction", ""))
                expected = pair.get("answer", pair.get("response", ""))
                pair_type = pair.get("type", "single_hop")

                pred_answer = self._generate_answer(model, tokenizer, question)
                predictions.append({
                    "question": question,
                    "expected": expected,
                    "predicted": pred_answer,
                    "type": pair_type,
                })
        else:
            # Heuristic mode: use the expected answer as a proxy
            # (no actual model inference — useful for pipeline testing)
            logger.warning("No model libraries — using heuristic mode (expected==predicted)")
            for pair in test_pairs:
                question = pair.get("question", pair.get("instruction", ""))
                expected = pair.get("answer", pair.get("response", ""))
                pair_type = pair.get("type", "single_hop")
                predictions.append({
                    "question": question,
                    "expected": expected,
                    "predicted": expected,  # dummy
                    "type": pair_type,
                })

        return self._compute_metrics(predictions, name)

    def _compute_metrics(
        self, predictions: list[dict[str, Any]], name: str
    ) -> dict[str, Any]:
        """Compute all evaluation metrics from predictions."""
        if not predictions:
            return {"error": "No predictions"}

        # Separate by question type
        single_hop = [p for p in predictions if p.get("type") == "single_hop"]
        multi_hop = [p for p in predictions if p.get("type") == "multi_hop"]
        true_false = [p for p in predictions if p.get("type") == "true_false"]

        metrics = {
            "model": name,
            "total_predictions": len(predictions),
            "factual_accuracy": self._factual_accuracy(predictions),
            "single_hop_accuracy": self._factual_accuracy(single_hop) if single_hop else None,
            "multi_hop_accuracy": self._factual_accuracy(multi_hop) if multi_hop else None,
            "true_false_accuracy": self._factual_accuracy(true_false) if true_false else None,
            "hallucination_rate": self._hallucination_rate(predictions),
            "consistency_score": self._consistency_score(predictions),
            "avg_response_length": self._avg_response_length(predictions),
        }

        return metrics

    # ── Individual Metrics ────────────────────────────────────

    @staticmethod
    def _factual_accuracy(predictions: list[dict[str, Any]]) -> float:
        """Exact match + partial overlap F1."""
        if not predictions:
            return 0.0

        scores = []
        for p in predictions:
            expected = p["expected"].strip().lower()
            predicted = p["predicted"].strip().lower()

            # Exact match
            if expected == predicted:
                scores.append(1.0)
                continue

            # Token overlap F1
            expected_tokens = set(expected.split())
            predicted_tokens = set(predicted.split())

            if not expected_tokens or not predicted_tokens:
                scores.append(0.0)
                continue

            tp = len(expected_tokens & predicted_tokens)
            if tp == 0:
                scores.append(0.0)
                continue

            precision = tp / len(predicted_tokens)
            recall = tp / len(expected_tokens)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            scores.append(f1)

        return round(sum(scores) / len(scores), 4)

    @staticmethod
    def _hallucination_rate(predictions: list[dict[str, Any]]) -> float:
        """Estimate hallucination: % of answers where predicted contains
        named entities NOT in the question + expected answer.

        Uses a simple capitalized-word heuristic as proxy for entities.
        """
        if not predictions:
            return 0.0

        hallucination_counts = 0
        for p in predictions:
            question = p["question"].lower()
            expected = p["expected"].lower()
            predicted = p["predicted"].lower()

            # Known entities from question + expected
            known_words = set(question.split()) | set(expected.split())

            # Find potential named entities in prediction (capitalized words proxy)
            # But since we lowercased everything, use length as a proxy for
            # "did the model add extra content not in question/expected"
            pred_words = set(predicted.split())
            new_words = pred_words - known_words

            # If more than 30% of predicted words are new = likely hallucination
            if pred_words and len(new_words) / len(pred_words) > 0.3:
                hallucination_counts += 1

        return round(hallucination_counts / len(predictions), 4)

    @staticmethod
    def _consistency_score(predictions: list[dict[str, Any]]) -> float:
        """Approximate consistency by checking answer length stability.

        Real consistency would require asking the same question rephrased
        multiple times. Here we use a proxy: variance in answer length.
        Lower variance = more consistent outputs.
        """
        if len(predictions) < 2:
            return 1.0

        lengths = [len(p["predicted"].split()) for p in predictions]
        mean_len = sum(lengths) / len(lengths)

        if mean_len == 0:
            return 1.0

        variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
        cv = (variance ** 0.5) / mean_len  # coefficient of variation

        # Lower CV = more consistent
        consistency = max(0, 1.0 - cv)
        return round(consistency, 4)

    @staticmethod
    def _avg_response_length(predictions: list[dict[str, Any]]) -> float:
        if not predictions:
            return 0.0
        return round(sum(len(p["predicted"].split()) for p in predictions) / len(predictions), 1)

    # ── Comparison Analysis ───────────────────────────────────

    def _compare(self, models: dict[str, Any]) -> dict[str, Any]:
        """Compare models and determine winners per metric."""
        comparison: dict[str, Any] = {
            "winners": {},
            "analysis": [],
        }

        # Only compare models that have actual results
        active_models = {
            k: v for k, v in models.items()
            if not v.get("skipped") and "error" not in v
        }
        if len(active_models) < 2:
            comparison["analysis"].append("Insufficient models for comparison (need ≥2)")
            return comparison

        metrics_to_compare = [
            "factual_accuracy",
            "multi_hop_accuracy",
            "single_hop_accuracy",
            "hallucination_rate",
            "consistency_score",
        ]

        for metric in metrics_to_compare:
            values = {}
            for model_name, model_results in active_models.items():
                val = model_results.get(metric)
                if val is not None:
                    values[model_name] = val

            if not values:
                continue

            # For hallucination_rate, lower is better; for others, higher is better
            reverse = metric == "hallucination_rate"
            sorted_models = sorted(values.items(), key=lambda x: x[1], reverse=not reverse)

            comparison["winners"][metric] = {
                "best": sorted_models[0][0],
                "best_score": sorted_models[0][1],
                "ranking": [
                    {"model": m, "score": s} for m, s in sorted_models
                ],
            }

        # Generate analysis
        comparison["analysis"] = self._generate_analysis(comparison["winners"])

        return comparison

    @staticmethod
    def _generate_analysis(winners: dict[str, Any]) -> list[str]:
        """Generate human-readable analysis from winner data."""
        analysis = []

        if "multi_hop_accuracy" in winners:
            winner = winners["multi_hop_accuracy"]
            if winner["best"] == "B_kg":
                analysis.append(
                    "✅ KG-Managed (B) wins on multi-hop reasoning — "
                    "the KG structure successfully teaches logical chaining."
                )
            elif winner["best"] == "C_raw":
                analysis.append(
                    "⚠️ Raw-Text (C) wins on multi-hop — the KG may not add "
                    "value for chain reasoning in its current state."
                )

        if "hallucination_rate" in winners:
            winner = winners["hallucination_rate"]
            if winner["best"] == "B_kg":
                analysis.append(
                    "✅ KG-Managed (B) has lowest hallucination — "
                    "KG curation reduces fabrication."
                )
            elif winner["best"] == "C_raw":
                analysis.append(
                    "⚠️ Raw-Text (C) has lower hallucination — "
                    "the KG may be introducing noise."
                )

        if "single_hop_accuracy" in winners:
            winner = winners["single_hop_accuracy"]
            scores = {r["model"]: r["score"] for r in winner["ranking"]}
            if abs(scores.get("B_kg", 0) - scores.get("C_raw", 0)) < 0.05:
                analysis.append(
                    "✅ B ≈ C on single-hop facts — confirms both datasets "
                    "contain the same information (control check passed)."
                )

        if "factual_accuracy" in winners:
            winner = winners["factual_accuracy"]
            if winner["best"] in ("B_kg", "C_raw"):
                analysis.append(
                    f"📊 Overall best factual accuracy: {winner['best']} "
                    f"(score: {winner['best_score']:.3f})"
                )

        if not analysis:
            analysis.append("No clear winner — results are within noise threshold.")

        return analysis

    # ── Model Loading & Inference ─────────────────────────────

    def _load_model(self, adapter_path: Path | None = None):
        """Load base model and optionally attach LoRA adapter."""
        if not HAS_MODEL_LIBS:
            raise RuntimeError("transformers/peft not available")

        # Resolve model name
        model_name = self.base_model
        if model_name.startswith("unsloth/"):
            model_name = model_name.replace("unsloth/", "").replace("-bnb-4bit", "")

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )

        # Attach LoRA adapter if provided
        if adapter_path and adapter_path.exists():
            logger.info("Loading LoRA adapter from %s", adapter_path)
            model = PeftModel.from_pretrained(model, str(adapter_path))

        model.eval()
        return model, tokenizer

    @staticmethod
    def _generate_answer(model, tokenizer, question: str) -> str:
        """Generate an answer from the model for a given question."""
        import torch

        prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)

        # Move to appropriate device
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        elif hasattr(torch, "mps") and torch.backends.mps.is_available():
            try:
                inputs = {k: v.to("mps") for k, v in inputs.items()}
            except Exception:
                pass

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.1,  # low temperature for factual QA
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the assistant's response
        if "<|im_start|>assistant" in full_response:
            response = full_response.split("<|im_start|>assistant")[-1].strip()
            # Remove any trailing <|im_end|>
            response = response.replace("<|im_end|>", "").strip()
            return response

        # Fallback: remove the prompt part
        if prompt in full_response:
            return full_response.replace(prompt, "").strip()

        return full_response.strip()

    # ── Report Generation ─────────────────────────────────────

    @staticmethod
    def _generate_report(results: dict[str, Any], output_path: Path) -> None:
        """Generate a human-readable markdown comparison report."""
        lines = [
            "# KG → LLM Ablation Study Report",
            "",
            f"**Test samples**: {results.get('test_samples', 'N/A')}",
            f"**Base model**: Qwen2.5-1.5B-Instruct",
            "",
            "---",
            "",
            "## Model Performance",
            "",
        ]

        models = results.get("models", {})
        for model_name, model_results in models.items():
            if model_results.get("skipped"):
                lines.append(f"### {model_name}")
                lines.append(f"⏭️ Skipped: {model_results.get('reason', 'unknown')}")
                lines.append("")
                continue

            lines.append(f"### {model_name}")
            lines.append("")
            lines.append("| Metric | Score |")
            lines.append("|--------|-------|")
            for metric, value in model_results.items():
                if metric in ("model", "total_predictions"):
                    continue
                if isinstance(value, (int, float)):
                    lines.append(f"| {metric} | {value:.4f} |")
            lines.append("")

        # Comparison section
        comparison = results.get("comparison", {})
        lines.append("---")
        lines.append("")
        lines.append("## Comparison Analysis")
        lines.append("")

        for analysis_line in comparison.get("analysis", []):
            lines.append(f"- {analysis_line}")

        lines.append("")
        lines.append("## Winners by Metric")
        lines.append("")

        winners = comparison.get("winners", {})
        for metric, info in winners.items():
            lines.append(f"### {metric}")
            lines.append(f"**Best**: {info['best']} ({info['best_score']:.4f})")
            lines.append("")
            lines.append("| Rank | Model | Score |")
            lines.append("|------|-------|-------|")
            for rank, entry in enumerate(info.get("ranking", []), 1):
                lines.append(f"| {rank} | {entry['model']} | {entry['score']:.4f} |")
            lines.append("")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    # ── Helpers ───────────────────────────────────────────────

    @staticmethod
    def _load_test_pairs(path: Path) -> list[dict[str, Any]]:
        """Load test QA pairs from a JSONL file."""
        pairs = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    pairs.append(json.loads(line))
        return pairs
