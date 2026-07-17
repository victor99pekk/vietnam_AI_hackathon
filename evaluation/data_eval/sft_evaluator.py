"""
Method 1, Step 3 — Extrinsic SFT Quality Evaluation

Uses deepeval (LLM-as-Judge framework) to evaluate the quality of SFT
training pairs generated from the knowledge graph.

Metrics:
  - Faithfulness: Does the response rely only on graph facts?
  - Answer Relevancy: Does the response actually answer the instruction?
  - Factual Correctness: Does the response match the original triples?
  - Semantic Diversity: Are the generated instructions varied enough?

Optionally falls back to a lightweight heuristic evaluation if deepeval
is not installed or the LLM API is unavailable.
"""

import json
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Try importing deepeval — gracefully degrade if not available
try:
    from deepeval import evaluate
    from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
    from deepeval.test_case import LLMTestCase
    DEEPEVAL_AVAILABLE = True
except ImportError:
    DEEPEVAL_AVAILABLE = False
    logger.warning("deepeval not installed — falling back to heuristic evaluation. "
                   "Install with: pip install deepeval")


class SFTEvaluator:
    """Evaluates SFT training data quality using deepeval (or heuristic fallback)."""

    def __init__(
        self,
        faithfulness_threshold: float = 0.7,
        relevancy_threshold: float = 0.7,
        judge_model: str = "deepseek-chat",
        min_diversity_score: float = 0.4,
    ) -> None:
        self.faithfulness_threshold = faithfulness_threshold
        self.relevancy_threshold = relevancy_threshold
        self.judge_model = judge_model
        self.min_diversity_score = min_diversity_score

    def evaluate(self, sft_file: Path) -> dict[str, Any]:
        """Evaluate a JSONL file of SFT pairs."""
        pairs = self._load_sft_pairs(sft_file)
        if not pairs:
            return {"error": "No SFT pairs found", "pairs_evaluated": 0}

        logger.info("Evaluating %d SFT pairs...", len(pairs))

        results: dict[str, Any] = {
            "pairs_evaluated": len(pairs),
            "metrics": {},
            "details": [],
            "overall_score": 0.0,
        }

        # Run deepeval if available, else use heuristics
        if DEEPEVAL_AVAILABLE:
            try:
                results["metrics"] = self._deepeval_evaluate(pairs)
            except Exception as e:
                logger.warning("deepeval evaluation failed: %s — falling back to heuristics", e)
                results["metrics"] = self._heuristic_evaluate(pairs)
        else:
            results["metrics"] = self._heuristic_evaluate(pairs)

        # Semantic diversity (works regardless of deepeval)
        results["metrics"]["semantic_diversity"] = self._semantic_diversity(pairs)

        # Compute overall score
        metrics = results["metrics"]
        scores = [
            metrics.get("faithfulness", 0) if isinstance(metrics.get("faithfulness"), (int, float)) else 0,
            metrics.get("answer_relevancy", 0) if isinstance(metrics.get("answer_relevancy"), (int, float)) else 0,
            metrics.get("factual_correctness", 0) if isinstance(metrics.get("factual_correctness"), (int, float)) else 0,
            metrics.get("semantic_diversity", {}).get("score", 0) if isinstance(metrics.get("semantic_diversity"), dict) else 0,
        ]
        results["overall_score"] = round(sum(scores) / len(scores), 3)

        # Verdict
        results["verdict"] = self._interpret(results["overall_score"], metrics)

        return results

    # ── deepeval Evaluation ───────────────────────────────────

    def _deepeval_evaluate(self, pairs: list[dict[str, Any]]) -> dict[str, Any]:
        """Evaluate SFT pairs using deepeval's LLM-as-Judge metrics.

        Configures deepeval to use DeepSeek API (OpenAI-compatible) if the
        DEEPSEEK_API_KEY environment variable is set.
        """
        if not DEEPEVAL_AVAILABLE:
            return self._heuristic_evaluate(pairs)

        # Configure OpenAI-compatible API for DeepSeek
        self._configure_deepseek_for_deepeval()

        test_cases = []
        for pair in pairs:
            test_cases.append(
                LLMTestCase(
                    input=pair.get("instruction", ""),
                    actual_output=pair.get("response", ""),
                    retrieval_context=[pair.get("context", "")],
                )
            )

        faithfulness = FaithfulnessMetric(
            threshold=self.faithfulness_threshold,
            model=self.judge_model,
        )
        relevancy = AnswerRelevancyMetric(
            threshold=self.relevancy_threshold,
            model=self.judge_model,
        )

        try:
            results = evaluate(test_cases, [faithfulness, relevancy])
        except Exception as e:
            logger.warning("deepeval evaluation failed: %s — falling back to heuristics", e)
            return self._heuristic_evaluate(pairs)

        # Aggregate scores
        faith_scores = []
        relevancy_scores = []
        for result in results:
            faith_scores.append(result.metrics[0].score if result.metrics else 0)
            relevancy_scores.append(result.metrics[1].score if len(result.metrics) > 1 else 0)

        return {
            "faithfulness": round(sum(faith_scores) / max(len(faith_scores), 1), 3),
            "answer_relevancy": round(sum(relevancy_scores) / max(len(relevancy_scores), 1), 3),
            "factual_correctness": self._factual_correctness_heuristic(pairs)["score"],
        }

    # ── Heuristic Fallback Evaluation ─────────────────────────

    def _heuristic_evaluate(self, pairs: list[dict[str, Any]]) -> dict[str, Any]:
        """Fallback heuristic evaluation when deepeval is unavailable."""
        faith = self._faithfulness_heuristic(pairs)
        relevancy = self._relevancy_heuristic(pairs)
        factual = self._factual_correctness_heuristic(pairs)

        return {
            "faithfulness": faith["score"],
            "faithfulness_detail": faith.get("detail", ""),
            "answer_relevancy": relevancy["score"],
            "relevancy_detail": relevancy.get("detail", ""),
            "factual_correctness": factual["score"],
            "factual_detail": factual.get("detail", ""),
            "note": "Heuristic evaluation (deepeval not available). Install deepeval for LLM-as-Judge scoring.",
        }

    def _faithfulness_heuristic(self, pairs: list[dict[str, Any]]) -> dict[str, Any]:
        """Heuristic: does the response contain ONLY entities from the context triples?

        Lower score = response mentions entities not in the source graph facts.
        """
        violations = 0
        for pair in pairs:
            context = pair.get("context", "")
            response = pair.get("response", "")

            # Extract capitalized phrases from response (proxy for entities)
            # Check if they appear in context
            response_entities = set(
                w for w in response.split()
                if w[0].isupper() and len(w) > 1
            )

            context_lower = context.lower()
            hallucinated = [
                e for e in response_entities
                if e.lower() not in context_lower
            ]

            if hallucinated:
                violations += 1

        score = 1.0 - (violations / max(len(pairs), 1))
        return {
            "score": round(score, 3),
            "detail": f"{violations}/{len(pairs)} responses contain entities not in source triples",
        }

    def _relevancy_heuristic(self, pairs: list[dict[str, Any]]) -> dict[str, Any]:
        """Heuristic: does the response share keywords with the instruction?

        Crude proxy — real relevancy needs LLM-as-Judge.
        """
        scores = []
        for pair in pairs:
            instruction = set(pair.get("instruction", "").lower().split())
            response = set(pair.get("response", "").lower().split())

            # Remove stopwords
            stopwords = {"the", "a", "an", "is", "was", "are", "were", "of", "to",
                         "in", "for", "on", "and", "or", "it", "its", "be", "has",
                         "have", "had", "do", "does", "did", "what", "who", "where",
                         "when", "why", "how", "?"}
            instruction = instruction - stopwords
            response = response - stopwords

            if not instruction:
                scores.append(0.5)
                continue

            overlap = instruction & response
            scores.append(len(overlap) / len(instruction))

        return {
            "score": round(sum(scores) / max(len(scores), 1), 3),
            "detail": "Keyword overlap between instruction and response (heuristic proxy)",
        }

    def _factual_correctness_heuristic(self, pairs: list[dict[str, Any]]) -> dict[str, Any]:
        """Heuristic: does the response contain the subject and object from the triples?"""
        scores = []
        for pair in pairs:
            triples = pair.get("triples", [])
            response = pair.get("response", "").lower()
            if not triples:
                scores.append(0.5)
                continue

            matches = 0
            for t in triples:
                subj = t.get("subject", "").lower()
                obj = t.get("object", "").lower()
                if subj and subj in response:
                    matches += 0.5
                if obj and obj in response:
                    matches += 0.5

            scores.append(matches / len(triples))

        return {
            "score": round(sum(scores) / max(len(scores), 1), 3),
            "detail": "Presence of triple entities in response (heuristic proxy)",
        }

    # ── Semantic Diversity ────────────────────────────────────

    def _semantic_diversity(self, pairs: list[dict[str, Any]]) -> dict[str, Any]:
        """Measure instruction diversity to detect template overfitting.

        Uses unique bigram ratio across all instructions.
        Low diversity = all instructions are essentially the same template.
        """
        if not pairs:
            return {"score": 0, "detail": "No pairs to evaluate"}

        # Extract bigrams from all instructions
        all_bigrams: list[str] = []
        for pair in pairs:
            words = pair.get("instruction", "").lower().split()
            all_bigrams.extend(
                f"{words[i]}_{words[i+1]}"
                for i in range(len(words) - 1)
            )

        if not all_bigrams:
            return {"score": 0, "detail": "No bigrams found"}

        unique_ratio = len(set(all_bigrams)) / len(all_bigrams)

        # Most common bigrams (template detection)
        bigram_counts = Counter(all_bigrams)
        top_bigrams = bigram_counts.most_common(5)

        flag = "green" if unique_ratio >= self.min_diversity_score else "red"

        return {
            "score": round(unique_ratio, 3),
            "unique_bigram_ratio": round(unique_ratio, 3),
            "most_common_bigrams": [
                {"bigram": b, "count": c} for b, c in top_bigrams
            ],
            "flag": flag,
            "detail": (
                f"Good diversity — {round(unique_ratio * 100)}% unique bigrams."
                if flag == "green"
                else f"Low diversity — only {round(unique_ratio * 100)}% unique bigrams. Instructions may be too templated."
            ),
        }

    # ── Helpers ───────────────────────────────────────────────

    @staticmethod
    def _configure_deepseek_for_deepeval() -> None:
        """Set OpenAI-compatible env vars so deepeval uses DeepSeek API."""
        import os
        from dotenv import load_dotenv
        load_dotenv()

        deepseek_key = os.getenv("DEEPSEEK_API_KEY", "")
        if deepseek_key and not os.getenv("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = deepseek_key
            os.environ["OPENAI_BASE_URL"] = "https://api.deepseek.com/v1"
            logger.info("Configured deepeval to use DeepSeek API")

    @staticmethod
    def _load_sft_pairs(path: Path) -> list[dict[str, Any]]:
        """Load SFT pairs from a JSONL file."""
        pairs = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        pairs.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning("Skipping malformed JSON line in %s", path)
        return pairs

    @staticmethod
    def _interpret(score: float, metrics: dict[str, Any]) -> str:
        if score >= 0.8:
            return "✅ Excellent — SFT data quality is high. Proceed to fine-tuning (Method 2)."
        elif score >= 0.6:
            return "⚠️ Good — usable but review flagged metrics before fine-tuning."
        elif score >= 0.4:
            return "🔶 Fair — consider improving the KG before generating SFT data."
        else:
            return "🔴 Poor — significant quality issues. Fix the KG structure first."
