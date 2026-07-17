"""
Method 1, Step 2 — SFT Synthetic Data Generation

Samples N-hop subgraphs from a knowledge graph and uses an LLM to generate
realistic instruction/response SFT training pairs.

The generated pairs are then evaluated in Step 3 (sft_evaluator.py) using
deepeval for faithfulness, relevancy, and factual correctness.
"""

import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import networkx as nx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Prompt template for generating SFT pairs from KG subgraphs
SFT_GENERATION_PROMPT = """You are a dataset curator creating high-quality instruction/response pairs for fine-tuning a language model.

Below are facts extracted from a knowledge graph. Each fact is a (subject, relation, object) triple.

FACTS:
{context}

TASK: Based ONLY on the facts above, generate a realistic instruction/response pair.
- The instruction should be a natural question that a user might ask.
- The response should be a comprehensive, accurate answer using ONLY the provided facts.
- Do NOT add any information not present in the facts.
- Make the instruction diverse — vary the phrasing, not just "What is X?"

Respond in JSON format:
{{"instruction": "<user question>", "response": "<answer using only the facts>"}}

JSON:"""


class SFTGenerator:
    """Generates SFT instruction/response pairs from KG subgraphs using an LLM."""

    def __init__(
        self,
        model: str = "deepseek-chat",
        provider: str = "deepseek",
        num_samples: int = 50,
        hop_distribution: tuple[float, float, float] = (0.3, 0.4, 0.3),
        temperature: float = 0.7,
        max_tokens: int = 512,
        seed: int = 42,
    ) -> None:
        self.model = model
        self.provider = provider
        self.num_samples = num_samples
        self.hop_distribution = hop_distribution
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        random.seed(seed)

        self._client = None

    @property
    def client(self):
        """Lazy init the LLM client based on provider."""
        if self._client is None:
            if self.provider == "deepseek":
                from openai import OpenAI
                api_key = os.getenv("DEEPSEEK_API_KEY", "")
                self._client = OpenAI(
                    api_key=api_key,
                    base_url="https://api.deepseek.com",
                )
            elif self.provider == "openai":
                from openai import OpenAI
                self._client = OpenAI()
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
        return self._client

    def generate(
        self,
        graph: nx.DiGraph,
        triples: list[tuple[str, str, str, str]],
        output_dir: Path,
    ) -> Path:
        """Generate SFT pairs from KG subgraphs and save to a JSONL file.

        Returns the path to the generated file.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Sample subgraphs at different hop depths
        samples_1hop = self._sample_subgraphs(graph, triples, hops=1)
        samples_2hop = self._sample_subgraphs(graph, triples, hops=2)
        samples_3hop = self._sample_subgraphs(graph, triples, hops=3)

        # Distribute according to hop_distribution
        n1 = int(self.num_samples * self.hop_distribution[0])
        n2 = int(self.num_samples * self.hop_distribution[1])
        n3 = self.num_samples - n1 - n2

        all_samples = (
            random.sample(samples_1hop, min(n1, len(samples_1hop)))
            + random.sample(samples_2hop, min(n2, len(samples_2hop)))
            + random.sample(samples_3hop, min(n3, len(samples_3hop)))
        )
        random.shuffle(all_samples)

        logger.info(
            "Generating %d SFT pairs (1-hop: %d, 2-hop: %d, 3-hop: %d)",
            len(all_samples), min(n1, len(samples_1hop)),
            min(n2, len(samples_2hop)), min(n3, len(samples_3hop)),
        )

        # Generate SFT pairs via LLM
        sft_pairs: list[dict[str, Any]] = []
        for i, (triple_set, hop_count) in enumerate(all_samples):
            try:
                pair = self._generate_single(triple_set, hop_count)
                if pair:
                    sft_pairs.append(pair)
                logger.info("  [%d/%d] Generated SFT pair (%d-hop)", i + 1, len(all_samples), hop_count)
            except Exception as e:
                logger.warning("  [%d/%d] Failed: %s", i + 1, len(all_samples), e)

        # Save
        output_path = output_dir / "sft_sample.jsonl"
        with open(output_path, "w", encoding="utf-8") as f:
            for pair in sft_pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")

        logger.info("Saved %d SFT pairs → %s", len(sft_pairs), output_path)
        return output_path

    # ── Internal methods ──────────────────────────────────────

    def _sample_subgraphs(
        self, graph: nx.DiGraph, triples: list[tuple[str, str, str, str]], hops: int
    ) -> list[tuple[list[tuple[str, str, str]], int]]:
        """Sample subgraphs of a given hop depth from the KG."""
        samples: list[tuple[list[tuple[str, str, str]], int]] = []

        if hops == 1:
            # Single triples
            for t in triples[: min(200, len(triples))]:
                samples.append(([(t[0], t[1], t[2])], 1))
        else:
            # Multi-hop: traverse from random nodes
            start_nodes = list(graph.nodes())[:200]
            random.shuffle(start_nodes)

            for start in start_nodes:
                paths = self._find_paths(graph, start, max_depth=hops)
                for path_edges in paths:
                    if len(path_edges) == hops:
                        triples_set = [
                            (s, graph.edges[s, o].get("predicates", ["related_to"])[0], o)
                            for s, o in path_edges
                            if graph.has_edge(s, o)
                        ]
                        if len(triples_set) == hops:
                            samples.append((triples_set, hops))
                        if len(samples) >= 100:
                            break
                if len(samples) >= 100:
                    break

        return samples

    @staticmethod
    def _find_paths(
        graph: nx.DiGraph, start: str, max_depth: int
    ) -> list[list[tuple[str, str]]]:
        """Find all paths of exactly max_depth edges from a start node."""
        all_paths: list[list[tuple[str, str]]] = []

        def dfs(current: str, path: list[tuple[str, str]], depth: int):
            if depth == max_depth:
                all_paths.append(list(path))
                return
            for neighbor in graph.successors(current):
                # Avoid cycles
                visited_nodes = {p[0] for p in path} | {p[1] for p in path}
                if neighbor not in visited_nodes:
                    path.append((current, neighbor))
                    dfs(neighbor, path, depth + 1)
                    path.pop()

        dfs(start, [], 0)
        return all_paths

    def _generate_single(
        self, triples_set: list[tuple[str, str, str]], hop_count: int
    ) -> dict[str, Any] | None:
        """Generate a single SFT pair from a set of triples."""
        context_lines = [
            f"- ({s} | {r.replace('_', ' ')} | {o})"
            for s, r, o in triples_set
        ]
        context = "\n".join(context_lines)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": SFT_GENERATION_PROMPT.format(context=context),
                }
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        if not content:
            return None

        try:
            pair = json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON from the response
            import re
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                try:
                    pair = json.loads(match.group())
                except json.JSONDecodeError:
                    return None
            else:
                return None

        return {
            "instruction": pair.get("instruction", ""),
            "response": pair.get("response", ""),
            "context": context,
            "triples": [
                {"subject": s, "relation": r, "object": o}
                for s, r, o in triples_set
            ],
            "hop_count": hop_count,
            "type": "sft_pair",
        }


# ── Fallback: Template-based generation (no LLM needed) ──────

class TemplateSFTGenerator:
    """Template-based SFT pair generator — no external LLM required.

    Useful for quick testing or when API access is unavailable.
    Produces simpler but deterministic instruction/response pairs.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        random.seed(seed)

    def generate(
        self,
        graph: nx.DiGraph,
        triples: list[tuple[str, str, str, str]],
        output_dir: Path,
        num_samples: int = 50,
    ) -> Path:
        """Generate template-based SFT pairs."""
        output_dir.mkdir(parents=True, exist_ok=True)

        sft_pairs: list[dict[str, Any]] = []

        # Single-hop templates
        for t in random.sample(triples, min(num_samples, len(triples))):
            subj, pred, obj, _ = t
            pred_readable = pred.replace("_", " ")

            templates = [
                (f"What is the {pred_readable} of {subj}?", f"{subj} {pred_readable} {obj}."),
                (f"Tell me about {subj}'s {pred_readable}.", f"{subj} {pred_readable} {obj}."),
                (f"How is {subj} connected to {obj}?", f"{subj} {pred_readable} {obj}."),
            ]

            for instruction, response in templates:
                sft_pairs.append({
                    "instruction": instruction,
                    "response": response,
                    "context": f"({subj} - {pred} - {obj})",
                    "triples": [{"subject": subj, "relation": pred, "object": obj}],
                    "hop_count": 1,
                    "type": "sft_pair_template",
                })

        random.shuffle(sft_pairs)
        sft_pairs = sft_pairs[:num_samples]

        output_path = output_dir / "sft_sample.jsonl"
        with open(output_path, "w", encoding="utf-8") as f:
            for pair in sft_pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")

        logger.info("Template SFT: %d pairs → %s", len(sft_pairs), output_path)
        return output_path
