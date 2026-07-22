"""GraphGen-style multi-hop QA generation from sampled subgraphs."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv


PROMPT_VERSION = "graphgen-figure15-inspired-v1"

MULTI_HOP_QA_PROMPT = """You are creating supervised fine-tuning data from a bounded knowledge subgraph.

The subgraph contains entity descriptions and relationship descriptions extracted from source documents.

SUBGRAPH:
{context}

Create exactly one multi-hop question and answer.

Requirements:
1. Select a connected path containing at least two listed relationships.
2. Ask for one conclusion that can only be reached by traversing that path.
3. Do not ask a compound question that merely requests two independent facts joined by "and".
4. The answer must be supported only by the listed descriptions; do not add outside knowledge.
5. Make the question natural and self-contained in {language}.
6. Return the relationship IDs in traversal order.
7. Do not put hidden chain-of-thought in the answer.

Return one JSON object with this schema:
{{
  "question": "...",
  "answer": "...",
  "used_edge_ids": ["edge:...", "edge:..."]
}}
"""


class GraphGenQAGenerator:
    """Use an LLM only after deterministic subgraph organization is complete."""

    def __init__(
        self,
        *,
        model: str = "deepseek-v4-pro",
        provider: str = "deepseek",
        language: str = "English",
        temperature: float = 0.2,
        max_tokens: int = 512,
        llm_call: Callable[[str], str] | None = None,
    ) -> None:
        self.model = model
        self.provider = provider
        self.language = language
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm_call = llm_call
        self._client: Any = None

    def generate(
        self,
        subgraphs: list[dict[str, Any]],
        output_dir: Path,
        *,
        max_questions: int | None = None,
    ) -> tuple[Path, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        qa_path = output_dir / "qa.jsonl"
        audit_path = output_dir / "qa_audit.jsonl"
        candidates = [item for item in subgraphs if len(item.get("edges", [])) >= 2]
        if max_questions is not None:
            candidates = candidates[:max_questions]

        pairs: list[dict[str, Any]] = []
        audit: list[dict[str, Any]] = []
        for subgraph in candidates:
            try:
                prompt = MULTI_HOP_QA_PROMPT.format(
                    context=self._format_context(subgraph), language=self.language
                )
                raw = self._call(prompt)
                payload = self._parse_json(raw)
                pair, reason = self._validate(payload, subgraph)
                if pair is None:
                    audit.append(self._audit(subgraph, "rejected", reason))
                    continue
                pairs.append(pair)
                audit.append(self._audit(subgraph, "accepted", "valid_multi_hop_qa"))
            except Exception as exc:
                audit.append(
                    self._audit(subgraph, "error", f"{type(exc).__name__}: {exc}")
                )

        with open(qa_path, "w", encoding="utf-8") as handle:
            for pair in pairs:
                handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
        with open(audit_path, "w", encoding="utf-8") as handle:
            for event in audit:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        return qa_path, audit_path

    def _call(self, prompt: str) -> str:
        if self.llm_call is not None:
            return self.llm_call(prompt)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
            extra_body={"thinking": {"type": "disabled"}}
            if self.provider == "deepseek"
            else None,
        )
        return response.choices[0].message.content or ""

    @property
    def client(self):
        if self._client is None:
            load_dotenv()
            from openai import OpenAI

            if self.provider == "deepseek":
                api_key = os.getenv("DEEPSEEK_API_KEY", "")
                if not api_key:
                    raise RuntimeError("DEEPSEEK_API_KEY is required for QA generation")
                self._client = OpenAI(
                    api_key=api_key, base_url="https://api.deepseek.com"
                )
            elif self.provider == "openai":
                self._client = OpenAI()
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
        return self._client

    def _validate(
        self, payload: dict[str, Any], subgraph: dict[str, Any]
    ) -> tuple[dict[str, Any] | None, str]:
        question = str(payload.get("question", "")).strip()
        answer = str(payload.get("answer", "")).strip()
        used_edge_ids = payload.get("used_edge_ids", [])
        if not question or not answer:
            return None, "missing_question_or_answer"
        if not isinstance(used_edge_ids, list):
            return None, "used_edge_ids_must_be_a_list"

        available = {edge["id"] for edge in subgraph.get("edges", [])}
        used = [str(edge_id) for edge_id in used_edge_ids]
        if len(set(used)) < 2:
            return None, "question_does_not_use_two_edges"
        if not set(used).issubset(available):
            return None, "unknown_used_edge_id"

        edge_by_id = {edge["id"]: edge for edge in subgraph.get("edges", [])}
        for left_id, right_id in zip(used, used[1:]):
            left = edge_by_id[left_id]
            right = edge_by_id[right_id]
            left_nodes = {left["source"], left["target"]}
            right_nodes = {right["source"], right["target"]}
            if not left_nodes.intersection(right_nodes):
                return None, "used_edges_do_not_form_an_ordered_path"

        evidence_by_id = {
            edge["id"]: edge.get("description", "")
            for edge in subgraph.get("edges", [])
        }
        return {
            "instruction": question,
            "response": answer,
            "scenario": "multi_hop",
            "subgraph_id": subgraph["id"],
            "used_edge_ids": used,
            "evidence_chain": [evidence_by_id[edge_id] for edge_id in used],
            "source_chunk_ids": subgraph.get("source_chunk_ids", []),
            "generator": {
                "method": "graphgen_style",
                "model": self.model,
                "provider": self.provider,
                "prompt_version": PROMPT_VERSION,
                "language": self.language,
            },
        }, ""

    @staticmethod
    def _format_context(subgraph: dict[str, Any]) -> str:
        node_by_id = {node["id"]: node for node in subgraph.get("nodes", [])}
        lines = ["ENTITIES:"]
        for node in subgraph.get("nodes", []):
            lines.append(
                f"- {node['id']} | {node.get('name', '')} | "
                f"{node.get('type', '')} | {node.get('description', '')}"
            )
        lines.append("RELATIONSHIPS:")
        for edge in subgraph.get("edges", []):
            source = node_by_id.get(edge["source"], {}).get("name", edge["source"])
            target = node_by_id.get(edge["target"], {}).get("name", edge["target"])
            lines.append(
                f"- {edge['id']} | {source} -> {target} | {edge.get('description', '')}"
            )
        return "\n".join(lines)

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any]:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                raise ValueError("LLM response did not contain a JSON object")
            payload = json.loads(match.group())
        if not isinstance(payload, dict):
            raise ValueError("LLM response must be a JSON object")
        return payload

    def _audit(
        self, subgraph: dict[str, Any], decision: str, reason: str
    ) -> dict[str, Any]:
        return {
            "subgraph_id": subgraph.get("id", ""),
            "decision": decision,
            "reason": reason,
            "edge_count": len(subgraph.get("edges", [])),
            "source_chunk_ids": subgraph.get("source_chunk_ids", []),
            "model": self.model,
            "prompt_version": PROMPT_VERSION,
        }
