"""Paper-faithful GraphGen KG extraction using DeepSeek as the synthesizer."""

from __future__ import annotations

from collections import defaultdict
import html
import logging
import os
import re
from typing import Any

from kg_generator.config import DEFAULT_GRAPHGEN_ENTITY_TYPES, Language
from kg_generator.extract.entities import Entity
from kg_generator.extract.graphgen_prompts import (
    COMPLETION_DELIMITER,
    CONTINUE_PROMPT,
    CONTINUE_PROMPT_VI,
    FIGURE_8_TEMPLATE,
    FIGURE_8_VI_TEMPLATE,
    FIGURE_9_SUMMARIZATION_TEMPLATE,
    FIGURE_9_SUMMARIZATION_TEMPLATE_VI,
    IF_LOOP_PROMPT,
    IF_LOOP_PROMPT_VI,
    RECORD_DELIMITER,
    TUPLE_DELIMITER,
)

logger = logging.getLogger(__name__)


class GraphGenExtractor:
    """Extract and aggregate the descriptive graph used by GraphGen.

    Figure 8 does not predict a predicate taxonomy. Each knowledge edge is an
    ordered entity pair plus a natural-language description. ``RELATION`` is
    only the neutral carrier label required by our tuple/Neo4j representation.
    """

    prompt_version = "graphgen-figure8-figure9-v2"

    def __init__(
        self,
        language: Language = Language.ENGLISH,
        model_name: str = "deepseek-v4-flash",
        entity_types: tuple[str, ...] = tuple(DEFAULT_GRAPHGEN_ENTITY_TYPES),
        client: Any | None = None,
        max_retries: int = 3,
        max_gleanings: int = 3,
    ) -> None:
        self.language = language
        self.model_name = model_name
        self.entity_types = tuple(entity_type.upper() for entity_type in entity_types)
        self._provided_client = client
        self.max_retries = max_retries
        self.max_gleanings = max_gleanings

    @property
    def output_language(self) -> str:
        return "Vietnamese" if self.language == Language.VIETNAMESE else "English"

    def extract(
        self,
        text: str,
        source_chunk_id: str = "",
    ) -> tuple[list[Entity], list[tuple[str, ...]]]:
        """Run Figure 8 extraction and the reference iterative gleaning loop."""
        if not text.strip():
            return [], []

        prompt_template = (
            FIGURE_8_VI_TEMPLATE
            if self.language == Language.VIETNAMESE
            else FIGURE_8_TEMPLATE
        )
        prompt = prompt_template.format(
            output_language=self.output_language,
            entity_types=", ".join(entity_type.lower() for entity_type in self.entity_types),
            tuple_delimiter=TUPLE_DELIMITER,
            record_delimiter=RECORD_DELIMITER,
            completion_delimiter=COMPLETION_DELIMITER,
            input_text=text,
        )
        initial = self._generate([{"role": "user", "content": prompt}])
        final_result = initial
        history = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": initial},
        ]

        for _ in range(self.max_gleanings):
            if_loop = self._generate(
                [
                    *history,
                    {
                        "role": "user",
                        "content": IF_LOOP_PROMPT_VI
                        if self.language == Language.VIETNAMESE
                        else IF_LOOP_PROMPT,
                    },
                ],
                max_tokens=64,
            )
            if if_loop.strip().strip('"').strip("'").casefold() != "yes":
                break

            glean = self._generate(
                [
                    *history,
                    {
                        "role": "user",
                        "content": CONTINUE_PROMPT_VI
                        if self.language == Language.VIETNAMESE
                        else CONTINUE_PROMPT,
                    },
                ]
            )
            final_result += f"{RECORD_DELIMITER}{glean}"
            history.extend(
                [
                    {
                        "role": "user",
                        "content": CONTINUE_PROMPT_VI
                        if self.language == Language.VIETNAMESE
                        else CONTINUE_PROMPT,
                    },
                    {"role": "assistant", "content": glean},
                ]
            )

        return self._parse(final_result, source_chunk_id)

    def aggregate_descriptions(
        self,
        resolved_entities: list[dict[str, Any]],
        original_entities: list[dict[str, Any]],
        entity_id_map: dict[str, str],
        triples: list[tuple[str, ...]],
    ) -> tuple[list[dict[str, Any]], list[tuple[str, ...]]]:
        """Apply Figure 9 to repeated entity and relationship descriptions."""
        entity_descriptions: dict[str, list[str]] = defaultdict(list)
        for entity in original_entities:
            entity_id = str(entity.get("id", ""))
            canonical_id = entity_id_map.get(entity_id, entity_id)
            description = str(entity.get("description", "")).strip()
            if canonical_id and description:
                entity_descriptions[canonical_id].append(description)

        for entity in resolved_entities:
            descriptions = entity_descriptions.get(str(entity.get("id", "")), [])
            if descriptions:
                entity["description"] = self._merge_descriptions(
                    str(entity.get("name", "entity")), descriptions
                )

        relation_descriptions: dict[tuple[str, str], list[str]] = defaultdict(list)
        for triple in triples:
            if len(triple) > 5 and triple[1] == "RELATION" and triple[5]:
                relation_descriptions[(triple[0], triple[2])].append(str(triple[5]))

        merged_relations = {
            key: self._merge_descriptions(f"({key[0]}, {key[1]})", descriptions)
            for key, descriptions in relation_descriptions.items()
        }
        aggregated_triples = [
            (
                *triple[:5],
                merged_relations[(triple[0], triple[2])],
            )
            if triple[1] == "RELATION" and (triple[0], triple[2]) in merged_relations
            else triple
            for triple in triples
        ]
        return resolved_entities, aggregated_triples

    def _client(self) -> Any:
        if self._provided_client is not None:
            return self._provided_client

        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY is not set. Add it to .env before using --llm."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The LLM dependencies are missing. Install with: "
                "/Users/armon/.local/bin/uv sync --extra llm"
            ) from exc

        self._provided_client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
        )
        return self._provided_client

    def _generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 4096,
    ) -> str:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._client().chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0,
                    max_tokens=max_tokens,
                    extra_body={"thinking": {"type": "disabled"}},
                )
                content = response.choices[0].message.content or ""
                if not content.strip():
                    raise ValueError("DeepSeek returned an empty response")
                return content.strip()
            except (AttributeError, IndexError, TypeError, ValueError) as exc:
                last_error = exc
                logger.warning(
                    "GraphGen request attempt %d/%d failed: %s",
                    attempt,
                    self.max_retries,
                    exc,
                )

        raise RuntimeError(
            f"DeepSeek GraphGen request failed after {self.max_retries} attempts"
        ) from last_error

    def _parse(
        self,
        content: str,
        source_chunk_id: str,
    ) -> tuple[list[Entity], list[tuple[str, ...]]]:
        records = re.split(
            f"{re.escape(RECORD_DELIMITER)}|{re.escape(COMPLETION_DELIMITER)}",
            content,
        )
        entity_records: list[tuple[str, str, str]] = []
        relationship_records: list[tuple[str, str, str]] = []
        content_keywords: list[str] = []

        for record in records:
            match = re.search(r"\((.*)\)", record.strip(), flags=re.DOTALL)
            if not match:
                continue
            fields = [self._clean(field) for field in match.group(1).split(TUPLE_DELIMITER)]
            if len(fields) < 2:
                continue
            record_type = fields[0].casefold()
            if record_type == "entity" and len(fields) >= 4:
                name, entity_type, description = fields[1], fields[2].upper(), fields[3]
                if name and entity_type in self.entity_types:
                    entity_records.append((name, entity_type, description))
            elif record_type == "relationship" and len(fields) >= 4:
                relationship_records.append((fields[1], fields[2], fields[3]))
            elif record_type == "content_keywords":
                content_keywords.extend(
                    keyword.strip() for keyword in fields[1].split(",") if keyword.strip()
                )

        entities = [
            Entity(
                name=name,
                label=entity_type,
                mentions=[name],
                description=description,
                source=source_chunk_id,
            )
            for name, entity_type, description in entity_records
        ]
        entities_by_name: dict[str, Entity] = {}
        for entity in entities:
            entities_by_name.setdefault(entity.name.casefold(), entity)

        relationships: list[tuple[str, ...]] = []
        for source_name, target_name, description in relationship_records:
            source = entities_by_name.get(source_name.casefold())
            target = entities_by_name.get(target_name.casefold())
            if source is None or target is None or source.id == target.id:
                logger.warning(
                    "Discarding GraphGen relationship with a missing/identical endpoint: %s -> %s",
                    source_name,
                    target_name,
                )
                continue
            relationships.append(
                (
                    source.id,
                    "RELATION",
                    target.id,
                    "",
                    source_chunk_id,
                    description,
                )
            )

        logger.debug(
            "GraphGenExtractor: %d entities, %d relationships, keywords=%s",
            len(entities),
            len(relationships),
            content_keywords,
        )
        return entities, relationships

    def _merge_descriptions(self, name: str, descriptions: list[str]) -> str:
        unique = sorted({description.strip() for description in descriptions if description.strip()})
        if not unique:
            return ""
        if len(unique) == 1:
            return unique[0]

        summary_template = (
            FIGURE_9_SUMMARIZATION_TEMPLATE_VI
            if self.language == Language.VIETNAMESE
            else FIGURE_9_SUMMARIZATION_TEMPLATE
        )
        prompt = summary_template.format(
            output_language=self.output_language,
            name=name,
            description_list=unique,
        )
        return self._generate([{"role": "user", "content": prompt}])

    @staticmethod
    def _clean(value: str) -> str:
        value = html.unescape(value.strip())
        value = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", value)
        return value.strip().strip('"').strip("'").strip()
