#!/usr/bin/env python3
"""Stream a fixed-size English or Vietnamese Wikipedia sample to pipeline-ready JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

DATASET_NAME = "wikimedia/wikipedia"
DATASET_URL = "https://huggingface.co/datasets/wikimedia/wikipedia"
LICENSE = "CC-BY-SA-3.0 AND GFDL"
DEFAULT_SNAPSHOT = "20231101"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=("en", "vi"), required=True)
    parser.add_argument("--count", type=int, required=True, help="Number of non-empty articles to write.")
    parser.add_argument("--output", type=Path, required=True, help="Destination JSONL file.")
    parser.add_argument("--snapshot", default=DEFAULT_SNAPSHOT, help="Wikimedia snapshot date, for example 20231101.")
    parser.add_argument(
        "--manifest-output",
        type=Path,
        help="Optional source-manifest YAML path. Defaults next to the JSONL output.",
    )
    return parser.parse_args()


def stream_articles(snapshot: str, language: str) -> Iterable[dict[str, Any]]:
    """Yield public Wikipedia articles without downloading the complete split."""
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "This script requires Hugging Face datasets. Install it with: pip install -e '.[data]'"
        ) from error
    return load_dataset(DATASET_NAME, f"{snapshot}.{language}", split="train", streaming=True)


def write_sample(articles: Iterable[dict[str, Any]], count: int, output_path: Path) -> int:
    """Write non-empty records in the JSONL shape accepted by kg-gen curate."""
    if count <= 0:
        raise ValueError("--count must be greater than zero.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(output_path, "w", encoding="utf-8") as handle:
        for article in articles:
            text = article.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            record = {
                "id": str(article["id"]),
                "text": text,
                "title": article.get("title", ""),
                "url": article.get("url", ""),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            if written == count:
                break
    return written


def write_manifest(path: Path, language: str, snapshot: str, count: int) -> None:
    """Write the provenance manifest required by the curation command."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join((
            f"dataset_name: wikimedia-wikipedia-{language}-sample",
            f"version: {snapshot}-{language}-{count}",
            f"license: {LICENSE}",
            f"source: {DATASET_URL}",
            f"language: {language}",
            f"collection_date: '{snapshot[:4]}-{snapshot[4:6]}-{snapshot[6:]}'",
            "",
        )),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    output_path = args.output.resolve()
    manifest_path = args.manifest_output or output_path.with_name(f"{output_path.stem}_manifest.yaml")
    written = write_sample(stream_articles(args.snapshot, args.language), args.count, output_path)
    if written != args.count:
        raise RuntimeError(f"Only found {written} non-empty articles; expected {args.count}.")
    write_manifest(manifest_path, args.language, args.snapshot, written)
    print(f"Wrote {written} {args.language} articles to {output_path}")
    print(f"Wrote source manifest to {manifest_path}")


if __name__ == "__main__":
    main()
