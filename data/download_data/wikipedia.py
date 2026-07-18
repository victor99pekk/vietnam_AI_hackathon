#!/usr/bin/env python3
"""Stream a fixed-size English or Vietnamese Wikipedia sample to pipeline-ready JSONL."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable

DATASET_NAME = "wikimedia/wikipedia"
DATASET_URL = "https://huggingface.co/datasets/wikimedia/wikipedia"
LICENSE = "CC-BY-SA-3.0 AND GFDL"
DEFAULT_SNAPSHOT = "20231101"


VIETNAM_KEYWORDS = [
    # Địa danh
    "Việt Nam", "Việt", "Hà Nội", "Hồ Chí Minh", "Sài Gòn",
    "Huế", "Đà Nẵng", "Hải Phòng", "Cần Thơ", "Nha Trang",
    "Đà Lạt", "Vũng Tàu", "Buôn Ma Thuột", "Biên Hòa",
    "Mekong", "Cửu Long", "Sông Hồng", "sông Mê Kông",
    "Đồng bằng sông Cửu Long", "Tây Nguyên", "Tây Bắc",
    "Đông Nam Bộ", "Bắc Trung Bộ", "Vịnh Hạ Long",
    "Phú Quốc", "Côn Đảo", "Trường Sa", "Hoàng Sa",
    "Sa Pa", "Mộc Châu", "Tam Đảo", "Bà Nà",
    # Lịch sử & Chính trị
    "triều đại", "nhà Nguyễn", "nhà Trần", "nhà Lê",
    "nhà Lý", "nhà Đinh", "nhà Hồ", "Hùng Vương",
    "Đại Việt", "Văn Lang", "Âu Lạc", "Chăm Pa",
    "kháng chiến", "cách mạng", "độc lập", "thống nhất",
    "cộng hòa", "xã hội chủ nghĩa", "Đảng Cộng sản",
    "Quốc hội", "Chính phủ", "Bộ Chính trị",
    "Hồ Chí Minh", "Võ Nguyên Giáp", "Phan Bội Châu",
    "Nguyễn Trãi", "Trần Hưng Đạo", "Quang Trung",
    # Văn hóa & Xã hội
    "dân tộc", "người Việt", "người Kinh", "dân tộc thiểu số",
    "Tết Nguyên Đán", "Tết Trung Thu", "lễ hội",
    "phở", "bánh mì", "nước mắm", "cà phê",
    "áo dài", "nón lá", "đàn bầu", "đàn tranh",
    "chùa", "đền", "đình", "miếu",
    "Phật giáo", "Cao Đài", "Hòa Hảo", "tín ngưỡng",
    "thơ", "văn học", "truyện Kiều", "Nguyễn Du",
    "ca trù", "quan họ", "chèo", "tuồng", "cải lương",
    # Kinh tế & Địa lý
    "nông nghiệp", "lúa gạo", "cà phê Việt", "hồ tiêu",
    "thủy sản", "nuôi trồng", "đánh bắt",
    "du lịch", "xuất khẩu", "kinh tế Việt",
    "VND", "đồng Việt Nam", "ASEAN", "Đông Nam Á",
    "biển Đông", "bờ biển", "rừng nhiệt đới", "đất nước",
    # Khoa học & Giáo dục
    "đại học", "trường đại học", "giáo dục Việt",
    "khoa học", "nhà khoa học", "nghiên cứu",
]


# Compile once at import time — single-pass regex instead of O(n) substring scans
_VIETNAM_PATTERN = re.compile(
    "|".join(re.escape(kw) for kw in VIETNAM_KEYWORDS),
    re.IGNORECASE,
)


def is_vietnam_related(text: str, title: str = "") -> bool:
    """Check if article text or title mentions Vietnam-related terms."""
    check = f"{title} {text[:2000]}"  # Title + first 2000 chars
    return bool(_VIETNAM_PATTERN.search(check))


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
    parser.add_argument(
        "--vietnam-only",
        action="store_true",
        help="Only keep articles that mention Vietnam-related terms.",
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


def write_sample(
    articles: Iterable[dict[str, Any]],
    count: int,
    output_path: Path,
    vietnam_only: bool = False,
) -> int:
    """Write non-empty records in the JSONL shape accepted by kg-gen curate."""
    if count <= 0:
        raise ValueError("--count must be greater than zero.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    with open(output_path, "w", encoding="utf-8") as handle:
        for article in articles:
            text = article.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            title = article.get("title", "")
            if vietnam_only and not is_vietnam_related(text, title):
                skipped += 1
                continue
            record = {
                "id": str(article["id"]),
                "text": text,
                "title": title,
                "url": article.get("url", ""),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            if written == count:
                break
    if vietnam_only and skipped > 0:
        print(f"  Filtered out {skipped} non-Vietnam articles (scanned {written + skipped} total)")
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
    if args.vietnam_only:
        print(f"Filtering for Vietnam-related articles (language={args.language})...")
    written = write_sample(stream_articles(args.snapshot, args.language), args.count, output_path, vietnam_only=args.vietnam_only)
    if written != args.count:
        raise RuntimeError(f"Only found {written} non-empty articles; expected {args.count}.")
    write_manifest(manifest_path, args.language, args.snapshot, written)
    print(f"Wrote {written} {args.language} articles to {output_path}")
    print(f"Wrote source manifest to {manifest_path}")


if __name__ == "__main__":
    main()
