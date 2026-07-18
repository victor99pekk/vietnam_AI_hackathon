"""LLM-enhanced processing for scraped web pages.

Two capabilities:
1. Article URL discovery — given a listing page, extract real article URLs
2. Content cleaning — given noisy page text, extract only the article body
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DeepSeek client (reuses project convention)
# ---------------------------------------------------------------------------

_DS_CLIENT: Any = None


def _get_client():
    global _DS_CLIENT
    if _DS_CLIENT is not None:
        return _DS_CLIENT

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY not set. Add it to .env")

    from openai import OpenAI

    _DS_CLIENT = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    return _DS_CLIENT


def _llm(prompt: str, max_tokens: int = 2048) -> str:
    client = _get_client()
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# 1. Article URL extraction
# ---------------------------------------------------------------------------

ARTICLE_DISCOVERY_PROMPT = """You are a web scraper assistant. Given the raw text of a Vietnamese
government news listing page (chinhphu.vn), extract ALL URLs that point to
**individual news articles** (not section pages, not navigation links).

Rules:
- Only include URLs that clearly point to a specific news article, announcement,
  or policy document
- Exclude: section landing pages (/chinh-phu, /cong-dan, /doanh-nghiep, etc.)
- Exclude: search pages, form pages, pagination links
- Exclude: URLs with query parameters like ?gmist=, ?city=, ?page=
- Return ONLY a JSON array of URL strings, nothing else

Page text:
{text}

Return format: ["url1", "url2", ...]"""


def discover_article_urls(page_text: str, max_urls: int = 30) -> list[str]:
    """Use LLM to find real article URLs in listing page text."""
    text = page_text[:8000]  # truncate to save tokens
    prompt = ARTICLE_DISCOVERY_PROMPT.format(text=text)
    response = _llm(prompt, max_tokens=1024)

    # Parse JSON array from response
    try:
        # Handle markdown code blocks
        if "```" in response:
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
        urls = json.loads(response.strip())
        if isinstance(urls, list):
            return [u for u in urls if isinstance(u, str) and u.startswith("http")][:max_urls]
    except json.JSONDecodeError:
        # Fallback: regex extract URLs
        urls = re.findall(r'https?://[^\s"\'\[\],]+', response)
        urls = [u.rstrip('"\'') for u in urls if "chinhphu.vn" in u]
        return urls[:max_urls]

    return []


# ---------------------------------------------------------------------------
# 2. Content cleaning
# ---------------------------------------------------------------------------

CONTENT_CLEAN_PROMPT = """You are a content cleaner for a Vietnamese news dataset. 
Given the raw extracted text from a web page, extract ONLY the main article body text.
Strip all navigation menus, sidebar content, footer text, legal document tables,
weather widgets, "Hệ thống văn bản" tables, ministry/province listings, 
pagination numbers, and repeated boilerplate.

Return ONLY the clean article text. Do not add any commentary, headers, or formatting.
Do not summarize or rewrite — preserve the original text exactly, just remove the noise.

Raw page text:
{text}"""


def clean_page_text(raw_text: str) -> str:
    """Use LLM to extract article body from noisy page text."""
    if len(raw_text) < 500:
        return raw_text  # too short to need cleaning

    # Truncate very long pages to save tokens
    text = raw_text[:10000]
    prompt = CONTENT_CLEAN_PROMPT.format(text=text)
    return _llm(prompt, max_tokens=4096)


# ---------------------------------------------------------------------------
# 3. Quality scoring
# ---------------------------------------------------------------------------

QUALITY_SCORE_PROMPT = """Score this Vietnamese web page text on a scale of 1-10 for quality
as training data for an LLM. Consider:

1. Information density: how much unique factual content per 100 words
2. Article completeness: is it a full article or just headlines/summaries
3. Noise ratio: how much boilerplate/navigation vs. real content
4. Language quality: proper Vietnamese, no encoding errors

Return ONLY a JSON object with these fields:
{{"score": <1-10>, "is_article": <true/false>, "topic": "<1-3 word topic>", "reason": "<one sentence>"}}

Page text (first 3000 chars):
{text}"""


def score_page_quality(text: str) -> dict[str, Any]:
    """Return quality assessment for a scraped page."""
    prompt = QUALITY_SCORE_PROMPT.format(text=text[:3000])
    response = _llm(prompt, max_tokens=256)
    try:
        if "```" in response:
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
        return json.loads(response.strip())
    except json.JSONDecodeError:
        return {"score": 0, "is_article": False, "topic": "unknown", "reason": "parse error"}


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def process_scraped_file(
    input_path: str,
    output_path: str,
    min_score: int = 5,
    clean: bool = True,
) -> dict[str, Any]:
    """Process a JSONL file: score, filter, and optionally clean pages.

    Returns stats dict.
    """
    with open(input_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    stats = {"total": len(records), "kept": 0, "cleaned": 0, "scores": []}

    kept = []
    for i, r in enumerate(records):
        logger.info(f"Scoring {i+1}/{len(records)}: {r.get('title', '')[:50]}...")
        quality = score_page_quality(r["text"])

        stats["scores"].append(quality.get("score", 0))

        if quality.get("score", 0) >= min_score:
            if clean:
                logger.info(f"  Cleaning (score={quality['score']})...")
                r["text"] = clean_page_text(r["text"])
                r["cleaned_by"] = "deepseek-chat"
                stats["cleaned"] += 1
            kept.append(r)
            stats["kept"] += 1
        else:
            logger.info(f"  Skipping (score={quality['score']}): {quality.get('reason', '')}")

    with open(output_path, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="LLM-enhanced web page processing")
    sub = parser.add_subparsers(dest="command")

    # discover
    disc = sub.add_parser("discover", help="Extract article URLs from listing page text")
    disc.add_argument("input_file", help="JSONL file with listing page text")
    disc.add_argument("--output", "-o", default="data/download_data/seeds/discovered_urls.txt")

    # clean
    cln = sub.add_parser("clean", help="Score, filter, and clean scraped pages")
    cln.add_argument("input_file", help="JSONL file to process")
    cln.add_argument("--output", "-o", required=True)
    cln.add_argument("--min-score", type=int, default=5)
    cln.add_argument("--no-clean", action="store_true", help="Only score/filter, don't clean text")

    args = parser.parse_args()

    if args.command == "discover":
        with open(args.input_file, encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]

        all_urls = []
        for r in records:
            urls = discover_article_urls(r["text"])
            logger.info(f"Found {len(urls)} URLs in: {r.get('title', '')[:50]}")
            all_urls.extend(urls)

        unique = list(dict.fromkeys(all_urls))  # preserve order, deduplicate
        with open(args.output, "w") as f:
            f.write("\n".join(unique))
        logger.info(f"Wrote {len(unique)} unique URLs to {args.output}")

    elif args.command == "clean":
        stats = process_scraped_file(
            args.input_file, args.output,
            min_score=args.min_score,
            clean=not args.no_clean,
        )
        avg_score = sum(stats["scores"]) / len(stats["scores"]) if stats["scores"] else 0
        print(f"\nDone: {stats['kept']}/{stats['total']} kept, {stats['cleaned']} cleaned")
        print(f"Avg score: {avg_score:.1f}/10")
