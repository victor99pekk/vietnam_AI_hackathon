#!/usr/bin/env python3
"""Legally-compliant Vietnamese web scraper for LLM dataset collection.

Politely discovers and fetches web pages via sitemaps or controlled crawling,
outputting pipeline-ready JSONL with full provenance metadata.

Usage:
    python data/download_data/scraper.py \
        --seed-file data/download_data/seeds/vietnamese_sources.txt \
        --max-pages 50 --language vi \
        --output data/scraped/vn_web_sample/
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

# trafilatura is imported lazily in _get_trafilatura() to avoid
# import-time errors when the optional dependency is missing.
_HAS_TRAFILATURA: bool | None = None  # None = not yet checked


def _get_trafilatura():
    """Lazy-import trafilatura. Returns the module or None."""
    global _HAS_TRAFILATURA
    if _HAS_TRAFILATURA is None:
        try:
            import trafilatura as _tf
            _HAS_TRAFILATURA = True
            return _tf
        except ImportError:
            _HAS_TRAFILATURA = False
            return None
    if _HAS_TRAFILATURA:
        import trafilatura as _tf
        return _tf
    return None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s]: %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_BOT_NAME = "VN-LLM-Data-Collector/1.0"
DEFAULT_CONTACT = "info@yourdomain.vn"
DEFAULT_DELAY = 2.0
DEFAULT_MAX_PAGES = 50
DEFAULT_MAX_TIME = 600  # 10 minutes
DEFAULT_DEPTH = 1
DEFAULT_LANGUAGE = "vi"
DEFAULT_DISCOVERY = "auto"
DEFAULT_SITEMAP_TIMEOUT = 30  # seconds before giving up on sitemap
DEFAULT_LICENSE = "Public / Legally Published Exception (Article 37a/Decree 134)"

# Article URL patterns — only crawl paths matching these regexes.
# Empty list means accept all same-domain paths.
_ARTICLE_PATTERNS: dict[str, str] = {
    # tuoitre.vn: article detail pages have section/slug.htm structure
    # e.g. /kinh-doanh/gia-vang-sjc-tang-20260718.htm
    # Section-only pages like /kinh-doanh.htm are skipped
    "tuoitre.vn": r"/[a-z-]+/[a-z0-9-]+\.htm$",
    # vnexpress.net: article URLs end with -NNNNNNN.html (numeric ID)
    # Also matches /tin-tuc-24h which is their news feed
    "vnexpress.net": r"/(?:tin-tuc-24h|[a-z-]+/[a-z0-9-]+-\d+\.html)",
    # mof.gov.vn: any path with content
    "mof.gov.vn": r"/(?:tin-tuc|van-ban|thong-tin|hoi-dap|.*)",
}


# ---------------------------------------------------------------------------
# URL discovery strategies
# ---------------------------------------------------------------------------

def discover_via_sitemap(
    base_url: str,
    path_prefix: str | None,
    rp: RobotFileParser,
    max_pages: int,
    timeout: int = DEFAULT_SITEMAP_TIMEOUT,
) -> list[str]:
    """Discover URLs from /sitemap.xml and /sitemap_index.xml, filtered by path_prefix.

    Respects robots.txt for each discovered URL before returning it.
    Has a hard timeout to prevent hanging on massive sitemaps.
    """
    parsed = urlparse(base_url)
    domain = f"{parsed.scheme}://{parsed.netloc}"
    found: list[str] = []

    # Check robots.txt for sitemap references first
    sitemaps = rp.site_maps() or []

    start = time.monotonic()
    for sm_url in sitemaps:
        if time.monotonic() - start > timeout:
            logger.warning(f"Sitemap discovery timed out after {timeout}s for {domain}")
            break
        _collect_from_sitemap(sm_url, domain, path_prefix, rp, found, max_pages, timeout)
        if len(found) >= max_pages:
            break

    # Fallback: try the standard location (with tight timeout)
    if not found and not sitemaps:
        for candidate in (f"{domain}/sitemap.xml", f"{domain}/sitemap_index.xml"):
            if time.monotonic() - start > timeout:
                break
            _collect_from_sitemap(candidate, domain, path_prefix, rp, found, max_pages, timeout)
            if found:
                break

    logger.info(f"Sitemap discovery found {len(found)} URLs for {domain}")
    return found[:max_pages]


def _collect_from_sitemap(
    sm_url: str,
    domain: str,
    path_prefix: str | None,
    rp: RobotFileParser,
    found: list[str],
    max_pages: int,
    timeout: int = DEFAULT_SITEMAP_TIMEOUT,
    _start: float | None = None,
) -> None:
    """Parse a single sitemap (or sitemap index) and add matching URLs."""
    if _start is None:
        _start = time.monotonic()
    if time.monotonic() - _start > timeout:
        return
    try:
        resp = requests.get(sm_url, timeout=15)
        if resp.status_code != 200:
            return
        soup = BeautifulSoup(resp.content, "xml")

        # Handle sitemap index (recursively)
        if time.monotonic() - _start > timeout:
            return
        sitemap_tags = soup.find_all("sitemap")
        if sitemap_tags:
            for sm in sitemap_tags:
                if len(found) >= max_pages or time.monotonic() - _start > timeout:
                    return
                loc = sm.find("loc")
                if loc and loc.text:
                    _collect_from_sitemap(loc.text.strip(), domain, path_prefix, rp, found, max_pages, timeout, _start)
            return

        # Regular sitemap: collect <url><loc> entries
        for url_tag in soup.find_all("url"):
            if len(found) >= max_pages or time.monotonic() - _start > timeout:
                return
            loc = url_tag.find("loc")
            if not loc or not loc.text:
                continue
            url = loc.text.strip()
            if not url.startswith(domain):
                continue
            if path_prefix and not url.startswith(path_prefix):
                continue
            if rp.can_fetch(DEFAULT_BOT_NAME, url) and rp.can_fetch("*", url):
                found.append(url)
    except Exception:
        logger.debug(f"Could not parse sitemap: {sm_url}", exc_info=True)


def discover_via_crawl(
    seed_url: str,
    path_prefix: str | None,
    rp: RobotFileParser,
    max_pages: int,
    max_depth: int,
    delay: float,
    headers: dict[str, str],
) -> list[str]:
    """Breadth-first crawl: follow <a href> links within the same domain.

    Filters discovered URLs against _ARTICLE_PATTERNS if a pattern is defined
    for the domain. This prevents wasting page quota on section listings,
    navigation pages, and other non-content URLs.
    """
    parsed = urlparse(seed_url)
    domain_url = f"{parsed.scheme}://{parsed.netloc}"
    domain_name = parsed.netloc.lower()

    # Check if we have an article pattern for this domain
    article_pattern: re.Pattern[str] | None = None
    for key, pat in _ARTICLE_PATTERNS.items():
        if key in domain_name:
            article_pattern = re.compile(pat)
            logger.info(f"Using article URL pattern for {domain_name}: {pat}")
            break

    visited: set[str] = set()
    to_visit: list[tuple[str, int]] = [(seed_url, 0)]
    found: list[str] = []

    try:
        crawl_delay = rp.crawl_delay(DEFAULT_BOT_NAME) or delay
    except Exception:
        crawl_delay = delay

    while to_visit and len(found) < max_pages:
        url, depth = to_visit.pop(0)
        if url in visited:
            continue
        visited.add(url)

        # Check path prefix
        if path_prefix and not url.startswith(path_prefix):
            continue

        # Robots.txt check
        if not rp.can_fetch(DEFAULT_BOT_NAME, url) or not rp.can_fetch("*", url):
            logger.debug(f"Skipping {url} — blocked by robots.txt")
            continue

        # Article pattern filter: if we have a pattern, skip non-matching URLs
        # Exception: always accept the seed URL (depth 0)
        if article_pattern and depth > 0 and not article_pattern.search(url):
            logger.debug(f"Skipping {url} — doesn't match article pattern")
            continue

        found.append(url)

        # Only discover more links if within depth
        if depth >= max_depth:
            continue

        # Fetch page to extract links
        try:
            time.sleep(crawl_delay)
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code != 200:
                continue
            soup = BeautifulSoup(resp.text, "html.parser")
            for link in soup.find_all("a", href=True):
                href = link["href"]
                absolute = urljoin(url, href)
                parsed_link = urlparse(absolute)
                if parsed_link.netloc != parsed.netloc:
                    continue  # same-domain only
                # Normalize: strip fragment
                clean = f"{parsed_link.scheme}://{parsed_link.netloc}{parsed_link.path}"
                if parsed_link.query:
                    clean += f"?{parsed_link.query}"
                if clean not in visited:
                    to_visit.append((clean, depth + 1))
        except Exception:
            logger.debug(f"Error crawling {url} for links", exc_info=True)

    logger.info(f"Crawl discovery found {len(found)} URLs from {seed_url}")
    return found[:max_pages]


# ---------------------------------------------------------------------------
# HTML-to-text extraction
# ---------------------------------------------------------------------------

# Minimal tag stripping: only tags that NEVER contain real content.
# Sites like chinhphu.vn (ASP.NET WebForms) wrap everything in <form>.
# Semantic tags like <header>/<nav>/<footer> often contain core layout.
# Batch-level line dedup handles the actual boilerplate removal.
_BOILERPLATE_TAGS = ["script", "style"]
_BOILERPLATE_ROLES: list[str] = []  # role-based stripping disabled — too fragile

# Known boilerplate text patterns that repeat across government pages (regex).
# Only patterns that NEVER carry unique content — batch dedup handles the rest.
_BOILERPLATE_PATTERNS: list[re.Pattern[str]] = [
    # chinhphu.vn: weather/city temperature widget (numbers vary but pattern is fixed)
    re.compile(
        r"(?:Hà Nội|Lai Châu|Điện Biên|Sơn La|Phú Thọ|Lào Cai|Tuyên Quang|Thái Nguyên|"
        r"Cao Bằng|Lạng Sơn|Quảng Ninh|Bắc Ninh|Tp Hải Phòng|Tp Huế|Tp Đà Nẵng|"
        r"Tp Hồ Chí Minh|Tp Cần Thơ)\s*\d{2}°\s*-\s*\d{2}°"
    ),
    # chinhphu.vn: copyright footer
    re.compile(r"©\s*Cục Thông tin và Truyền thông Chính phủ\s*Cục trưởng:.*?(?=\n\n|\Z)", re.DOTALL),
    re.compile(r"Ghi rõ nguồn\s*'Cổng Thông tin điện tử Chính phủ'.+", re.DOTALL),
    # RadEditor placeholder text (form widget, never content)
    re.compile(r"RadEditor\s*-\s*HTML WYSIWYG Editor\..*?(?=RadEditor's Modules)", re.DOTALL),
    re.compile(r"RadEditor's Modules\s*-.*?(?=\n\n|\Z)", re.DOTALL),
    re.compile(r"RadEditor\s*-\s*please enable JavaScript.*?(?=\n|$)", re.DOTALL),
    # Generic "English" language toggle on every page
    re.compile(r"^\s*English\s*$", re.MULTILINE),
    # Breadcrumb-like repeated page furniture
    re.compile(r"^\s*(?:Trang chủ|Báo Điện tử Chính Phủ)\s*$", re.MULTILINE),
    # chinhphu.vn: "CỔNG DỊCH VỤ CÔNG QUỐC GIA" section header (repeats on many pages)
    re.compile(r"CỔNG\s*\nDỊCH VỤ CÔNG\s*\nQUỐC GIA.*?(?=Chính sách mới|\n\n|\Z)", re.DOTALL),
    # chinhphu.vn: "GÓP Ý DỰ THẢO" section with file downloads
    re.compile(r"GÓP Ý DỰ THẢO\s*\n.*?(?=\n\n[A-Z]|\Z)", re.DOTALL),
]


def _clean_boilerplate(text: str) -> str:
    """Remove known boilerplate text patterns from extracted content."""
    for pattern in _BOILERPLATE_PATTERNS:
        text = pattern.sub(" ", text)
    # Clean up resulting whitespace
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def _strip_common_domain_boilerplate(text: str) -> str:
    """Remove only very specific repeating boilerplate fragments.

    These are exact strings that appear on virtually every page of
    Vietnamese government portals as page furniture, never as content.
    Keep this list minimal — batch dedup handles the rest.
    """
    # Only remove these exact repeating page-furniture lines.
    # Do NOT remove content-bearing phrases.
    remove_exact_lines = {
        "BẢN ĐỒ HÀNH CHÍNH",
        "Các trang thành phần",
        "Tiếng chuông",
        "English 中文",
        "English",
        "Trang chủ Báo Điện tử Chính Phủ",
        "© Cục Thông tin và Truyền thông Chính phủ",
        "Tài liệu đính kèm",
        "TẬP TIN",
        # Additional chinhphu.vn furniture
        "CỔNG DỊCH VỤ CÔNG QUỐC GIA",
        "Dịch vụ công trực tuyến",
        "Thủ tục hành chính",
        "Phản ánh kiến nghị",
        "Xây dựng chính sách",
        "GÓP Ý DỰ THẢO",
        "Tải xuống",
        "Xem toàn văn",
        "Xem các góp ý",
        "Gửi góp ý",
        "Gửi phản ánh, kiến nghị",
        "Tham vấn chính sách",
        "Chính sách mới",
        "THÔNG CÁO BÁO CHÍ",
        "CÁC TRANG THÀNH PHẦN",
        "Báo Điện tử Chính Phủ",
        "Cổng Thông tin điện tử Chính phủ",
    }
    # Lines that start with these prefixes are navigation/furniture
    remove_prefixes = (
        "Số ký hiệu",
        "Ngày ban hành",
        "Trích yếu",
        "/CĐ-TTg",
        "/VPCP-",
        "/QĐ-TTg",
        "/NĐ-CP",
        "/NQ-CP",
        "/CT-TTg",
        "/TB-VPCP",
        "/QĐ-UBND",
        "/BTC-",
    )
    lines = text.split("\n")
    keep = []
    for line in lines:
        stripped = line.strip()
        if len(stripped) < 3:
            continue
        if stripped in remove_exact_lines:
            continue
        # Skip legal document reference lines (table rows from "Hệ thống văn bản")
        if any(stripped.startswith(p) for p in remove_prefixes):
            continue
        # Skip lines that are just a date pattern like "17/07/2026"
        if re.match(r"^\d{2}/\d{2}/\d{4}$", stripped):
            continue
        # Skip bare pagination numbers (single integer on its own line)
        if re.match(r"^\d{2,4}$", stripped):
            continue
        keep.append(line)
    result = "\n".join(keep)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def _compute_unique_ratio(text: str) -> float:
    """Ratio of unique lines to total lines. Low ratio = heavy boilerplate."""
    lines = [l.strip() for l in text.split("\n") if len(l.strip()) > 5]
    if not lines:
        return 0.0
    return len(set(lines)) / len(lines)


class TrafilaturaExtractor:
    """Use trafilatura for robust main-content extraction.

    trafilatura handles boilerplate removal, navigation stripping, and
    content deduplication far better than raw BeautifulSoup extraction.
    Falls back to GenericExtractor when trafilatura finds no content.
    """

    def extract(self, html: str, url: str) -> tuple[str, str]:
        """Return (title, cleaned_text) using trafilatura."""
        tf = _get_trafilatura()
        if tf is None:
            # Fall back to generic BeautifulSoup extraction
            return GenericExtractor().extract(html, url)

        # Extract with trafilatura — include formatting for readability
        text = tf.extract(
            html,
            include_comments=False,
            include_tables=True,
            include_links=False,
            include_images=False,
            output_format="txt",
            favor_recall=True,  # get more content, dedup handles the rest
        )

        # Extract title
        title = ""
        extracted_title = tf.extract(html, output_format="txt",
                                      include_comments=False,
                                      include_tables=False,
                                      include_links=False,
                                      include_images=False,
                                      favor_recall=False)
        if extracted_title:
            # Use first meaningful line as title
            lines = [l.strip() for l in extracted_title.split("\n") if len(l.strip()) > 10]
            if lines:
                title = lines[0][:200]

        if not text or len(text) < 100:
            # trafilatura couldn't find content — try BeautifulSoup
            return GenericExtractor().extract(html, url)

        # Clean up whitespace
        text = re.sub(r"[ \t\f\v]+", " ", text)
        text = re.sub(r" *\n *", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = _strip_common_domain_boilerplate(text)
        text = _clean_boilerplate(text)

        return title, text.strip()


class GenericExtractor:
    """Strip standard boilerplate tags by element type and ARIA role.

    Relies on batch-level line dedup to remove cross-page boilerplate
    (navigation menus, weather widgets, footer text) rather than
    fragile CSS-class-based stripping that varies per site.
    """

    def _strip_boilerplate_tags(self, soup: BeautifulSoup) -> None:
        """Remove known boilerplate elements from the soup in-place."""
        for tag_name in _BOILERPLATE_TAGS:
            for tag in soup.find_all(tag_name):
                tag.decompose()
        for role in _BOILERPLATE_ROLES:
            for tag in soup.find_all(attrs={"role": role}):
                tag.decompose()

    def extract(self, html: str, url: str) -> tuple[str, str]:
        """Return (title, cleaned_text) from HTML."""
        soup = BeautifulSoup(html, "html.parser")

        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()

        self._strip_boilerplate_tags(soup)

        text = soup.get_text(separator="\n")
        text = re.sub(r"[ \t\f\v]+", " ", text)
        text = re.sub(r" *\n *", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Remove known boilerplate patterns (weather, copyright, form widgets)
        text = _strip_common_domain_boilerplate(text)
        text = _clean_boilerplate(text)

        return title, text


class ChinhphuVnExtractor(GenericExtractor):
    """Extractor tuned for chinhphu.vn — uses trafilatura when available.

    Falls back to targeted CSS-selector extraction for article pages
    and generic extraction for listing pages.
    """

    # chinhphu.vn specific: the unique article text lives in these containers
    _CONTENT_SELECTORS = [
        "div#divPrint",              # printable version — often cleanest
        "div.detail-content",        # article detail pages
        "div.news-detail",           # news article pages
        "div.post-content",          # generic post
        "div.article-content",       # article body
        "div#ctl00_mainContent",     # ASP.NET main content area
        "div.main-content",          # generic main content
        "div.wrap-content-body",     # chinhphu content wrapper
        "div.detail",                # article detail
        "main", "article",
    ]

    # CSS selectors for navigation/sidebar elements to strip
    _NAV_STRIP_SELECTORS = [
        "nav", "header", "footer",
        "div.menu", "div.nav", "div.sidebar",
        "div#sidebar", "div#left", "div#right",
        "div[class*='menu']", "div[class*='nav']",
        "div[class*='sidebar']", "div[class*='footer']",
        "div[class*='header']",
        "ul.menu", "ul.nav",
    ]

    def extract(self, html: str, url: str) -> tuple[str, str]:
        # Use site-specific selectors for chinhphu.vn — trafilatura can
        # misidentify content on ASP.NET listing pages.
        soup = BeautifulSoup(html, "html.parser")

        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()

        # Strip navigation/sidebar first
        for sel in self._NAV_STRIP_SELECTORS:
            for tag in soup.select(sel):
                tag.decompose()

        # Try to find a content container; if none, extract from whole page
        content_area = None
        for sel in self._CONTENT_SELECTORS:
            content_area = soup.select_one(sel)
            if content_area:
                break

        target = content_area if content_area else soup
        self._strip_boilerplate_tags(target)

        # chinhphu.vn specific: strip weather widget containers by class
        for bad in target.select(
            "[class*='weather'], [class*='Weather'], "
            "[class*='temperature'], [class*='nhiet-do']"
        ):
            bad.decompose()

        text = target.get_text(separator="\n")
        text = re.sub(r"[ \t\f\v]+", " ", text)
        text = re.sub(r" *\n *", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        text = _strip_common_domain_boilerplate(text)
        text = _clean_boilerplate(text)

        return title, text


class DataGovVnExtractor(GenericExtractor):
    """Extractor for data.gov.vn — dataset listing pages."""

    _CONTENT_SELECTORS = [
        "div.dataset-content", "div#content",
        "div.module-content", "main", "article",
    ]

    def extract(self, html: str, url: str) -> tuple[str, str]:
        soup = BeautifulSoup(html, "html.parser")
        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()

        content_area = None
        for sel in self._CONTENT_SELECTORS:
            content_area = soup.select_one(sel)
            if content_area:
                break

        target = content_area if content_area else soup
        self._strip_boilerplate_tags(target)
        text = target.get_text(separator="\n")
        text = re.sub(r"[ \t\f\v]+", " ", text)
        text = re.sub(r" *\n *", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = _strip_common_domain_boilerplate(text)
        text = _clean_boilerplate(text)

        return title, text


class VietnameseExtractor(GenericExtractor):
    """Vietnamese-aware extractor — prefers trafilatura, with underthesea fallback."""

    def extract(self, html: str, url: str) -> tuple[str, str]:
        # Prefer trafilatura for best content extraction
        if _get_trafilatura() is not None:
            title, text = TrafilaturaExtractor().extract(html, url)
            if text and len(text) >= 100:
                # Apply Vietnamese sentence segmentation for cleaner text
                try:
                    from underthesea import sent_tokenize
                    sentences = sent_tokenize(text)
                    text = " ".join(sentences)
                except ImportError:
                    pass
                return title, text

        # Fall back to BeautifulSoup extraction
        title, text = super().extract(html, url)
        try:
            from underthesea import sent_tokenize
            sentences = sent_tokenize(text)
            text = " ".join(sentences)
        except ImportError:
            logger.debug("underthesea not installed — using raw text extraction")
        return title, text


def _get_extractor(language: str, url: str = "") -> GenericExtractor:
    """Pick the best extractor for the given URL and language.

    Precedence:
    1. Domain-specific extractors (best for known sites)
    2. Trafilatura for any other URL when available
    3. Language-based fallback
    """
    domain = urlparse(url).netloc.lower() if url else ""

    # Domain-specific extractors take priority — they know page structure
    if "chinhphu.vn" in domain:
        return ChinhphuVnExtractor()
    if "data.gov.vn" in domain:
        return DataGovVnExtractor()

    # Trafilatura is the best generic extractor for unknown domains
    if _get_trafilatura() is not None:
        return TrafilaturaExtractor()  # type: ignore[return-value]

    # Language-based fallback
    if language == "vi":
        return VietnameseExtractor()
    return GenericExtractor()


def _batch_dedup_lines(records: list[dict[str, Any]], threshold: float = 0.33) -> list[dict[str, Any]]:
    """Remove lines that appear in more than `threshold` fraction of pages.

    This catches navigation text, weather widgets, and other boilerplate
    that repeats across many pages of the same site.
    Also detects and removes repeating block patterns (e.g. pagination numbers,
    ministry lists, province lists).
    """
    if len(records) < 3:
        return records

    # Build line frequency map
    line_counts: dict[str, int] = {}
    for r in records:
        seen_in_this_doc: set[str] = set()
        for line in r["record"]["text"].split("\n"):
            stripped = line.strip()
            if len(stripped) < 4:
                continue
            if stripped not in seen_in_this_doc:
                line_counts[stripped] = line_counts.get(stripped, 0) + 1
                seen_in_this_doc.add(stripped)

    # Find lines that appear in too many pages
    cutoff = max(2, int(len(records) * threshold))
    common_lines = {line for line, count in line_counts.items() if count >= cutoff}

    # Also detect: lines that are just ministry names, province names, etc.
    # These appear across many gov pages and are navigation, not content
    known_boilerplate_starts = (
        "Bộ ", "Bộ ", "Tỉnh ", "Thành phố ", "Tập đoàn ", "Tổng công ty ",
        "Tổng Công ty ", "Công ty TNHH ", "Trung tâm ", "Đoạn ",
        "Phó Thủ tướng ", "Thủ tướng ", "Lĩnh vực ",
    )
    for line, count in list(line_counts.items()):
        if count >= cutoff:
            continue  # already captured
        if count >= 2 and any(line.startswith(p) for p in known_boilerplate_starts):
            common_lines.add(line)

    if common_lines:
        logger.info(f"Found {len(common_lines)} boilerplate lines (appear in ≥{cutoff}/{len(records)} pages)")

    # Filter them out
    for r in records:
        lines = r["record"]["text"].split("\n")
        filtered = [l for l in lines if l.strip() not in common_lines]
        r["record"]["text"] = "\n".join(filtered)
        # Update content hash and length
        r["record"]["content_hash"] = hashlib.sha256(
            r["record"]["text"].encode("utf-8")
        ).hexdigest()
        r["record"]["content_length_chars"] = len(r["record"]["text"])

    return records


def _infer_content_type(url: str, title: str, text: str) -> str:
    """Infer a coarse content type from URL structure and text signals.

    Heuristics:
      - Very short text (<200 chars) → 'landing_page'
      - URL contains /dataset/, /data/, /open-data/ → 'dataset'
      - URL contains /thu-tuc/, /van-ban/, /law/, /legal/ → 'legal_document'
      - URL contains /news/, /tin-tuc/, /article/, /bai-viet/ → 'article'
      - Text contains many Vietnamese government keywords → 'government_document'
      - Otherwise → 'web_page'
    """
    text_lower = text.lower()
    url_lower = url.lower()

    # Short pages are usually landing/navigation
    if len(text) < 200:
        return "landing_page"

    # URL-based signals
    dataset_signals = ["/dataset", "/data/", "/open-data", "/du-lieu", "/so-lieu"]
    legal_signals = ["/thu-tuc", "/van-ban", "/law/", "/legal", "/nghi-dinh", "/thong-tu", "/quyet-dinh"]
    article_signals = ["/news/", "/tin-tuc", "/tin-", "/article", "/bai-viet", "/thoi-su", "/xa-hoi"]

    if any(s in url_lower for s in dataset_signals):
        return "dataset"
    if any(s in url_lower for s in legal_signals):
        return "legal_document"
    if any(s in url_lower for s in article_signals):
        return "article"

    # Content-based signals for government documents
    gov_keywords = ["chính phủ", "thủ tướng", "bộ trưởng", "quốc hội",
                    "nghị định", "thông tư", "quyết định", "công văn"]
    gov_hits = sum(1 for kw in gov_keywords if kw in text_lower)
    if gov_hits >= 3:
        return "government_document"

    return "web_page"


# ---------------------------------------------------------------------------
# Core scraper
# ---------------------------------------------------------------------------

class LegallyCompliantScraper:
    """Ethical scraper: robots.txt compliance, rate limiting, provenance logging."""

    def __init__(
        self,
        bot_name: str = DEFAULT_BOT_NAME,
        contact_info: str = DEFAULT_CONTACT,
        language: str = DEFAULT_LANGUAGE,
        delay: float = DEFAULT_DELAY,
    ) -> None:
        self.bot_name = bot_name
        self.language = language
        self.delay = delay
        self.headers = {
            "User-Agent": f"{self.bot_name} (+{contact_info})",
            "Accept-Language": f"{language},en;q=0.9",
        }
        self._robot_parsers: dict[str, RobotFileParser] = {}

    def _get_robot_parser(self, base_url: str) -> RobotFileParser:
        """Fetch and cache robots.txt for a domain."""
        if base_url in self._robot_parsers:
            return self._robot_parsers[base_url]

        rp = RobotFileParser()
        rp.set_url(f"{base_url}/robots.txt")
        try:
            resp = requests.get(f"{base_url}/robots.txt", headers=self.headers, timeout=5)
            if resp.status_code == 200:
                rp.parse(resp.text.splitlines())
                logger.info(f"Parsed robots.txt for {base_url}")
            else:
                rp.parse([])  # no robots.txt — assume open access
        except Exception:
            logger.warning(f"Could not read robots.txt for {base_url} — blocking by default")
            rp.parse(["User-agent: *", "Disallow: /"])

        self._robot_parsers[base_url] = rp
        return rp

    def discover_urls(
        self,
        seeds: list[str],
        discovery: str,
        path_prefix: str | None,
        max_pages: int,
        max_depth: int,
    ) -> list[str]:
        """Resolve seed entries into a list of concrete URLs to scrape."""
        all_urls: list[str] = []

        for seed in seeds:
            seed = seed.strip()
            if not seed:
                continue

            parsed = urlparse(seed)
            domain = f"{parsed.scheme}://{parsed.netloc}"
            rp = self._get_robot_parser(domain)

            # Determine if this is an exact URL or needs discovery
            has_path = bool(parsed.path and parsed.path not in ("", "/"))
            needs_discovery = not has_path  # domain-only → needs discovery

            if discovery == "exact" or (discovery == "auto" and not needs_discovery):
                # Treat as exact URL
                if rp.can_fetch(self.bot_name, seed) and rp.can_fetch("*", seed):
                    all_urls.append(seed)
                else:
                    logger.warning(f"Blocked by robots.txt: {seed}")
                continue

            # Discovery needed
            if discovery in ("sitemap", "auto"):
                sitemap_urls = discover_via_sitemap(
                    domain,
                    path_prefix or (seed if has_path else None),
                    rp, max_pages,
                )
                if sitemap_urls:
                    all_urls.extend(sitemap_urls)
                    if len(all_urls) >= max_pages:
                        break
                    continue  # sitemap succeeded, skip crawl for this seed

            if discovery in ("crawl", "auto"):
                # For crawl, only use explicit --path-prefix, never infer from seed.
                # Otherwise following links to other paths on the same domain
                # would be incorrectly blocked.
                crawl_urls = discover_via_crawl(
                    seed, path_prefix, rp, max_pages, max_depth, self.delay, self.headers,
                )
                all_urls.extend(crawl_urls)

            if len(all_urls) >= max_pages:
                break

        return all_urls[:max_pages]

    def verify_and_fetch(self, target_url: str, crawl_depth: int = 0, min_unique_chars: int = 100, min_unique_density: float = 0.01) -> dict[str, Any] | None:
        """Check robots.txt, fetch page, extract text with provenance.

        Args:
            target_url: URL to scrape.
            crawl_depth: Link depth from seed (0 = seed page).
            min_unique_chars: Skip pages with fewer unique characters after
                              boilerplate stripping. Default 100.
            min_unique_density: Skip pages where unique_chars / total_chars
                                falls below this ratio (catches boilerplate-heavy
                                listing pages). Default 0.01 (1%).
        """
        parsed = urlparse(target_url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"

        # 1. Robots.txt check
        rp = self._get_robot_parser(base_url)
        if not rp.can_fetch(self.bot_name, target_url) or not rp.can_fetch("*", target_url):
            logger.warning(f"🚫 Blocked by robots.txt: {target_url}")
            return None

        # 2. Polite fetch
        try:
            time.sleep(self.delay)
            resp = requests.get(target_url, headers=self.headers, timeout=10)

            # No paywall/auth bypass
            if resp.status_code in (401, 403):
                logger.warning(f"🔒 Access denied ({resp.status_code}): {target_url}")
                return None
            if resp.status_code != 200:
                logger.debug(f"Skipping {target_url} — HTTP {resp.status_code}")
                return None

            # 3. Extract with domain-aware extractor
            extractor = _get_extractor(self.language, target_url)
            title, text = extractor.extract(resp.text, target_url)

            # 4. Quality checks
            if not text or len(text) < 50:
                logger.debug(f"Skipping {target_url} — too little content ({len(text)} chars)")
                return None

            # Compute unique char count (after boilerplate stripping)
            unique_chars = len(set(text))
            if unique_chars < min_unique_chars:
                logger.debug(
                    f"Skipping {target_url} — only {unique_chars} unique chars "
                    f"(min: {min_unique_chars})"
                )
                return None

            # Unique density check: reject high-boilerplate pages
            # Pages like navigation/listings often have 10K+ chars but <1% unique
            unique_density = unique_chars / max(len(text), 1)
            if unique_density < min_unique_density:
                logger.debug(
                    f"Skipping {target_url} — unique density {unique_density:.3f} "
                    f"below minimum {min_unique_density}"
                )
                return None

            unique_ratio = _compute_unique_ratio(text)

            # 5. Compute metadata
            collected_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            url_hash = hashlib.sha256(target_url.encode()).hexdigest()[:12]
            inferred_type = _infer_content_type(target_url, title, text)
            content_length_chars = len(text)

            # 6. Build enriched record matching project JSONL schema
            record = {
                "id": f"{parsed.netloc.replace('.', '-')}-{url_hash}",
                "text": text,
                "title": title or target_url,
                "url": target_url,
                "scraped_at": collected_at,
                "source_domain": parsed.netloc,
                "license": DEFAULT_LICENSE,
                "crawler": self.bot_name,
                "content_hash": content_hash,
                "content_length_chars": content_length_chars,
                "http_status": resp.status_code,
                "crawl_depth": crawl_depth,
                "inferred_type": inferred_type,
                "unique_chars": unique_chars,
                "unique_line_ratio": round(unique_ratio, 3),
            }

            # 7. Provenance for audit CSV
            provenance = {
                "source_url": target_url,
                "domain": base_url,
                "collected_at": collected_at,
                "crawler_identity": self.bot_name,
                "license_asserted": DEFAULT_LICENSE,
                "http_status": resp.status_code,
                "content_length": content_length_chars,
                "content_hash": content_hash,
                "title": title,
                "crawl_depth": crawl_depth,
                "inferred_type": inferred_type,
                "unique_chars": unique_chars,
                "unique_line_ratio": round(unique_ratio, 3),
            }

            return {"record": record, "provenance": provenance}

        except Exception:
            logger.error(f"Error fetching {target_url}", exc_info=True)
            return None


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def write_jsonl(records: list[dict[str, Any]], path: Path) -> int:
    """Write records to JSONL. Returns count written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item["record"], ensure_ascii=False) + "\n")
    return len(records)


def write_audit_csv(records: list[dict[str, Any]], path: Path) -> None:
    """Write provenance audit trail."""
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_url", "domain", "collected_at", "crawler_identity",
        "license_asserted", "http_status", "content_length", "content_hash",
        "title", "crawl_depth", "inferred_type", "unique_chars", "unique_line_ratio",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in records:
            writer.writerow(item["provenance"])


def write_manifest(
    path: Path,
    dataset_name: str,
    version: str,
    license_str: str,
    source: str,
    language: str,
    record_count: int,
) -> None:
    """Write source manifest YAML matching the existing SourceManifest schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        f"dataset_name: {dataset_name}\n"
        f"version: {version}\n"
        f"license: {license_str}\n"
        f"source: {source}\n"
        f"language: {language}\n"
        f"collection_date: '{datetime.now(timezone.utc).strftime('%Y-%m-%d')}'\n"
        f"# record_count: {record_count}\n"
    )
    path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Input
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--seed-file", type=Path,
        help="Path to text file with URLs/domains/path-prefixes (one per line)",
    )
    input_group.add_argument(
        "--seed-urls", nargs="+",
        help="Inline URLs/domains/path-prefixes",
    )

    # Discovery
    parser.add_argument(
        "--discovery",
        choices=("exact", "sitemap", "crawl", "auto"),
        default=DEFAULT_DISCOVERY,
        help="URL discovery mode (default: auto — try sitemap, fall back to crawl)",
    )
    parser.add_argument(
        "--path-prefix",
        help="Only collect URLs starting with this prefix",
    )

    # Limits
    parser.add_argument("--max-pages", type=int, default=DEFAULT_MAX_PAGES, help="Max pages to scrape")
    parser.add_argument("--max-time", type=int, default=DEFAULT_MAX_TIME, help="Max wall-clock seconds (0 = no limit)")
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH, help="Crawl link depth (0 = seed only)")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY, help="Delay between requests in seconds")

    # Quality filters
    parser.add_argument("--min-unique-chars", type=int, default=120,
                        help="Skip pages with fewer unique characters after boilerplate stripping (default: 120)")
    parser.add_argument("--min-unique-density", type=float, default=0.015,
                        help="Skip pages where unique/total char ratio is below this (default: 0.015 = 1.5%%). Lower = more permissive.")
    parser.add_argument("--no-batch-dedup", action="store_true",
                        help="Disable cross-page line deduplication (keeps more content, but includes boilerplate)")

    # Output
    parser.add_argument("--output", type=Path, required=True, help="Output directory for JSONL + manifest + audit")
    parser.add_argument(
        "--manifest-output", type=Path,
        help="Optional explicit manifest path (default: <output>/<stem>_manifest.yaml)",
    )

    # Metadata
    parser.add_argument("--language", choices=("en", "vi"), default=DEFAULT_LANGUAGE, help="Language (default: vi)")
    parser.add_argument("--dataset-name", default="vietnamese-web-corpus", help="Dataset name for manifest")
    parser.add_argument("--version", default="v1", help="Dataset version")
    parser.add_argument("--license", default=DEFAULT_LICENSE, dest="license_str", help="License string for manifest")
    parser.add_argument("--contact-info", default=DEFAULT_CONTACT, help="Contact for User-Agent header")
    parser.add_argument("--bot-name", default=DEFAULT_BOT_NAME, help="Bot name for User-Agent header")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve seeds
    if args.seed_file:
        seeds = args.seed_file.read_text(encoding="utf-8").strip().splitlines()
    else:
        seeds = list(args.seed_urls)

    seeds = [s.strip() for s in seeds if s.strip() and not s.strip().startswith("#")]
    if not seeds:
        logger.error("No seeds provided.")
        sys.exit(1)

    # Output paths
    output_dir = args.output.resolve()
    stem = output_dir.name or "scraped"
    jsonl_path = output_dir / f"{stem}.jsonl"
    audit_path = output_dir / f"{stem}_audit.csv"
    manifest_path = args.manifest_output or output_dir / f"{stem}_manifest.yaml"

    # Build scraper
    scraper = LegallyCompliantScraper(
        bot_name=args.bot_name,
        contact_info=args.contact_info,
        language=args.language,
        delay=args.delay,
    )

    # Phase 1: Discover URLs
    logger.info(f"Discovering URLs from {len(seeds)} seed(s) (mode={args.discovery})...")
    urls = scraper.discover_urls(
        seeds=seeds,
        discovery=args.discovery,
        path_prefix=args.path_prefix,
        max_pages=args.max_pages,
        max_depth=args.depth,
    )
    logger.info(f"Discovered {len(urls)} URLs to scrape")

    # Phase 2: Scrape with time limit
    start_time = time.monotonic()
    results: list[dict[str, Any]] = []

    for i, url in enumerate(urls):
        # Check time limit
        elapsed = time.monotonic() - start_time
        if args.max_time > 0 and elapsed >= args.max_time:
            logger.info(f"⏰ Time limit reached ({args.max_time}s). Stopping after {len(results)} pages.")
            break

        logger.info(f"[{i+1}/{len(urls)}] {url}")
        result = scraper.verify_and_fetch(url, crawl_depth=0, min_unique_chars=args.min_unique_chars, min_unique_density=args.min_unique_density)
        if result:
            results.append(result)
            logger.info(f"  ✅ {len(result['record']['text'])} chars ({result['record']['unique_chars']} unique) — \"{result['record']['title'][:60]}\"")

    # Phase 3: Batch-level line deduplication (remove cross-page boilerplate)
    if not args.no_batch_dedup and len(results) > 2:
        before_total_chars = sum(len(r["record"]["text"]) for r in results)
        results = _batch_dedup_lines(results)
        after_total_chars = sum(len(r["record"]["text"]) for r in results)
        removed_chars = before_total_chars - after_total_chars
        logger.info(f"Batch dedup removed {removed_chars} chars of cross-page boilerplate ({(removed_chars / before_total_chars * 100):.1f}%)")

    # Phase 4: Write output
    written = write_jsonl(results, jsonl_path)
    write_audit_csv(results, audit_path)

    source_description = f"Web scrape from {len(seeds)} seed(s) via {args.discovery} discovery"
    write_manifest(
        manifest_path,
        dataset_name=args.dataset_name,
        version=args.version,
        license_str=args.license_str,
        source=source_description,
        language=args.language,
        record_count=written,
    )

    logger.info(f"✅ Done: {written} pages → {jsonl_path}")
    logger.info(f"   Manifest → {manifest_path}")
    logger.info(f"   Audit   → {audit_path}")


if __name__ == "__main__":
    main()
