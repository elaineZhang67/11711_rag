from __future__ import annotations
import re
import unicodedata
from bs4 import BeautifulSoup  # type: ignore
from pathlib import Path

REMOVAL_TAGS = {
    "script",
    "style",
    "noscript",
    "svg",
    "canvas",
    "form",
    "button",
    "footer",
    "nav",
    "aside",
}


def _normalize_whitespace(text) :
    return re.sub(r"\s+", " ", text).strip()


def _clean_text_basic(text) :
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _remove_noise(soup) :
    for tag in soup.find_all(REMOVAL_TAGS):
        tag.decompose()
    for tag in soup.find_all(attrs={"aria-hidden": "true"}):
        tag.decompose()
    for cls in [
        "reference",
        "mw-editsection",
        "navbox",
        "sidebar",
        "infobox",
        "toc",
        "metadata",
    ]:
        for tag in soup.select(f".{cls}"):
            tag.decompose()


def _best_content_root(soup):
    # Site-specific selectors first, then generic fallbacks.
    selectors = [
        "main",
        "article",
        "#mw-content-text",
        ".mw-parser-output",
        "#content",
        ".content",
    ]
    for sel in selectors:
        node = soup.select_one(sel)
        if node is not None:
            return node
    return soup.body or soup


def extract_html_document(path, source_url= None) :
    p = Path(path)
    raw_html = p.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(raw_html, "html.parser")
    _remove_noise(soup)
    root = _best_content_root(soup)

    title = None
    if soup.title and soup.title.string:
        title = _normalize_whitespace(soup.title.string)
    if not title:
        h1 = root.find(["h1", "h2"]) if root else None
        title = _normalize_whitespace(h1.get_text(" ", strip=True)) if h1 else p.stem

    lines = []
    # Preserve a rough document structure for better chunking and retrieval.
    for node in root.find_all(["h1", "h2", "h3", "h4", "p", "li", "th", "td"]):
        text = node.get_text(" ", strip=True)
        text = _normalize_whitespace(text)
        if not text:
            continue
        if len(text) <= 1:
            continue
        lines.append(text)

    # Fallback if structure extraction is too aggressive.
    if len(lines) < 5:
        text = root.get_text("\n", strip=True)
    else:
        text = "\n".join(lines)

    text = _clean_text_basic(text)
    return {
        "title": title,
        "text": text,
        "source_path": str(p),
        "source_url": source_url,
        "doc_type": "html",
        "metadata": {"parser": "bs4"},
    }
