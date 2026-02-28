from __future__ import annotations

import hashlib
import json
import mimetypes
import re
import time
from collections import deque
from pathlib import Path
from urllib.parse import urljoin, urlparse, urldefrag

import requests
from bs4 import BeautifulSoup

class CrawlConfig:
    def __init__(
        self,
        seed_urls,
        out_dir,
        allowed_domains,
        max_pages= 200,
        max_depth= 5,
        timeout_sec= 20,
        sleep_sec= 0.5,
        user_agent= "CMU-ANLP-HW2-RAG-Collector/1.0",
        allow_query= False,
        download_pdfs= True,
        url_allow_prefixes= None,
        url_allow_substrings= None,
        url_deny_substrings= None,
        url_deny_regexes= None,
        extra_headers= None,
    ) :
        self.seed_urls = list(seed_urls)
        self.out_dir = out_dir
        self.allowed_domains = list(allowed_domains)
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.timeout_sec = timeout_sec
        self.sleep_sec = sleep_sec
        self.user_agent = user_agent
        self.allow_query = allow_query
        self.download_pdfs = download_pdfs
        self.url_allow_prefixes = list(url_allow_prefixes) if url_allow_prefixes else None
        self.url_allow_substrings = list(url_allow_substrings) if url_allow_substrings else None
        self.url_deny_substrings = list(url_deny_substrings) if url_deny_substrings else None
        self.url_deny_regexes = list(url_deny_regexes) if url_deny_regexes else None
        self.extra_headers = dict(extra_headers) if extra_headers else None


def _ensure_dir(path) :
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _write_jsonl(records, path) :
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _canonicalize_url(url, allow_query) :
    url, _frag = urldefrag(url)
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    path = parsed.path or "/"
    query = parsed.query if allow_query else ""
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    rebuilt = parsed._replace(scheme=scheme, netloc=netloc, path=path, query=query, fragment="").geturl()
    return rebuilt


def _is_allowed(url, allowed_domains) :
    host = urlparse(url).netloc.lower()
    return any(host == d or host.endswith("." + d) for d in allowed_domains)


def _url_to_filename(url, ext) :
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
    ext = ext if ext.startswith(".") else f".{ext}"
    return f"{h}{ext}"


def _matches_any_regex(text, patterns) :
    if not patterns:
        return False
    return any(re.search(p, text) is not None for p in patterns)


def _passes_url_filters(url, config) :
    if config.url_allow_prefixes and not any(url.startswith(p) for p in config.url_allow_prefixes):
        return False
    if config.url_allow_substrings and not any(s in url for s in config.url_allow_substrings):
        return False
    if config.url_deny_substrings and any(s in url for s in config.url_deny_substrings):
        return False
    if _matches_any_regex(url, config.url_deny_regexes):
        return False
    return True


def _guess_ext(url, content_type, is_html= False, is_pdf= False) :
    if is_html:
        return ".html"
    if is_pdf:
        return ".pdf"
    parsed_path = urlparse(url).path
    suffix = Path(parsed_path).suffix.lower()
    if suffix:
        return suffix
    mime = content_type.split(";")[0].strip().lower()
    return mimetypes.guess_extension(mime) or ".bin"


def _extract_links(html, base_url) :
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith(("mailto:", "javascript:", "tel:")):
            continue
        links.append(urljoin(base_url, href))
    return links


def crawl(config, verbose= False) :
    out_dir = _ensure_dir(config.out_dir)
    html_dir = _ensure_dir(out_dir / "pages")
    file_dir = _ensure_dir(out_dir / "files")
    session = requests.Session()
    session.headers.update({"User-Agent": config.user_agent})
    if config.extra_headers:
        session.headers.update(config.extra_headers)

    queue = deque()
    for u in config.seed_urls:
        cu = _canonicalize_url(u, config.allow_query)
        if _is_allowed(cu, config.allowed_domains) and _passes_url_filters(cu, config):
            queue.append((cu, 0))

    visited = set()
    manifest = []

    while queue and len(manifest) < config.max_pages:
        url, depth = queue.popleft()
        if url in visited:
            continue
        visited.add(url)
        if verbose:
            print(f"[crawl] ({len(manifest)+1}/{config.max_pages}) depth={depth} {url}")
        try:
            resp = session.get(url, timeout=config.timeout_sec)
            content_type = resp.headers.get("content-type", "")
            resp.raise_for_status()
        except Exception as e:
            manifest.append(
                {
                    "url": url,
                    "status": "error",
                    "error": repr(e),
                    "depth": depth,
                    "timestamp": int(time.time()),
                }
            )
            continue

        ctype = content_type.lower()
        is_html = "text/html" in ctype
        is_pdf = "application/pdf" in ctype or url.lower().endswith(".pdf")

        if not is_html and not (config.download_pdfs and is_pdf):
            manifest.append(
                {
                    "url": url,
                    "status": "skipped",
                    "reason": "unsupported_content_type",
                    "content_type": content_type,
                    "depth": depth,
                    "timestamp": int(time.time()),
                }
            )
            continue

        ext = _guess_ext(url, content_type, is_html=is_html, is_pdf=is_pdf)
        file_name = _url_to_filename(url, ext)
        file_path = (html_dir if is_html else file_dir) / file_name
        if is_html:
            html = resp.text
            file_path.write_text(html, encoding="utf-8", errors="ignore")
        else:
            file_path.write_bytes(resp.content)

        rec = {
            "url": url,
            "status": "ok",
            "depth": depth,
            "file_path": str(file_path),
            "http_status": resp.status_code,
            "content_type": content_type,
            "resource_type": "html" if is_html else "pdf",
            "timestamp": int(time.time()),
        }
        manifest.append(rec)

        if is_html and depth < config.max_depth:
            for nxt in _extract_links(resp.text, base_url=url):
                cu = _canonicalize_url(nxt, config.allow_query)
                if not cu.startswith(("http://", "https://")):
                    continue
                if not _is_allowed(cu, config.allowed_domains):
                    continue
                if not _passes_url_filters(cu, config):
                    continue
                if cu not in visited:
                    queue.append((cu, depth + 1))

        if config.sleep_sec > 0:
            time.sleep(config.sleep_sec)

    _write_jsonl(manifest, out_dir / "manifest.jsonl")
    return manifest
