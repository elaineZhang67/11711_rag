from __future__ import annotations

import hashlib
import mimetypes
import re
import time
from collections import deque
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from compact.io_utils import ensure_dir, write_json, write_jsonl


class CrawlConfig:
    def __init__(
        self,
        seed_urls: list[str],
        out_dir: str,
        allowed_domains: list[str],
        max_pages: int = 200,
        max_depth: int = 5,
        timeout_sec: int = 20,
        sleep_sec: float = 0.5,
        user_agent: str = "CMU-ANLP-HW2-RAG-Collector/1.0",
        allow_query: bool = False,
        download_pdfs: bool = True,
        url_allow_prefixes: list[str] | None = None,
        url_allow_substrings: list[str] | None = None,
        url_deny_substrings: list[str] | None = None,
        url_deny_regexes: list[str] | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
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


def _canonicalize_url(url: str, allow_query: bool) -> str:
    url, _ = urldefrag(url)
    parsed = urlparse(url)
    path = parsed.path or "/"
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    return parsed._replace(
        scheme=parsed.scheme.lower(),
        netloc=parsed.netloc.lower(),
        path=path,
        query=parsed.query if allow_query else "",
        fragment="",
    ).geturl()


def _is_allowed(url: str, allowed_domains: list[str]) -> bool:
    host = urlparse(url).netloc.lower()
    return any(host == d or host.endswith("." + d) for d in allowed_domains)


def _url_to_filename(url: str, ext: str) -> str:
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
    ext = ext if ext.startswith(".") else f".{ext}"
    return f"{h}{ext}"


def _matches_any_regex(text: str, patterns: list[str] | None) -> bool:
    if not patterns:
        return False
    return any(re.search(p, text) for p in patterns)


def _passes_url_filters(url: str, config: CrawlConfig) -> bool:
    if config.url_allow_prefixes and not any(url.startswith(p) for p in config.url_allow_prefixes):
        return False
    if config.url_allow_substrings and not any(s in url for s in config.url_allow_substrings):
        return False
    if config.url_deny_substrings and any(s in url for s in config.url_deny_substrings):
        return False
    if _matches_any_regex(url, config.url_deny_regexes):
        return False
    return True


def _guess_ext(url: str, content_type: str, is_html: bool = False, is_pdf: bool = False) -> str:
    if is_html:
        return ".html"
    if is_pdf:
        return ".pdf"
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix:
        return suffix
    mime = content_type.split(";")[0].strip().lower()
    return mimetypes.guess_extension(mime) or ".bin"


def _extract_links(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    out: list[str] = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith(("mailto:", "javascript:", "tel:")):
            continue
        out.append(urljoin(base_url, href))
    return out


def crawl(config: CrawlConfig, verbose: bool = False) -> list[dict]:
    out_dir = ensure_dir(config.out_dir)
    html_dir = ensure_dir(out_dir / "pages")
    file_dir = ensure_dir(out_dir / "files")

    session = requests.Session()
    session.headers.update({"User-Agent": config.user_agent})
    if config.extra_headers:
        session.headers.update(config.extra_headers)

    queue: deque[tuple[str, int]] = deque()
    visited: set[str] = set()
    manifest: list[dict] = []

    for u in config.seed_urls:
        cu = _canonicalize_url(u, config.allow_query)
        if _is_allowed(cu, config.allowed_domains) and _passes_url_filters(cu, config):
            queue.append((cu, 0))

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
        file_path = (html_dir if is_html else file_dir) / _url_to_filename(url, ext)
        if is_html:
            file_path.write_text(resp.text, encoding="utf-8", errors="ignore")
        else:
            file_path.write_bytes(resp.content)

        manifest.append(
            {
                "url": url,
                "status": "ok",
                "depth": depth,
                "file_path": str(file_path),
                "http_status": resp.status_code,
                "content_type": content_type,
                "resource_type": "html" if is_html else "pdf",
                "timestamp": int(time.time()),
            }
        )

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

    write_jsonl(manifest, Path(config.out_dir) / "manifest.jsonl")
    return manifest


COMMON_DENY_SUBSTRINGS = [
    "mailto:",
    "javascript:",
    "/wp-json/",
    "/feed",
    "/rss",
    "/tag/",
    "/author/",
    "/privacy",
    "/terms",
    "/login",
    "/signin",
    "/account",
    "/cart",
    "/checkout",
    "facebook.com",
    "instagram.com",
    "x.com/",
    "twitter.com",
    "youtube.com",
    "linkedin.com",
]

COMMON_DENY_REGEXES = [
    r"\.(?:jpg|jpeg|png|gif|webp|svg|ico)(?:\?.*)?$",
    r"\.(?:css|js|xml)(?:\?.*)?$",
    r"/amp(?:/|$)",
]


class SourceJob:
    def __init__(
        self,
        name: str,
        group: str,
        seed_urls: list[str],
        allowed_domains: list[str],
        max_pages: int = 120,
        max_depth: int = 5,
        allow_query: bool = False,
        download_pdfs: bool = True,
        sleep_sec: float = 0.6,
        url_allow_prefixes: list[str] | None = None,
        url_allow_substrings: list[str] | None = None,
        url_deny_substrings: list[str] | None = None,
        url_deny_regexes: list[str] | None = None,
    ) -> None:
        self.name = name
        self.group = group
        self.seed_urls = list(seed_urls)
        self.allowed_domains = list(allowed_domains)
        self.max_pages = max_pages
        depth = int(max_depth)
        if depth < 4:
            depth = 4
        if depth > 5:
            depth = 5
        self.max_depth = depth
        self.allow_query = allow_query
        self.download_pdfs = download_pdfs
        self.sleep_sec = sleep_sec
        self.url_allow_prefixes = list(url_allow_prefixes) if url_allow_prefixes else None
        self.url_allow_substrings = list(url_allow_substrings) if url_allow_substrings else None
        self.url_deny_substrings = list(url_deny_substrings) if url_deny_substrings else None
        self.url_deny_regexes = list(url_deny_regexes) if url_deny_regexes else None

    def out_dir(self, out_root: str | Path) -> str:
        safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", self.name.strip()).strip("_").lower()
        return str(Path(out_root) / self.group / safe)

    def to_crawl_config(self, out_root: str | Path, sleep_override: float | None = None) -> CrawlConfig:
        deny_substrings = list(COMMON_DENY_SUBSTRINGS)
        if self.url_deny_substrings:
            deny_substrings.extend(self.url_deny_substrings)
        deny_regexes = list(COMMON_DENY_REGEXES)
        if self.url_deny_regexes:
            deny_regexes.extend(self.url_deny_regexes)
        return CrawlConfig(
            seed_urls=self.seed_urls,
            out_dir=self.out_dir(out_root),
            allowed_domains=self.allowed_domains,
            max_pages=self.max_pages,
            max_depth=self.max_depth,
            allow_query=self.allow_query,
            download_pdfs=self.download_pdfs,
            sleep_sec=self.sleep_sec if sleep_override is None else sleep_override,
            url_allow_prefixes=self.url_allow_prefixes,
            url_allow_substrings=self.url_allow_substrings,
            url_deny_substrings=deny_substrings,
            url_deny_regexes=deny_regexes,
        )


def _sports_deny() -> list[str]:
    return [
        "/news",
        "/video",
        "/videos",
        "/watch",
        "/scores",
        "/stats",
        "/gameday",
        "/podcast",
        "/tickets",
        "/shop",
        "/community",
        "/fantasy",
        "/betting",
    ]


def recommended_source_jobs() -> list[SourceJob]:
    return [
        SourceJob(
            name="wikipedia_pittsburgh_history",
            group="general",
            seed_urls=[
                "https://en.wikipedia.org/wiki/Pittsburgh",
                "https://en.wikipedia.org/wiki/History_of_Pittsburgh",
            ],
            allowed_domains=["wikipedia.org"],
            max_pages=80,
            max_depth=1,
            url_allow_prefixes=["https://en.wikipedia.org/wiki/"],
            url_deny_substrings=["/wiki/Special:", "/wiki/Help:", "/wiki/Talk:"],
        ),
        SourceJob(
            name="city_of_pittsburgh_home",
            group="general",
            seed_urls=["https://www.pittsburghpa.gov/Home"],
            allowed_domains=["pittsburghpa.gov"],
            max_pages=80,
            max_depth=2,
        ),
        SourceJob(
            name="city_tax_regulations",
            group="general",
            seed_urls=["https://pittsburghpa.gov/finance/tax-forms"],
            allowed_domains=["pittsburghpa.gov"],
            max_pages=120,
            max_depth=2,
            download_pdfs=True,
            url_allow_substrings=["pittsburghpa.gov/finance", "/tax", "/regulation", ".pdf"],
        ),
        SourceJob(
            name="city_budget_2025_pdf",
            group="general",
            seed_urls=[
                "https://www.pittsburghpa.gov/files/assets/city/v/4/omb/documents/operating-budgets/2025-operating-budget.pdf"
            ],
            allowed_domains=["pittsburghpa.gov"],
            max_pages=5,
            max_depth=0,
            download_pdfs=True,
        ),
        SourceJob(
            name="britannica_pittsburgh",
            group="general",
            seed_urls=["https://www.britannica.com/place/Pittsburgh"],
            allowed_domains=["britannica.com"],
            max_pages=30,
            max_depth=1,
            url_allow_substrings=["britannica.com/place/Pittsburgh"],
        ),
        SourceJob(
            name="visit_pittsburgh_general",
            group="general",
            seed_urls=["https://www.visitpittsburgh.com/", "https://www.visitpittsburgh.com/things-to-do/"],
            allowed_domains=["visitpittsburgh.com"],
            max_pages=160,
            max_depth=2,
        ),
        SourceJob(
            name="cmu_about_history",
            group="general",
            seed_urls=["https://www.cmu.edu/about/", "https://www.cmu.edu/about/history.html"],
            allowed_domains=["cmu.edu"],
            max_pages=80,
            max_depth=2,
            url_allow_substrings=["cmu.edu/about"],
        ),
        SourceJob(
            name="pittsburgh_events_calendar",
            group="events",
            seed_urls=["https://pittsburgh.events", "https://pittsburgh.events/calendar"],
            allowed_domains=["pittsburgh.events"],
            max_pages=200,
            max_depth=3,
            allow_query=True,
            url_deny_substrings=["/news", "/blog", "/contact", "/advertise"],
        ),
        SourceJob(
            name="downtown_pittsburgh_events",
            group="events",
            seed_urls=["https://downtownpittsburgh.com/events/"],
            allowed_domains=["downtownpittsburgh.com"],
            max_pages=150,
            max_depth=3,
        ),
        SourceJob(
            name="pgh_citypaper_events",
            group="events",
            seed_urls=["https://www.pghcitypaper.com/pittsburgh/EventSearch?v=d"],
            allowed_domains=["pghcitypaper.com"],
            max_pages=150,
            max_depth=2,
            allow_query=True,
            url_allow_substrings=["/pittsburgh/EventSearch", "/event", "/EventSearch", "pghcitypaper.com"],
        ),
        SourceJob(
            name="cmu_events_calendar",
            group="events",
            seed_urls=["https://events.cmu.edu"],
            allowed_domains=["events.cmu.edu"],
            max_pages=180,
            max_depth=3,
            allow_query=True,
        ),
        SourceJob(
            name="cmu_campus_events_page",
            group="events",
            seed_urls=["https://www.cmu.edu/engage/alumni/events/campus/index.html"],
            allowed_domains=["cmu.edu"],
            max_pages=80,
            max_depth=2,
            url_allow_substrings=["cmu.edu/engage/alumni/events/campus"],
        ),
        SourceJob(
            name="pittsburgh_symphony",
            group="culture",
            seed_urls=["https://www.pittsburghsymphony.org/"],
            allowed_domains=["pittsburghsymphony.org"],
            max_pages=140,
            max_depth=2,
        ),
        SourceJob(
            name="pittsburgh_opera",
            group="culture",
            seed_urls=["https://pittsburghopera.org/"],
            allowed_domains=["pittsburghopera.org"],
            max_pages=140,
            max_depth=2,
        ),
        SourceJob(
            name="trustarts",
            group="culture",
            seed_urls=["https://trustarts.org/"],
            allowed_domains=["trustarts.org"],
            max_pages=180,
            max_depth=3,
        ),
        SourceJob(
            name="carnegie_museums",
            group="culture",
            seed_urls=["https://carnegiemuseums.org/"],
            allowed_domains=["carnegiemuseums.org"],
            max_pages=160,
            max_depth=3,
        ),
        SourceJob(
            name="heinz_history_center",
            group="culture",
            seed_urls=["https://www.heinzhistorycenter.org/"],
            allowed_domains=["heinzhistorycenter.org"],
            max_pages=160,
            max_depth=3,
        ),
        SourceJob(
            name="the_frick_pittsburgh",
            group="culture",
            seed_urls=["https://www.thefrickpittsburgh.org/"],
            allowed_domains=["thefrickpittsburgh.org"],
            max_pages=160,
            max_depth=3,
        ),
        SourceJob(
            name="visit_pittsburgh_arts_culture",
            group="culture",
            seed_urls=[
                "https://www.visitpittsburgh.com/things-to-do/arts-culture/",
                "https://www.visitpittsburgh.com/things-to-do/arts-culture/museums/",
            ],
            allowed_domains=["visitpittsburgh.com"],
            max_pages=120,
            max_depth=2,
        ),
        SourceJob(
            name="visit_pittsburgh_food_festivals",
            group="food",
            seed_urls=["https://www.visitpittsburgh.com/events-festivals/food-festivals/"],
            allowed_domains=["visitpittsburgh.com"],
            max_pages=100,
            max_depth=2,
        ),
        SourceJob(
            name="picklesburgh",
            group="food",
            seed_urls=["https://www.picklesburgh.com/"],
            allowed_domains=["picklesburgh.com"],
            max_pages=100,
            max_depth=2,
        ),
        SourceJob(
            name="pittsburgh_taco_fest",
            group="food",
            seed_urls=["https://www.pghtacofest.com/"],
            allowed_domains=["pghtacofest.com"],
            max_pages=100,
            max_depth=2,
        ),
        SourceJob(
            name="pittsburgh_restaurant_week",
            group="food",
            seed_urls=["https://pittsburghrestaurantweek.com/"],
            allowed_domains=["pittsburghrestaurantweek.com"],
            max_pages=120,
            max_depth=2,
        ),
        SourceJob(
            name="little_italy_days",
            group="food",
            seed_urls=["https://littleitalydays.com/"],
            allowed_domains=["littleitalydays.com"],
            max_pages=100,
            max_depth=2,
        ),
        SourceJob(
            name="banana_split_fest",
            group="food",
            seed_urls=["https://bananasplitfest.com/"],
            allowed_domains=["bananasplitfest.com"],
            max_pages=100,
            max_depth=2,
        ),
        SourceJob(
            name="visit_pittsburgh_sports_teams",
            group="sports",
            seed_urls=["https://www.visitpittsburgh.com/things-to-do/pittsburgh-sports-teams/"],
            allowed_domains=["visitpittsburgh.com"],
            max_pages=80,
            max_depth=2,
        ),
        SourceJob(
            name="pirates_site",
            group="sports",
            seed_urls=["https://www.mlb.com/pirates"],
            allowed_domains=["mlb.com"],
            max_pages=120,
            max_depth=2,
            url_allow_substrings=["mlb.com/pirates"],
            url_deny_substrings=_sports_deny(),
        ),
        SourceJob(
            name="steelers_site",
            group="sports",
            seed_urls=["https://www.steelers.com/"],
            allowed_domains=["steelers.com"],
            max_pages=120,
            max_depth=2,
            url_deny_substrings=_sports_deny(),
        ),
        SourceJob(
            name="penguins_site",
            group="sports",
            seed_urls=["https://www.nhl.com/penguins/"],
            allowed_domains=["nhl.com"],
            max_pages=120,
            max_depth=2,
            url_allow_substrings=["nhl.com/penguins"],
            url_deny_substrings=_sports_deny(),
        ),
    ]


def filter_jobs(
    jobs: list[SourceJob],
    groups: list[str] | None = None,
    include_names: list[str] | None = None,
    exclude_names: list[str] | None = None,
) -> list[SourceJob]:
    out = jobs
    if groups:
        group_set = {g.lower() for g in groups}
        out = [j for j in out if j.group.lower() in group_set]
    if include_names:
        inc = {x.lower() for x in include_names}
        out = [j for j in out if j.name.lower() in inc]
    if exclude_names:
        exc = {x.lower() for x in exclude_names}
        out = [j for j in out if j.name.lower() not in exc]
    return out


def list_recommended_sources(
    groups: list[str] | None = None,
    include_names: list[str] | None = None,
    exclude_names: list[str] | None = None,
) -> list[SourceJob]:
    return sorted(
        filter_jobs(recommended_source_jobs(), groups, include_names, exclude_names),
        key=lambda j: (j.group, j.name),
    )


def run_recommended_crawls(
    out_root: str | Path,
    groups: list[str] | None = None,
    include_names: list[str] | None = None,
    exclude_names: list[str] | None = None,
    max_pages_multiplier: float = 1.0,
    sleep_sec: float | None = None,
    verbose: bool = False,
) -> list[dict]:
    jobs = list_recommended_sources(groups, include_names, exclude_names)
    if not jobs:
        raise ValueError("No source jobs selected.")
    summary: list[dict] = []
    for job in jobs:
        cfg = job.to_crawl_config(out_root, sleep_override=sleep_sec)
        cfg.max_pages = max(1, int(round(cfg.max_pages * max_pages_multiplier)))
        manifest = crawl(cfg, verbose=verbose)
        ok = sum(1 for r in manifest if r.get("status") == "ok")
        skipped = sum(1 for r in manifest if r.get("status") == "skipped")
        errors = sum(1 for r in manifest if r.get("status") == "error")
        html = sum(1 for r in manifest if r.get("resource_type") == "html")
        pdf = sum(1 for r in manifest if r.get("resource_type") == "pdf")
        summary.append(
            {
                "group": job.group,
                "name": job.name,
                "out_dir": cfg.out_dir,
                "max_pages": cfg.max_pages,
                "max_depth": cfg.max_depth,
                "ok": ok,
                "html": html,
                "pdf": pdf,
                "skipped": skipped,
                "errors": errors,
                "seed_urls": cfg.seed_urls,
            }
        )
    write_json(summary, Path(out_root) / "crawl_summary.json", indent=2)
    return summary
