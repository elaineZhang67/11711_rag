#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_hw2.collect.crawler import CrawlConfig, crawl


# Broad CMU seeds across university pages, schools, departments, research units,
# history/about pages, events, athletics, and commonly asked academic sites.
CMU_SEEDS = [
    "https://www.cmu.edu/",
    "https://www.cmu.edu/about/",
    "https://www.cmu.edu/about/history.html",
    "https://www.cmu.edu/about/rankings-and-facts.html",
    "https://www.cmu.edu/research/",
    "https://www.cmu.edu/news/",
    "https://www.cmu.edu/innovation/",
    "https://www.cmu.edu/events/",
    "https://www.cmu.edu/campus-life/",
    "https://www.cmu.edu/leadership/",
    "https://www.cmu.edu/commencement/",
    "https://www.cmu.edu/engage/alumni/events/campus/spring-carnival/",
    "https://www.cmu.edu/visit/",
    "https://www.cmu.edu/academics/",
    "https://www.cmu.edu/admission/",
    "https://www.cmu.edu/graduate/",
    "https://www.cmu.edu/undergraduate/",
    "https://www.cmu.edu/student-affairs/",
    "https://www.cmu.edu/tepper/",
    "https://www.cmu.edu/dietrich/",
    "https://www.cmu.edu/engineering/",
    "https://www.cmu.edu/cfa/",
    "https://www.cmu.edu/heinz/",
    "https://www.cmu.edu/mcs/",
    "https://www.cmu.edu/scs/",
    "https://www.cmu.edu/robotics/",
    "https://www.cmu.edu/hcii/",
    "https://www.cmu.edu/iii/",
    "https://www.cmu.edu/brand/",
    "https://www.cmu.edu/maps/",
    "https://www.cmu.edu/piper/",
    "https://www.cmu.edu/energy-week/",
    "https://www.cmu.edu/utc/",
    "https://www.cmu.edu/swartz-center-for-entrepreneurship/",
    "https://www.cs.cmu.edu/",
    "https://www.cs.cmu.edu/about/",
    "https://www.cs.cmu.edu/people/faculty",
    "https://www.cs.cmu.edu/research",
    "https://www.cs.cmu.edu/scs25/history",
    "https://www.cs.cmu.edu/scs25/25things",
    "https://scs.cmu.edu/",
    "https://scs.cmu.edu/about-scs",
    "https://www.ml.cmu.edu/",
    "https://lti.cs.cmu.edu/",
    "https://lti.cs.cmu.edu/people",
    "https://www.ri.cmu.edu/",
    "https://www.ri.cmu.edu/about/",
    "https://www.ri.cmu.edu/research/",
    "https://www.cylab.cmu.edu/",
    "https://www.sei.cmu.edu/",
    "https://www.sei.cmu.edu/about/",
    "https://www.ece.cmu.edu/",
    "https://www.ece.cmu.edu/about/index.html",
    "https://www.cmu.edu/bme/",
    "https://www.cmu.edu/chemistry/",
    "https://www.cmu.edu/physics/",
    "https://www.cmu.edu/math/",
    "https://www.cmu.edu/stat/",
    "https://www.cmu.edu/english/",
    "https://www.cmu.edu/history/",
    "https://www.cmu.edu/philosophy/",
    "https://www.cmu.edu/sds/",
    "https://www.cmu.edu/tepper/programs/",
    "https://www.cmu.edu/cfa/schools/",
    "https://www.cmu.edu/cfa/school-of-art/",
    "https://www.cmu.edu/cfa/drama/",
    "https://www.cmu.edu/cfa/music/",
    "https://www.cmu.edu/heinz/",
    "https://www.cmu.edu/mcs/biology/",
    "https://www.cmu.edu/mcs/chemistry/",
    "https://www.cmu.edu/mcs/physics/",
    "https://www.cmu.edu/mcs/math/",
    "https://athletics.cmu.edu/",
    "https://athletics.cmu.edu/athletics/tartanfacts",
    "https://athletics.cmu.edu/athletics/mascot/about",
    "https://athletics.cmu.edu/athletics/kiltieband/index",
    "https://athletics.cmu.edu/sports/football",
    "https://athletics.cmu.edu/sports/mbkb",
    "https://athletics.cmu.edu/sports/wbkb",
    "https://athletics.cmu.edu/sports/baseball",
    "https://athletics.cmu.edu/sports/msoc",
    "https://athletics.cmu.edu/sports/wsoc",
    "https://athletics.cmu.edu/sports/track",
    "https://cmu-ca.org/",
    "https://www.qatar.cmu.edu/",
    "https://www.africa.engineering.cmu.edu/",
]


DEFAULT_DENY_SUBSTRINGS = [
    "mailto:",
    "javascript:",
    "/search",
    "/search-results",
    "/give-to-cmu",
    "/apply",
    "/visit/admission/events/register",
    "/account",
    "/login",
    "/signin",
    "/logout",
    "/sitemap",
    "/calendar.ics",
    "/ics/",
    "/rss",
    "/feed",
    "facebook.com",
    "instagram.com",
    "twitter.com",
    "x.com/",
    "youtube.com",
    "linkedin.com",
    "tiktok.com",
]


DEFAULT_DENY_REGEXES = [
    r"\.(?:jpg|jpeg|png|gif|webp|svg|ico)(?:\?.*)?$",
    r"\.(?:css|js|xml)(?:\?.*)?$",
    r"\.(?:mp4|mov|avi|mkv|webm)(?:\?.*)?$",
    r"\.(?:mp3|wav|ogg)(?:\?.*)?$",
    r"\.(?:zip|tar|gz|rar)(?:\?.*)?$",
]


def _write_json(obj, path, indent=2):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=indent), encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser(description="Broad crawl of CMU-related websites (*.cmu.edu) into one raw-data directory.")
    p.add_argument("--out-dir", default="data-raw-cmu")
    p.add_argument("--max-pages", type=int, default=12000)
    p.add_argument("--max-depth", type=int, default=5)
    p.add_argument("--sleep-sec", type=float, default=0.6)
    p.add_argument("--timeout-sec", type=int, default=20)
    p.add_argument("--allow-query", action="store_true", help="Keep query strings in URLs (usually off).")
    p.add_argument("--no-download-pdfs", action="store_true", help="Do not save PDF files (HTML only).")
    p.add_argument("--extra-seed", action="append", default=[], help="Additional CMU seed URL(s).")
    p.add_argument("--seed-file", default=None, help="Optional text file with one seed URL per line.")
    p.add_argument("--list-seeds", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def _load_seed_file(path):
    seeds = []
    if not path:
        return seeds
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"Seed file not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            seeds.append(line)
    return seeds


def _dedupe_keep_order(items):
    seen = set()
    out = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def main():
    args = parse_args()

    seeds = list(CMU_SEEDS)
    seeds.extend(_load_seed_file(args.seed_file))
    seeds.extend(args.extra_seed or [])
    seeds = [s.strip() for s in seeds if s and s.strip()]
    seeds = [s for s in seeds if "cmu.edu" in s]
    seeds = _dedupe_keep_order(seeds)

    if args.list_seeds:
        print(f"Total CMU seeds: {len(seeds)}")
        for s in seeds:
            print(s)
        return

    max_depth = int(args.max_depth)
    if max_depth < 4:
        max_depth = 4
    if max_depth > 5:
        max_depth = 5

    cfg = CrawlConfig(
        seed_urls=seeds,
        out_dir=args.out_dir,
        allowed_domains=["cmu.edu"],
        max_pages=int(args.max_pages),
        max_depth=max_depth,
        timeout_sec=int(args.timeout_sec),
        sleep_sec=float(args.sleep_sec),
        allow_query=bool(args.allow_query),
        download_pdfs=not bool(args.no_download_pdfs),
        url_deny_substrings=list(DEFAULT_DENY_SUBSTRINGS),
        url_deny_regexes=list(DEFAULT_DENY_REGEXES),
    )

    print(f"CMU crawl starting: seeds={len(seeds)} max_pages={cfg.max_pages} max_depth={cfg.max_depth} out={cfg.out_dir}")
    manifest = crawl(cfg, verbose=args.verbose)

    ok = sum(1 for r in manifest if r.get("status") == "ok")
    html = sum(1 for r in manifest if r.get("resource_type") == "html")
    pdf = sum(1 for r in manifest if r.get("resource_type") == "pdf")
    skipped = sum(1 for r in manifest if r.get("status") == "skipped")
    errors = sum(1 for r in manifest if r.get("status") == "error")
    summary = {
        "out_dir": cfg.out_dir,
        "seed_count": len(seeds),
        "max_pages": cfg.max_pages,
        "max_depth": cfg.max_depth,
        "sleep_sec": cfg.sleep_sec,
        "timeout_sec": cfg.timeout_sec,
        "ok": ok,
        "html": html,
        "pdf": pdf,
        "skipped": skipped,
        "errors": errors,
        "seeds": seeds,
    }
    summary_path = Path(cfg.out_dir) / "crawl_summary.json"
    _write_json(summary, summary_path, indent=2)
    print(f"Done. ok={ok} html={html} pdf={pdf} skipped={skipped} errors={errors}")
    print(f"Manifest: {Path(cfg.out_dir) / 'manifest.jsonl'}")
    print(f"Summary:  {summary_path}")


if __name__ == "__main__":
    main()
