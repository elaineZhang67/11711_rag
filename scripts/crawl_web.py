#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_hw2.cli_utils import load_config_file
from rag_hw2.collect.crawler import CrawlConfig, crawl


def parse_args() :
    p = argparse.ArgumentParser(description="Crawl whitelisted domains and save raw HTML pages.")
    p.add_argument("--config", type=str, help="JSON/YAML config for crawl.")
    p.add_argument("--seed-url", action="append", default=[], help="Seed URL (repeatable).")
    p.add_argument("--allowed-domain", action="append", default=[], help="Allowed domain (repeatable).")
    p.add_argument("--out-dir", type=str, default="data/raw/crawl_run")
    p.add_argument("--max-pages", type=int, default=200)
    p.add_argument("--max-depth", type=int, default=2)
    p.add_argument("--sleep-sec", type=float, default=0.5)
    p.add_argument("--timeout-sec", type=int, default=20)
    p.add_argument("--allow-query", action="store_true")
    p.add_argument("--download-pdfs", action="store_true", help="Save PDF resources as well as HTML.")
    p.add_argument("--url-allow-prefix", action="append", default=None)
    p.add_argument("--url-allow-substring", action="append", default=None)
    p.add_argument("--url-deny-substring", action="append", default=None)
    p.add_argument("--url-deny-regex", action="append", default=None)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() :
    args = parse_args()
    cfg = load_config_file(args.config)
    seed_urls = args.seed_url or cfg.get("seed_urls", [])
    allowed_domains = args.allowed_domain or cfg.get("allowed_domains", [])
    if not seed_urls or not allowed_domains:
        raise SystemExit("Need --seed-url and --allowed-domain (or config file).")
    config = CrawlConfig(
        seed_urls=seed_urls,
        allowed_domains=allowed_domains,
        out_dir=cfg.get("out_dir", args.out_dir),
        max_pages=int(cfg.get("max_pages", args.max_pages)),
        max_depth=int(cfg.get("max_depth", args.max_depth)),
        timeout_sec=int(cfg.get("timeout_sec", args.timeout_sec)),
        sleep_sec=float(cfg.get("sleep_sec", args.sleep_sec)),
        allow_query=bool(cfg.get("allow_query", args.allow_query)),
        download_pdfs=bool(cfg.get("download_pdfs", args.download_pdfs)),
        url_allow_prefixes=args.url_allow_prefix or cfg.get("url_allow_prefixes"),
        url_allow_substrings=args.url_allow_substring or cfg.get("url_allow_substrings"),
        url_deny_substrings=args.url_deny_substring or cfg.get("url_deny_substrings"),
        url_deny_regexes=args.url_deny_regex or cfg.get("url_deny_regexes"),
        extra_headers=cfg.get("extra_headers"),
    )
    manifest = crawl(config, verbose=args.verbose)
    ok = sum(1 for r in manifest if r.get("status") == "ok")
    print(f"Crawled {ok} pages. Manifest: {Path(config.out_dir) / 'manifest.jsonl'}")


if __name__ == "__main__":
    main()
