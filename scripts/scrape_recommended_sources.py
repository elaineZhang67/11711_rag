#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_hw2.collect.crawler import crawl
from rag_hw2.collect.source_catalog import filter_jobs, recommended_source_jobs


def _write_json(obj, path, indent= 2) :
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=indent), encoding="utf-8")


def parse_args() :
    p = argparse.ArgumentParser(
        description="Scrape the assignment's recommended Pittsburgh/CMU sources into raw HTML/PDF files."
    )
    p.add_argument(
        "--out-root",
        type=str,
        default="data/raw/recommended_sources",
        help="Root directory for per-source crawl outputs.",
    )
    p.add_argument(
        "--groups",
        nargs="+",
        default=None,
        choices=["general", "events", "culture", "food", "sports"],
        help="Optional subset of source groups to crawl.",
    )
    p.add_argument("--include-source", action="append", default=None, help="Run only specific source job name(s).")
    p.add_argument("--exclude-source", action="append", default=None, help="Skip specific source job name(s).")
    p.add_argument("--list-sources", action="store_true", help="List curated source jobs and exit.")
    p.add_argument("--max-pages-multiplier", type=float, default=1.0, help="Scale max_pages for all jobs.")
    p.add_argument("--sleep-sec", type=float, default=None, help="Override per-job sleep interval.")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() :
    args = parse_args()
    jobs = filter_jobs(
        recommended_source_jobs(),
        groups=args.groups,
        include_names=args.include_source,
        exclude_names=args.exclude_source,
    )
    jobs = sorted(jobs, key=lambda j: (j.group, j.name))

    if args.list_sources:
        for job in jobs:
            print(f"{job.group:>7}  {job.name:<32}  pages={job.max_pages:<4} depth={job.max_depth}  seeds={len(job.seed_urls)}")
            for seed in job.seed_urls:
                print(f"         - {seed}")
        return

    if not jobs:
        raise SystemExit("No source jobs selected.")

    summary = []
    print(f"Selected {len(jobs)} source jobs. Output root: {args.out_root}")
    for i, job in enumerate(jobs, start=1):
        cfg = job.to_crawl_config(args.out_root, sleep_override=args.sleep_sec)
        cfg.max_pages = max(1, int(round(cfg.max_pages * args.max_pages_multiplier)))
        print(f"\n[{i}/{len(jobs)}] {job.group}/{job.name}")
        print(f"  out_dir={cfg.out_dir}")
        print(f"  seeds={len(cfg.seed_urls)} max_pages={cfg.max_pages} max_depth={cfg.max_depth}")
        manifest = crawl(cfg, verbose=args.verbose)
        ok = sum(1 for r in manifest if r.get("status") == "ok")
        skipped = sum(1 for r in manifest if r.get("status") == "skipped")
        errors = sum(1 for r in manifest if r.get("status") == "error")
        html = sum(1 for r in manifest if r.get("resource_type") == "html")
        pdf = sum(1 for r in manifest if r.get("resource_type") == "pdf")
        print(f"  ok={ok} html={html} pdf={pdf} skipped={skipped} errors={errors}")
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

    summary_path = Path(args.out_root) / "crawl_summary.json"
    _write_json(summary, summary_path, indent=2)
    print(f"\nWrote crawl summary to {summary_path}")


if __name__ == "__main__":
    main()
