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


CMU_EXTRA_SEEDS = [
    "https://www.library.cmu.edu/",
    "https://www.cmu.edu/sustainability/",
    "https://www.cmu.edu/transportation/",
    "https://www.cmu.edu/parking/",
    "https://www.cmu.edu/health-services/",
    "https://www.cmu.edu/wellness/",
    "https://www.cmu.edu/life/",
    "https://www.cmu.edu/facilities/",
    "https://www.cmu.edu/about/rankings-and-facts.html",
    "https://www.cmu.edu/graduate/programs/",
    "https://www.cmu.edu/undergraduate/apply/",
    "https://www.cmu.edu/admission/visit/",
    "https://www.cmu.edu/student-affairs/",
    "https://www.cmu.edu/student-affairs/slice/",
    "https://www.cmu.edu/career/",
    "https://www.cmu.edu/hub/",
    "https://www.cmu.edu/leadership/the-provost/",
    "https://www.cmu.edu/global/",
    "https://www.cmu.edu/research/centers-institutes/",
    "https://www.cmu.edu/innovation/transfer/",
    "https://www.cmu.edu/cfa/school-of-architecture/",
    "https://www.cmu.edu/cfa/school-of-design/",
    "https://www.cmu.edu/cfa/school-of-drama/",
    "https://www.cmu.edu/cfa/school-of-music/",
    "https://www.cmu.edu/engineering/research/",
    "https://www.cmu.edu/mcs/research/",
    "https://www.cmu.edu/heinz/research/",
    "https://www.cmu.edu/tepper/research/",
    "https://www.cmu.edu/scs/research/",
    "https://www.cmu.edu/robotics/research/",
]


PITTSBURGH_EXTRA_SEEDS = [
    "https://www.pittsburghpa.gov/Home",
    "https://www.pittsburghpa.gov/Government/Departments",
    "https://www.pittsburghpa.gov/City-Government/Mayor-and-Administration",
    "https://www.pittsburghpa.gov/Business/Permits-Licenses-and-Inspections",
    "https://www.pittsburghpa.gov/Business/Taxes-and-Payments",
    "https://www.pittsburghpa.gov/Services/Public-Safety",
    "https://www.pittsburghpa.gov/Services/Parks-and-Recreation",
    "https://www.pittsburghpa.gov/Services/Public-Works",
    "https://www.pittsburghpa.gov/Services/Planning-and-Development",
    "https://www.pittsburghpa.gov/Events",
    "https://www.alleghenycounty.us/",
    "https://www.alleghenycounty.us/Services",
    "https://www.alleghenycounty.us/Government",
    "https://www.rideprt.org/",
    "https://www.rideprt.org/inside-Pittsburgh-Regional-Transit/",
    "https://www.rideprt.org/fares-and-passes/",
    "https://www.rideprt.org/schedules/",
    "https://www.flypittsburgh.com/",
    "https://www.flypittsburgh.com/pittsburgh-international-airport/",
    "https://www.flypittsburgh.com/flights/",
    "https://www.visitpittsburgh.com/neighborhoods/",
    "https://www.visitpittsburgh.com/things-to-do/outdoors/",
    "https://www.visitpittsburgh.com/things-to-do/tours/",
    "https://www.visitpittsburgh.com/plan-your-trip/getting-around/",
    "https://www.pittsburghparks.org/",
    "https://www.pittsburghparks.org/parks/",
    "https://www.pittsburghzoo.org/",
    "https://www.phipps.conservatory.org/",
    "https://carnegiesciencecenter.org/",
    "https://www.pointpark.edu/community",
]


CMU_DENY_SUBSTRINGS = [
    "mailto:",
    "javascript:",
    "/apply",
    "/give",
    "/login",
    "/signin",
    "/account",
    "/cart",
    "/search",
    "/calendar.ics",
    "/ics/",
    "/feed",
    "/rss",
    "facebook.com",
    "instagram.com",
    "x.com/",
    "twitter.com",
    "youtube.com",
    "linkedin.com",
]


PITTSBURGH_DENY_SUBSTRINGS = [
    "mailto:",
    "javascript:",
    "/login",
    "/signin",
    "/account",
    "/cart",
    "/checkout",
    "/search",
    "/feed",
    "/rss",
    "facebook.com",
    "instagram.com",
    "x.com/",
    "twitter.com",
    "youtube.com",
    "linkedin.com",
    "tiktok.com",
]


COMMON_DENY_REGEXES = [
    r"\.(?:jpg|jpeg|png|gif|webp|svg|ico)(?:\?.*)?$",
    r"\.(?:css|js|xml)(?:\?.*)?$",
    r"\.(?:mp4|mov|avi|mkv|webm)(?:\?.*)?$",
    r"\.(?:mp3|wav|ogg)(?:\?.*)?$",
    r"\.(?:zip|tar|gz|rar)(?:\?.*)?$",
]


def _iter_manifest_paths(roots):
    seen = set()
    for root in roots:
        p = Path(root)
        if not p.exists():
            continue
        for mp in p.rglob("manifest.jsonl"):
            rp = str(mp.resolve())
            if rp in seen:
                continue
            seen.add(rp)
            yield mp


def _load_existing_urls(roots):
    urls = set()
    for mp in _iter_manifest_paths(roots):
        with mp.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("status") != "ok":
                    continue
                u = rec.get("url")
                if u:
                    urls.add(u)
    return urls


def _run_one(name, config, verbose):
    print(f"\n[{name}] seeds={len(config.seed_urls)} max_pages={config.max_pages} max_depth={config.max_depth}")
    print(f"[{name}] out_dir={config.out_dir}")
    manifest = crawl(config, verbose=verbose)
    ok = sum(1 for r in manifest if r.get("status") == "ok")
    html = sum(1 for r in manifest if r.get("resource_type") == "html")
    pdf = sum(1 for r in manifest if r.get("resource_type") == "pdf")
    skipped = sum(1 for r in manifest if r.get("status") == "skipped")
    errors = sum(1 for r in manifest if r.get("status") == "error")
    print(f"[{name}] ok={ok} html={html} pdf={pdf} skipped={skipped} errors={errors}")
    return {
        "name": name,
        "out_dir": config.out_dir,
        "seed_count": len(config.seed_urls),
        "max_pages": config.max_pages,
        "max_depth": config.max_depth,
        "ok": ok,
        "html": html,
        "pdf": pdf,
        "skipped": skipped,
        "errors": errors,
    }


def parse_args():
    p = argparse.ArgumentParser(
        description="Scrape more CMU/Pittsburgh pages while skipping URLs already present in previous manifests."
    )
    p.add_argument("--out-root", default="data/raw/more_cmu_pittsburgh")
    p.add_argument(
        "--existing-root",
        action="append",
        default=["data/raw", "data-raw", "data-raw-cmu"],
        help="Directory to scan for existing manifest.jsonl files (repeatable).",
    )
    p.add_argument("--max-pages-cmu", type=int, default=2000)
    p.add_argument("--max-pages-pittsburgh", type=int, default=2000)
    p.add_argument("--max-depth", type=int, default=5)
    p.add_argument("--sleep-sec", type=float, default=0.8)
    p.add_argument("--timeout-sec", type=int, default=20)
    p.add_argument("--allow-query", action="store_true")
    p.add_argument("--cmu-only", action="store_true")
    p.add_argument("--pittsburgh-only", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.cmu_only and args.pittsburgh_only:
        raise SystemExit("Use only one of --cmu-only or --pittsburgh-only.")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    existing_roots = list(args.existing_root)
    existing_roots.append(str(out_root))
    existing_urls = _load_existing_urls(existing_roots)
    print(f"Loaded {len(existing_urls)} existing URLs from manifest files.")

    depth = int(args.max_depth)
    if depth < 4:
        depth = 4
    if depth > 5:
        depth = 5

    summary = {
        "out_root": str(out_root),
        "existing_roots": existing_roots,
        "existing_url_count": len(existing_urls),
        "jobs": [],
    }

    run_cmu = not args.pittsburgh_only
    run_pgh = not args.cmu_only

    if run_cmu:
        cmu_cfg = CrawlConfig(
            seed_urls=CMU_EXTRA_SEEDS,
            out_dir=str(out_root / "cmu"),
            allowed_domains=["cmu.edu"],
            max_pages=int(args.max_pages_cmu),
            max_depth=depth,
            timeout_sec=int(args.timeout_sec),
            sleep_sec=float(args.sleep_sec),
            allow_query=bool(args.allow_query),
            download_pdfs=True,
            url_deny_substrings=CMU_DENY_SUBSTRINGS,
            url_deny_regexes=COMMON_DENY_REGEXES,
            skip_urls=list(existing_urls),
        )
        cmu_result = _run_one("cmu_more", cmu_cfg, verbose=args.verbose)
        summary["jobs"].append(cmu_result)

        cmu_manifest = Path(cmu_cfg.out_dir) / "manifest.jsonl"
        if cmu_manifest.exists():
            with cmu_manifest.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    u = rec.get("url")
                    if u:
                        existing_urls.add(u)

    if run_pgh:
        pgh_cfg = CrawlConfig(
            seed_urls=PITTSBURGH_EXTRA_SEEDS,
            out_dir=str(out_root / "pittsburgh"),
            allowed_domains=[
                "pittsburghpa.gov",
                "alleghenycounty.us",
                "rideprt.org",
                "flypittsburgh.com",
                "visitpittsburgh.com",
                "pittsburghparks.org",
                "pittsburghzoo.org",
                "phipps.conservatory.org",
                "carnegiesciencecenter.org",
                "pointpark.edu",
            ],
            max_pages=int(args.max_pages_pittsburgh),
            max_depth=depth,
            timeout_sec=int(args.timeout_sec),
            sleep_sec=float(args.sleep_sec),
            allow_query=bool(args.allow_query),
            download_pdfs=True,
            url_deny_substrings=PITTSBURGH_DENY_SUBSTRINGS,
            url_deny_regexes=COMMON_DENY_REGEXES,
            skip_urls=list(existing_urls),
        )
        pgh_result = _run_one("pittsburgh_more", pgh_cfg, verbose=args.verbose)
        summary["jobs"].append(pgh_result)

    summary_path = out_root / "crawl_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote summary: {summary_path}")


if __name__ == "__main__":
    main()
