#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_hw2.preprocess.corpus_builder import build_corpus


def parse_args() :
    p = argparse.ArgumentParser(description="Build cleaned document corpus JSONL from raw files.")
    p.add_argument(
        "--input-dir",
        action="append",
        required=True,
        help="Directory or file to ingest (repeatable).",
    )
    p.add_argument("--output", type=str, default="data/processed/documents.jsonl")
    p.add_argument("--min-chars", type=int, default=100)
    p.add_argument("--no-dedupe", action="store_true")
    p.add_argument("--include-images", action="store_true", help="OCR image files (png/jpg/etc.) into the corpus.")
    p.add_argument("--quality-filter", action="store_true", help="Enable source/text quality filtering.")
    p.add_argument("--drop-url-pattern", action="append", default=[], help="Regex pattern to drop source_url.")
    p.add_argument("--drop-path-pattern", action="append", default=[], help="Regex pattern to drop source_path.")
    p.add_argument("--drop-title-pattern", action="append", default=[], help="Regex pattern to drop title.")
    p.add_argument("--drop-text-pattern", action="append", default=[], help="Regex pattern to drop text.")
    p.add_argument("--min-alpha-ratio", type=float, default=0.45)
    p.add_argument("--max-digit-ratio", type=float, default=0.45)
    p.add_argument("--max-repeat-line-ratio", type=float, default=0.5)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() :
    args = parse_args()
    docs = build_corpus(
        input_dirs=args.input_dir,
        output_jsonl=args.output,
        min_chars=args.min_chars,
        dedupe_exact=not args.no_dedupe,
        include_images=args.include_images,
        quality_filter=args.quality_filter,
        drop_url_patterns=args.drop_url_pattern,
        drop_path_patterns=args.drop_path_pattern,
        drop_title_patterns=args.drop_title_pattern,
        drop_text_patterns=args.drop_text_pattern,
        min_alpha_ratio=args.min_alpha_ratio,
        max_digit_ratio=args.max_digit_ratio,
        max_repeat_line_ratio=args.max_repeat_line_ratio,
        verbose=args.verbose,
    )
    print(f"Wrote {len(docs)} documents to {args.output}")


if __name__ == "__main__":
    main()
