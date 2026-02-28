#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_hw2.chunking import ChunkingConfig, build_chunks


def parse_args() :
    p = argparse.ArgumentParser(description="Chunk cleaned documents into retrieval chunks.")
    p.add_argument("--documents", type=str, default="data/processed/documents.jsonl")
    p.add_argument("--output", type=str, default="data/processed/chunks.jsonl")
    p.add_argument("--strategy", choices=["fixed", "sentence"], default="fixed")
    p.add_argument("--chunk-size-words", type=int, default=250)
    p.add_argument("--overlap-words", type=int, default=50)
    p.add_argument("--min-chunk-words", type=int, default=20)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() :
    args = parse_args()
    cfg = ChunkingConfig(
        strategy=args.strategy,
        chunk_size_words=args.chunk_size_words,
        overlap_words=args.overlap_words,
        min_chunk_words=args.min_chunk_words,
    )
    chunks = build_chunks(args.documents, args.output, cfg, verbose=args.verbose)
    print(f"Wrote {len(chunks)} chunks to {args.output}")


if __name__ == "__main__":
    main()
