#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_hw2.retrieval.chunk_store import load_chunk_store_from_jsonl
from rag_hw2.retrieval.sparse_bm25 import build_sparse_bm25_index


def parse_args() :
    p = argparse.ArgumentParser(description="Build BM25 sparse retrieval index.")
    p.add_argument("--chunks", type=str, default="data/processed/chunks.jsonl")
    p.add_argument("--out-dir", type=str, default="data/indices/sparse_bm25")
    return p.parse_args()


def main() :
    args = parse_args()
    store = load_chunk_store_from_jsonl(args.chunks)
    idx = build_sparse_bm25_index(store)
    idx.save(args.out_dir)
    print(f"Built sparse BM25 index for {len(store.chunks)} chunks at {args.out_dir}")


if __name__ == "__main__":
    main()
