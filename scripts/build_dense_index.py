#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_hw2.retrieval.chunk_store import load_chunk_store_from_jsonl
from rag_hw2.retrieval.dense_faiss import build_dense_faiss_index


def parse_args() :
    p = argparse.ArgumentParser(description="Build dense FAISS retrieval index.")
    p.add_argument("--chunks", type=str, default="data/processed/chunks.jsonl")
    p.add_argument("--out-dir", type=str, default="data/indices/dense_faiss")
    p.add_argument(
        "--embedding-model",
        type=str,
        default="BAAI/bge-large-en-v1.5",
        help="SentenceTransformers model name",
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--no-normalize", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() :
    args = parse_args()
    store = load_chunk_store_from_jsonl(args.chunks)
    idx = build_dense_faiss_index(
        store,
        model_name=args.embedding_model,
        batch_size=args.batch_size,
        normalize_embeddings=not args.no_normalize,
        verbose=args.verbose,
    )
    idx.save(args.out_dir)
    print(f"Built dense FAISS index for {len(store.chunks)} chunks at {args.out_dir}")


if __name__ == "__main__":
    main()
