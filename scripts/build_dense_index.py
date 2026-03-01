#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_hw2.retrieval.chunk_store import load_chunk_store_from_jsonl
from rag_hw2.retrieval.dense_faiss import build_dense_faiss_index


def _load_embed_text_map(path, text_key) :
    m = {}
    p = Path(path)
    if not p.exists():
        return m
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cid = str(row.get("chunk_id", "")).strip()
            txt = str(row.get(text_key, "")).strip()
            if cid and txt:
                m[cid] = txt
    return m


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
    p.add_argument("--embed-texts-jsonl", type=str, default=None, help="Optional JSONL with {chunk_id, text} used as dense embedding text.")
    p.add_argument("--embed-text-key", type=str, default="text", help="Field in --embed-texts-jsonl used for embedding text.")
    p.add_argument("--embedding-text-mode", choices=["chunk", "hypo_question"], default="chunk")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() :
    args = parse_args()
    store = load_chunk_store_from_jsonl(args.chunks)
    embed_texts_by_chunk_id = None
    if args.embed_texts_jsonl:
        embed_texts_by_chunk_id = _load_embed_text_map(args.embed_texts_jsonl, args.embed_text_key)
        if args.verbose:
            print(f"[dense] loaded {len(embed_texts_by_chunk_id)} alternate texts from {args.embed_texts_jsonl}")
    idx = build_dense_faiss_index(
        store,
        model_name=args.embedding_model,
        batch_size=args.batch_size,
        normalize_embeddings=not args.no_normalize,
        embed_texts_by_chunk_id=embed_texts_by_chunk_id,
        embedding_text_mode=args.embedding_text_mode,
        verbose=args.verbose,
    )
    idx.save(args.out_dir)
    print(f"Built dense FAISS index for {len(store.chunks)} chunks at {args.out_dir}")


if __name__ == "__main__":
    main()
