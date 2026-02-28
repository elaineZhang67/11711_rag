#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_hw2.pipeline import RAGPipeline, RetrievalConfig
from rag_hw2.query_io import build_submission_json, load_queries
from rag_hw2.reader.qa_reader import make_reader
from rag_hw2.retrieval.hybrid import HybridRetriever, build_hybrid_retriever_from_dirs
from rag_hw2.retrieval.reranker import CrossEncoderReranker


def _write_json(obj, path, indent= 2) :
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=indent), encoding="utf-8")


def _write_jsonl(rows, path) :
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() :
    p = argparse.ArgumentParser(description="Answer a JSON query file with sparse/dense/hybrid RAG.")
    p.add_argument("--queries", type=str, required=True, help="Input JSON list/dict of queries.")
    p.add_argument("--output", type=str, required=True, help="Output predictions JSON.")
    p.add_argument("--debug-output", type=str, default=None, help="Optional JSONL with retrieval traces.")
    p.add_argument("--chunks", type=str, default="data/processed/chunks.jsonl")
    p.add_argument("--sparse-dir", type=str, default="data/indices/sparse_bm25")
    p.add_argument("--dense-dir", type=str, default="data/indices/dense_faiss")
    p.add_argument("--mode", choices=["sparse", "dense", "hybrid", "closedbook"], default="hybrid")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--fetch-k-each", type=int, default=80)
    p.add_argument("--fusion-method", choices=["rrf", "weighted"], default="rrf")
    p.add_argument("--fusion-alpha", type=float, default=0.5)
    p.add_argument("--reranker-model", type=str, default="BAAI/bge-reranker-large", help="Optional cross-encoder reranker model.")
    p.add_argument("--rerank-fetch-k", type=int, default=30, help="Retrieve this many candidates before reranking.")
    p.add_argument("--reranker-batch-size", type=int, default=16)
    p.add_argument("--reranker-max-length", type=int, default=512)
    p.add_argument("--reranker-device", type=str, default=None, help="Reranker device, e.g., cpu or cuda:0.")
    p.add_argument("--reader-backend", choices=["transformers", "heuristic"], default="transformers")
    p.add_argument("--reader-model", type=str, default="Qwen/Qwen2.5-14B-Instruct")
    p.add_argument("--reader-task", choices=["text-generation", "text2text-generation"], default="text-generation")
    p.add_argument("--max-new-tokens", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--device", type=str, default=None, help="Transformers pipeline device (e.g., 0 or cpu).")
    p.add_argument("--max-context-chars", type=int, default=5000)
    p.add_argument("--andrewid", type=str, default="Venonat1", help="Include in output JSON (leaderboard format).")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() :
    args = parse_args()
    queries = load_queries(args.queries)

    retriever = None
    if args.mode != "closedbook":
        sparse_dir = args.sparse_dir if args.mode in {"sparse", "hybrid"} else None
        dense_dir = args.dense_dir if args.mode in {"dense", "hybrid"} else None
        retriever = build_hybrid_retriever_from_dirs(args.chunks, sparse_dir=sparse_dir, dense_dir=dense_dir)

    device = None
    if args.device is not None:
        if args.device.isdigit():
            device = int(args.device)
        elif args.device.lower() == "cpu":
            device = -1
        else:
            device = args.device

    reranker = None
    if args.reranker_model:
        reranker_device = args.reranker_device
        if reranker_device is None and args.device is not None:
            if args.device.isdigit():
                reranker_device = f"cuda:{args.device}"
            elif args.device.lower() == "cpu":
                reranker_device = "cpu"
            else:
                reranker_device = args.device
        reranker = CrossEncoderReranker(
            model_name=args.reranker_model,
            device=reranker_device,
            batch_size=args.reranker_batch_size,
            max_length=args.reranker_max_length,
        )

    reader = make_reader(
        backend=args.reader_backend,
        model_name=args.reader_model,
        task=args.reader_task,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        device=device,
        max_context_chars=args.max_context_chars,
    )
    pipeline = RAGPipeline(
        retriever=retriever,
        reader=reader,
        retrieval_cfg=RetrievalConfig(
            mode=args.mode,
            top_k=args.top_k,
            fetch_k_each=args.fetch_k_each,
            fusion_method=args.fusion_method,
            fusion_alpha=args.fusion_alpha,
            rerank_fetch_k=args.rerank_fetch_k,
        ),
        reranker=reranker,
    )

    preds = {}
    debug_rows = []
    for i, q in enumerate(queries, start=1):
        row = pipeline.answer_query(q.qid, q.question)
        preds[q.qid] = row["answer"]
        debug_rows.append(row)
        if args.verbose:
            print(f"[{i}/{len(queries)}] {q.qid}: {q.question}\n  -> {row['answer']}")

    submission = build_submission_json(preds, andrewid=args.andrewid)
    _write_json(submission, args.output, indent=2)
    if args.debug_output:
        _write_jsonl(debug_rows, args.debug_output)
    print(f"Wrote predictions to {args.output}")
    if args.debug_output:
        print(f"Wrote debug traces to {args.debug_output}")


if __name__ == "__main__":
    main()
