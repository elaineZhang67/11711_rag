#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_preds(path: str | Path) -> dict[str, str]:
    data = _read_json(path)
    return {str(k): str(v) for k, v in data.items() if k != "andrewid"}


def _load_refs(path: str | Path) -> dict[str, str | list[str]]:
    data = _read_json(path)
    return {str(k): v for k, v in data.items()}


def cmd_list_sources(args: argparse.Namespace) -> None:
    from compact.collect_compact import list_recommended_sources

    jobs = list_recommended_sources(args.groups, args.include_source, args.exclude_source)
    for job in jobs:
        print(f"{job.group:>7}  {job.name:<32}  pages={job.max_pages:<4} depth={job.max_depth}  seeds={len(job.seed_urls)}")
        for seed in job.seed_urls:
            print(f"         - {seed}")


def cmd_scrape(args: argparse.Namespace) -> None:
    from compact.collect_compact import run_recommended_crawls

    summary = run_recommended_crawls(
        out_root=args.out_root,
        groups=args.groups,
        include_names=args.include_source,
        exclude_names=args.exclude_source,
        max_pages_multiplier=args.max_pages_multiplier,
        sleep_sec=args.sleep_sec,
        verbose=args.verbose,
    )
    print(f"Wrote crawl summary to {Path(args.out_root) / 'crawl_summary.json'}")
    for row in summary:
        print(f"{row['group']}/{row['name']}: ok={row['ok']} html={row['html']} pdf={row['pdf']} skipped={row['skipped']} errors={row['errors']}")


def cmd_build_data(args: argparse.Namespace) -> None:
    from compact.data_compact import ChunkingConfig, build_chunks, build_corpus

    docs = build_corpus(
        input_dirs=args.input_dir,
        output_jsonl=args.documents_output,
        min_chars=args.min_chars,
        dedupe_exact=not args.no_dedupe,
        verbose=args.verbose,
    )
    print(f"Wrote {len(docs)} documents to {args.documents_output}")
    if args.skip_chunks:
        return
    cfg = ChunkingConfig(
        strategy=args.chunk_strategy,
        chunk_size_words=args.chunk_size_words,
        overlap_words=args.overlap_words,
        min_chunk_words=args.min_chunk_words,
    )
    chunks = build_chunks(args.documents_output, args.chunks_output, cfg, verbose=args.verbose)
    print(f"Wrote {len(chunks)} chunks to {args.chunks_output}")


def cmd_build_indices(args: argparse.Namespace) -> None:
    from compact.retrieval_compact import ChunkStore, DenseFAISSIndex, SparseBM25Index

    store = ChunkStore.from_jsonl(args.chunks)
    print(f"Loaded {len(store.chunks)} chunks from {args.chunks}")
    if not args.skip_sparse:
        s = SparseBM25Index.build(store)
        s.save(args.sparse_dir)
        print(f"Built sparse BM25 index at {args.sparse_dir}")
    if not args.skip_dense:
        d = DenseFAISSIndex.build(
            store,
            model_name=args.embedding_model,
            batch_size=args.batch_size,
            normalize_embeddings=not args.no_normalize,
            verbose=args.verbose,
        )
        d.save(args.dense_dir)
        print(f"Built dense FAISS index at {args.dense_dir}")


def cmd_answer(args: argparse.Namespace) -> None:
    from compact.io_utils import write_json, write_jsonl
    from compact.qa_compact import (
        RAGPipeline,
        RetrievalConfig,
        build_submission_json,
        load_queries,
        make_reader,
    )
    from compact.retrieval_compact import HybridRetriever

    queries = load_queries(args.queries)
    retriever = None
    if args.mode != "closedbook":
        sparse_dir = args.sparse_dir if args.mode in {"sparse", "hybrid"} else None
        dense_dir = args.dense_dir if args.mode in {"dense", "hybrid"} else None
        retriever = HybridRetriever.from_dirs(args.chunks, sparse_dir=sparse_dir, dense_dir=dense_dir)

    device = None
    if args.device is not None:
        if args.device.isdigit():
            device = int(args.device)
        elif args.device.lower() == "cpu":
            device = -1
        else:
            device = args.device

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
        ),
    )

    preds: dict[str, str] = {}
    debug_rows: list[dict] = []
    for i, q in enumerate(queries, start=1):
        row = pipeline.answer_query(q.qid, q.question)
        preds[q.qid] = row["answer"]
        debug_rows.append(row)
        if args.verbose:
            print(f"[{i}/{len(queries)}] {q.qid}: {q.question}\n  -> {row['answer']}")

    write_json(build_submission_json(preds, andrewid=args.andrewid), args.output, indent=2)
    print(f"Wrote predictions to {args.output}")
    if args.debug_output:
        write_jsonl(debug_rows, args.debug_output)
        print(f"Wrote debug traces to {args.debug_output}")


def cmd_eval(args: argparse.Namespace) -> None:
    from compact.io_utils import write_json
    from compact.qa_compact import score_predictions

    preds = _load_preds(args.predictions)
    refs = _load_refs(args.references)
    summary, rows = score_predictions(preds, refs)
    print(f"n={summary.n}")
    print(f"EM={summary.em:.4f}")
    print(f"F1={summary.f1:.4f}")
    print(f"ROUGE-L={summary.rouge_l:.4f}")
    if args.output:
        write_json({"summary": summary.__dict__, "rows": rows}, args.output, indent=2)
        print(f"Wrote detailed scores to {args.output}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compact CLI using merged implementation files.")
    sp = p.add_subparsers(dest="cmd", required=True)

    p_ls = sp.add_parser("list-sources")
    p_ls.add_argument("--groups", nargs="+", choices=["general", "events", "culture", "food", "sports"])
    p_ls.add_argument("--include-source", action="append", default=None)
    p_ls.add_argument("--exclude-source", action="append", default=None)
    p_ls.set_defaults(func=cmd_list_sources)

    p_sc = sp.add_parser("scrape")
    p_sc.add_argument("--out-root", default="data/raw/recommended_sources")
    p_sc.add_argument("--groups", nargs="+", choices=["general", "events", "culture", "food", "sports"])
    p_sc.add_argument("--include-source", action="append", default=None)
    p_sc.add_argument("--exclude-source", action="append", default=None)
    p_sc.add_argument("--max-pages-multiplier", type=float, default=1.0)
    p_sc.add_argument("--sleep-sec", type=float, default=None)
    p_sc.add_argument("--verbose", action="store_true")
    p_sc.set_defaults(func=cmd_scrape)

    p_bd = sp.add_parser("build-data")
    p_bd.add_argument("--input-dir", action="append", required=True)
    p_bd.add_argument("--documents-output", default="data/processed/documents.jsonl")
    p_bd.add_argument("--chunks-output", default="data/processed/chunks.jsonl")
    p_bd.add_argument("--min-chars", type=int, default=100)
    p_bd.add_argument("--no-dedupe", action="store_true")
    p_bd.add_argument("--skip-chunks", action="store_true")
    p_bd.add_argument("--chunk-strategy", choices=["fixed", "sentence"], default="fixed")
    p_bd.add_argument("--chunk-size-words", type=int, default=250)
    p_bd.add_argument("--overlap-words", type=int, default=50)
    p_bd.add_argument("--min-chunk-words", type=int, default=20)
    p_bd.add_argument("--verbose", action="store_true")
    p_bd.set_defaults(func=cmd_build_data)

    p_bi = sp.add_parser("build-indices")
    p_bi.add_argument("--chunks", default="data/processed/chunks.jsonl")
    p_bi.add_argument("--sparse-dir", default="data/indices/sparse_bm25")
    p_bi.add_argument("--dense-dir", default="data/indices/dense_faiss")
    p_bi.add_argument("--skip-sparse", action="store_true")
    p_bi.add_argument("--skip-dense", action="store_true")
    p_bi.add_argument("--embedding-model", default="BAAI/bge-large-en-v1.5")
    p_bi.add_argument("--batch-size", type=int, default=64)
    p_bi.add_argument("--no-normalize", action="store_true")
    p_bi.add_argument("--verbose", action="store_true")
    p_bi.set_defaults(func=cmd_build_indices)

    p_an = sp.add_parser("answer")
    p_an.add_argument("--queries", required=True)
    p_an.add_argument("--output", required=True)
    p_an.add_argument("--debug-output", default=None)
    p_an.add_argument("--chunks", default="data/processed/chunks.jsonl")
    p_an.add_argument("--sparse-dir", default="data/indices/sparse_bm25")
    p_an.add_argument("--dense-dir", default="data/indices/dense_faiss")
    p_an.add_argument("--mode", choices=["sparse", "dense", "hybrid", "closedbook"], default="hybrid")
    p_an.add_argument("--top-k", type=int, default=3)
    p_an.add_argument("--fetch-k-each", type=int, default=30)
    p_an.add_argument("--fusion-method", choices=["rrf", "weighted"], default="rrf")
    p_an.add_argument("--fusion-alpha", type=float, default=0.5)
    p_an.add_argument("--reader-backend", choices=["heuristic", "transformers"], default="transformers")
    p_an.add_argument("--reader-model", default="Qwen/Qwen2.5-14B-Instruct")
    p_an.add_argument("--reader-task", choices=["text-generation", "text2text-generation"], default="text-generation")
    p_an.add_argument("--max-new-tokens", type=int, default=120)
    p_an.add_argument("--temperature", type=float, default=0.0)
    p_an.add_argument("--device", default=None)
    p_an.add_argument("--max-context-chars", type=int, default=5000)
    p_an.add_argument("--andrewid", default="Venonat1")
    p_an.add_argument("--verbose", action="store_true")
    p_an.set_defaults(func=cmd_answer)

    p_ev = sp.add_parser("eval")
    p_ev.add_argument("--predictions", required=True)
    p_ev.add_argument("--references", required=True)
    p_ev.add_argument("--output", default=None)
    p_ev.set_defaults(func=cmd_eval)
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
