#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() :
    p = argparse.ArgumentParser(description="Run sparse/dense/hybrid/closedbook ablations and save predictions.")
    p.add_argument("--queries", required=True)
    p.add_argument("--out-dir", default="data/outputs/ablations")
    p.add_argument("--chunks", default="data/processed/chunks.jsonl")
    p.add_argument("--sparse-dir", default="data/indices/sparse_bm25")
    p.add_argument("--dense-dir", default="data/indices/dense_faiss")
    p.add_argument("--reader-backend", choices=["transformers"], default="transformers")
    p.add_argument("--reader-model", default="Qwen/Qwen2.5-14B-Instruct")
    p.add_argument("--reader-task", choices=["text-generation", "text2text-generation"], default="text-generation")
    p.add_argument("--modes", nargs="+", default=["sparse", "dense", "hybrid"])
    p.add_argument("--fusion-methods", nargs="+", default=["rrf", "weighted", "weighted_rrf", "combmnz"])
    p.add_argument("--fusion-alpha", type=float, default=0.5)
    p.add_argument("--rrf-k", type=int, default=60)
    p.add_argument("--dense-weight", type=float, default=0.5)
    p.add_argument("--sparse-weight", type=float, default=0.5)
    p.add_argument("--multi-query", action="store_true")
    p.add_argument("--multi-query-max", type=int, default=2)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--fetch-k-each", type=int, default=80)
    p.add_argument("--reranker-model", default="BAAI/bge-reranker-large")
    p.add_argument("--rerank-fetch-k", type=int, default=30)
    p.add_argument("--reranker-batch-size", type=int, default=16)
    p.add_argument("--reranker-max-length", type=int, default=512)
    p.add_argument("--reranker-device", default=None)
    p.add_argument("--references", default=None)
    return p.parse_args()


def main() :
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []

    runs = []
    for mode in args.modes:
        if mode == "hybrid":
            for fm in args.fusion_methods:
                runs.append((mode, fm))
        else:
            runs.append((mode, None))

    for mode, fusion in runs:
        name = mode if not fusion else f"{mode}_{fusion}"
        pred_path = out_dir / f"{name}.json"
        dbg_path = out_dir / f"{name}.debug.jsonl"
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "answer_queries.py"),
            "--run-name",
            str((out_dir / name).as_posix().replace("/", "_")),
            "--queries",
            args.queries,
            "--output",
            str(pred_path),
            "--debug-output",
            str(dbg_path),
            "--chunks",
            args.chunks,
            "--sparse-dir",
            args.sparse_dir,
            "--dense-dir",
            args.dense_dir,
            "--mode",
            mode,
            "--top-k",
            str(args.top_k),
            "--fetch-k-each",
            str(args.fetch_k_each),
            "--reader-backend",
            args.reader_backend,
            "--reader-task",
            args.reader_task,
        ]
        if fusion:
            cmd.extend(["--fusion-method", fusion])
            cmd.extend(["--fusion-alpha", str(args.fusion_alpha)])
            cmd.extend(["--rrf-k", str(args.rrf_k)])
            cmd.extend(["--dense-weight", str(args.dense_weight)])
            cmd.extend(["--sparse-weight", str(args.sparse_weight)])
        if args.multi_query:
            cmd.append("--multi-query")
            cmd.extend(["--multi-query-max", str(args.multi_query_max)])
        if args.reader_model:
            cmd.extend(["--reader-model", args.reader_model])
        if args.reranker_model:
            cmd.extend(["--reranker-model", args.reranker_model])
        if args.rerank_fetch_k is not None:
            cmd.extend(["--rerank-fetch-k", str(args.rerank_fetch_k)])
        if args.reranker_batch_size is not None:
            cmd.extend(["--reranker-batch-size", str(args.reranker_batch_size)])
        if args.reranker_max_length is not None:
            cmd.extend(["--reranker-max-length", str(args.reranker_max_length)])
        if args.reranker_device:
            cmd.extend(["--reranker-device", str(args.reranker_device)])
        print(f"[ablation] running {name}")
        subprocess.run(cmd, check=True, cwd=str(ROOT))
        row = {"run": name, "predictions": str(pred_path), "debug": str(dbg_path)}
        if args.references:
            eval_path = out_dir / f"{name}.scores.json"
            eval_cmd = [
                sys.executable,
                str(ROOT / "scripts" / "evaluate_predictions.py"),
                "--predictions",
                str(pred_path),
                "--references",
                args.references,
                "--output",
                str(eval_path),
            ]
            subprocess.run(eval_cmd, check=True, cwd=str(ROOT))
            row["scores"] = str(eval_path)
        records.append(row)

    csv_path = out_dir / "runs.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(records[0].keys()) if records else ["run"])
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote ablation run manifest to {csv_path}")


if __name__ == "__main__":
    main()
