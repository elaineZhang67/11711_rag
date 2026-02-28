#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_hw2.eval.metrics import score_predictions


def _write_json(obj, path, indent= 2) :
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=indent), encoding="utf-8")


def parse_args() :
    p = argparse.ArgumentParser(description="Evaluate predictions against reference answers.")
    p.add_argument("--predictions", type=str, required=True)
    p.add_argument("--references", type=str, required=True, help="JSON mapping qid -> answer or [answers].")
    p.add_argument("--output", type=str, default=None, help="Optional JSON file for per-question scores.")
    return p.parse_args()


def _load_preds(path) :
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(k): str(v) for k, v in data.items() if k != "andrewid"}


def _load_refs(path) :
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(k): v for k, v in data.items()}


def main() :
    args = parse_args()
    preds = _load_preds(args.predictions)
    refs = _load_refs(args.references)
    summary, rows = score_predictions(preds, refs)
    print(f"n={summary.n}")
    print(f"EM={summary.em:.4f}")
    print(f"F1={summary.f1:.4f}")
    print(f"ROUGE-L={summary.rouge_l:.4f}")
    if args.output:
        _write_json(
            {
                "summary": {
                    "n": summary.n,
                    "em": summary.em,
                    "f1": summary.f1,
                    "rouge_l": summary.rouge_l,
                },
                "rows": rows,
            },
            args.output,
            indent=2,
        )
        print(f"Wrote detailed scores to {args.output}")


if __name__ == "__main__":
    main()
