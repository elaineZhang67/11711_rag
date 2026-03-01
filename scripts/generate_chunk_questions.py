#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from transformers import pipeline as hf_pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_hw2.retrieval.chunk_store import load_chunk_store_from_jsonl

_WS_RE = re.compile(r"\s+")
_QUOTE_RE = re.compile(r'^[\"\'“”‘’`]+|[\"\'“”‘’`]+$')


def _normalize_ws(text) :
    return _WS_RE.sub(" ", (text or "")).strip()


def _clean_question(raw) :
    q = _normalize_ws(raw)
    if not q:
        return q
    if q.lower().startswith("question:"):
        q = q[len("question:") :].strip()
    q = q.splitlines()[0].strip()
    q = _QUOTE_RE.sub("", q).strip()
    q = q.strip(" .,:;-")
    words = q.split()
    if len(words) > 24:
        q = " ".join(words[:24]).strip()
    if q and not q.endswith("?"):
        q = q + "?"
    return q


def _parse_device(device_arg) :
    if device_arg is None:
        return None
    if device_arg.isdigit():
        return int(device_arg)
    if device_arg.lower() == "cpu":
        return -1
    return device_arg


def parse_args() :
    p = argparse.ArgumentParser(description="Generate one synthetic question per chunk for dense question-vector indexing.")
    p.add_argument("--chunks", type=str, default="data/processed/chunks_sentence.jsonl")
    p.add_argument("--output", type=str, default="data/processed/chunk_questions.jsonl")
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-14B-Instruct")
    p.add_argument("--task", choices=["text-generation", "text2text-generation"], default="text-generation")
    p.add_argument("--device", type=str, default=None, help="e.g. 0, cpu, or cuda:0")
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--max-chars", type=int, default=900)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() :
    args = parse_args()
    store = load_chunk_store_from_jsonl(args.chunks)
    chunks = store.chunks
    if args.limit:
        chunks = chunks[: args.limit]

    kwargs = {
        "task": args.task,
        "model": args.model,
        "tokenizer": args.model,
    }
    device = _parse_device(args.device)
    if device is not None:
        kwargs["device"] = device
    pipe = hf_pipeline(**kwargs)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for i, c in enumerate(chunks, start=1):
            snippet = _normalize_ws(c.text)[: args.max_chars]
            prompt = (
                "Write one factual question that can be answered by this passage.\n"
                "Return only the question.\n\n"
                f"Passage: {snippet}\n"
                "Question:"
            )
            out = pipe(prompt, max_new_tokens=args.max_new_tokens, do_sample=False)
            raw = ""
            if out:
                raw = out[0].get("generated_text") or out[0].get("summary_text") or ""
            if args.task == "text-generation" and raw.startswith(prompt):
                raw = raw[len(prompt) :]
            q = _clean_question(raw)
            if not q:
                q = "What is this passage about?"
            row = {
                "chunk_id": c.chunk_id,
                "text": q,
                "doc_id": c.doc_id,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if args.verbose and (i % 200 == 0 or i == len(chunks)):
                print(f"[{i}/{len(chunks)}] {c.chunk_id} -> {q}")

    print(f"Wrote {len(chunks)} generated questions to {out_path}")


if __name__ == "__main__":
    main()
