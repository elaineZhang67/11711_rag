#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _normalize_ws(text):
    return re.sub(r"\s+", " ", (text or "")).strip()


def _load_questions(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [_normalize_ws(x) for x in lines if _normalize_ws(x)]


def _load_refs(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(k): str(v).strip() for k, v in data.items()}


def _build_pairs(questions, refs):
    pairs = []
    missing = 0
    for i, q in enumerate(questions, start=1):
        ans = refs.get(str(i), "")
        if not ans:
            missing += 1
        pairs.append((str(i), q, ans))
    if missing:
        print(f"[warn] {missing} question(s) do not have matching reference answers by id.")
    return pairs


def _variants_for_question(q):
    q = _normalize_ws(q)
    out = []

    m = re.match(r"^When was (.+?) (founded|established|created|opened|launched|chartered)\?$", q, flags=re.IGNORECASE)
    if m:
        ent = m.group(1).strip()
        verb = m.group(2).lower()
        out.extend(
            [
                f"In what year was {ent} {verb}?",
                f"What year was {ent} {verb}?",
                f"When did {ent} get {verb}?",
            ]
        )
        return out

    m = re.match(r"^When did (.+?) begin\?$", q, flags=re.IGNORECASE)
    if m:
        ent = m.group(1).strip()
        out.extend(
            [
                f"What year did {ent} begin?",
                f"When did {ent} start?",
                f"In what year did {ent} begin?",
            ]
        )
        return out

    m = re.match(r"^Where is (.+?) located\?$", q, flags=re.IGNORECASE)
    if m:
        ent = m.group(1).strip()
        out.extend(
            [
                f"What is the location of {ent}?",
                f"Where can {ent} be found?",
                f"Which location is {ent} in?",
            ]
        )
        return out

    m = re.match(r"^Where is (.+?) headquartered\?$", q, flags=re.IGNORECASE)
    if m:
        ent = m.group(1).strip()
        out.extend(
            [
                f"What is the headquarters location of {ent}?",
                f"Where is the headquarters of {ent}?",
            ]
        )
        return out

    m = re.match(r"^Who founded (.+?)\?$", q, flags=re.IGNORECASE)
    if m:
        ent = m.group(1).strip()
        out.extend(
            [
                f"Who is the founder of {ent}?",
                f"Who was the founder of {ent}?",
                f"Which person founded {ent}?",
            ]
        )
        return out

    m = re.match(r"^What is the mascot of (.+?)\?$", q, flags=re.IGNORECASE)
    if m:
        ent = m.group(1).strip()
        out.extend(
            [
                f"What mascot does {ent} use?",
                f"What is {ent}'s mascot?",
            ]
        )
        return out

    m = re.match(r"^What is (.+?) known as\?$", q, flags=re.IGNORECASE)
    if m:
        ent = m.group(1).strip()
        out.extend(
            [
                f"What is another name for {ent}?",
                f"How is {ent} commonly known?",
            ]
        )
        return out

    # Fallback: keep augmentation conservative; avoid low-quality rewrites.
    return out


def _dedupe_questions(items):
    out = []
    seen = set()
    for x in items:
        key = _normalize_ws(x).lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(_normalize_ws(x))
    return out


def _write_questions(questions, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(questions) + ("\n" if questions else ""), encoding="utf-8")


def _write_refs(refs, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(refs, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(rows, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    p = argparse.ArgumentParser(description="Create rule-based augmented training QA files.")
    p.add_argument("--questions", default="data/train/questions.txt")
    p.add_argument("--references", default="data/train/reference_answers.json")
    p.add_argument("--out-dir", default="data/train/augmented")
    p.add_argument("--max-aug-per-question", type=int, default=3)
    args = p.parse_args()

    base_questions = _load_questions(args.questions)
    base_refs = _load_refs(args.references)
    pairs = _build_pairs(base_questions, base_refs)

    aug_questions = []
    aug_refs = {}
    mapping_rows = []

    next_id = 1
    for src_id, q, a in pairs:
        if not a:
            continue
        variants = _variants_for_question(q)
        variants = _dedupe_questions(variants)[: max(0, int(args.max_aug_per_question))]
        for v in variants:
            if v.lower() == q.lower():
                continue
            aug_qid = str(next_id)
            next_id += 1
            aug_questions.append(v)
            aug_refs[aug_qid] = a
            mapping_rows.append(
                {
                    "aug_qid": aug_qid,
                    "source_qid": src_id,
                    "source_question": q,
                    "aug_question": v,
                    "answer": a,
                }
            )

    # Combined dataset (base + augmented)
    combined_questions = []
    combined_refs = {}
    cid = 1
    for _src_id, q, a in pairs:
        if not a:
            continue
        combined_questions.append(q)
        combined_refs[str(cid)] = a
        cid += 1
    for q, a in zip(aug_questions, [aug_refs[str(i)] for i in range(1, len(aug_questions) + 1)]):
        combined_questions.append(q)
        combined_refs[str(cid)] = a
        cid += 1

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_questions(aug_questions, out / "questions.txt")
    _write_refs(aug_refs, out / "reference_answers.json")
    _write_jsonl(mapping_rows, out / "mapping.jsonl")

    _write_questions(combined_questions, out / "combined_questions.txt")
    _write_refs(combined_refs, out / "combined_reference_answers.json")

    print(f"Wrote augmented questions: {out / 'questions.txt'} ({len(aug_questions)})")
    print(f"Wrote augmented refs: {out / 'reference_answers.json'} ({len(aug_refs)})")
    print(f"Wrote mapping: {out / 'mapping.jsonl'} ({len(mapping_rows)})")
    print(f"Wrote combined questions: {out / 'combined_questions.txt'} ({len(combined_questions)})")
    print(f"Wrote combined refs: {out / 'combined_reference_answers.json'} ({len(combined_refs)})")


if __name__ == "__main__":
    main()

