#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


_WS_RE = re.compile(r"\s+")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_SITE_SUFFIX_RE = re.compile(r"\s*[\-|–—]\s*(Carnegie Mellon University|CMU|Visit Pittsburgh|Wikipedia)\s*$", re.IGNORECASE)


def _normalize_ws(text):
    return _WS_RE.sub(" ", (text or "")).strip()


def _clean_answer(text):
    text = _normalize_ws(text)
    text = re.sub(r"\[[0-9]+\]", "", text).strip()
    text = text.strip(" \t,;:.")
    return _normalize_ws(text)


def _split_sentences(text):
    text = _normalize_ws(text)
    if not text:
        return []
    return [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]


def _entity_from_doc(doc):
    title = _normalize_ws(doc.get("title", "") or "")
    if "|" in title:
        title = title.split("|", 1)[0].strip()
    if title:
        title = _SITE_SUFFIX_RE.sub("", title).strip()
    if " - " in title and len(title.split(" - ")) > 1:
        title = title.split(" - ", 1)[0].strip()
    if title:
        return title
    url = (doc.get("source_url", "") or "").strip("/")
    if not url:
        return "this organization"
    slug = url.split("/")[-1]
    slug = re.sub(r"[-_]+", " ", slug)
    slug = re.sub(r"\.(html?|php)$", "", slug, flags=re.IGNORECASE)
    slug = _normalize_ws(slug).strip()
    if slug:
        return slug.title()
    return "this organization"


def _is_scraped_doc(doc):
    sp = doc.get("source_path", "") or ""
    return "data/raw/" in sp or "data-raw/" in sp


def _looks_good_answer(ans):
    if not ans:
        return False
    if len(ans) < 2:
        return False
    if len(ans.split()) > 12:
        return False
    if "|" in ans:
        return False
    if "http" in ans.lower():
        return False
    if "<" in ans or ">" in ans:
        return False
    low = ans.lower()
    bad = [
        "click here",
        "read more",
        "learn more",
        "copyright",
        "all rights reserved",
    ]
    return not any(b in low for b in bad)


def _looks_good_entity(entity):
    if not entity:
        return False
    if len(entity.split()) > 9:
        return False
    low = entity.lower()
    bad = [
        "blog",
        "insider",
        "guide",
        "visit pittsburgh",
        "wikipedia",
        "about us",
        "things to do",
        "best ",
        "top ",
        "free things",
        "contact us",
        "privacy",
        "terms",
        "cookie",
        "site map",
        "sitemap",
        "index",
    ]
    return not any(b in low for b in bad)


def _qa_patterns_for_sentence(entity, sent):
    out = []
    s = sent

    m = re.search(
        r"\b(?:founded|established|formed|created|chartered|opened|launched|began)\b[^.]{0,80}?\b(?:in|on)\s+((?:[A-Z][a-z]+\s+\d{1,2},\s+\d{4})|(?:\d{4}))\b",
        s,
        flags=re.IGNORECASE,
    )
    if m:
        year_or_date = _clean_answer(m.group(1))
        verb = re.search(r"\b(founded|established|formed|created|chartered|opened|launched|began)\b", s, flags=re.IGNORECASE)
        v = verb.group(1).lower() if verb else "founded"
        if v in {"founded", "established", "formed", "created", "chartered", "opened"}:
            q = f"When was {entity} {v}?"
        elif v == "began":
            q = f"When did {entity} begin?"
        elif v == "launched":
            q = f"When was {entity} launched?"
        else:
            q = f"When was {entity} founded?"
        out.append((q, year_or_date, "date"))

    m = re.search(r"\bis located in\s+([^.;\n]{3,100})", s, flags=re.IGNORECASE)
    if m:
        out.append((f"Where is {entity} located?", _clean_answer(m.group(1)), "location"))

    m = re.search(r"\bis located at\s+([^.;\n]{3,100})", s, flags=re.IGNORECASE)
    if m:
        out.append((f"Where is {entity} located?", _clean_answer(m.group(1)), "location"))

    m = re.search(r"\bheadquartered in\s+([^.;\n]{3,100})", s, flags=re.IGNORECASE)
    if m:
        out.append((f"Where is {entity} headquartered?", _clean_answer(m.group(1)), "location"))

    m = re.search(r"\bfounded by\s+([^.;\n]{2,100})", s, flags=re.IGNORECASE)
    if m:
        out.append((f"Who founded {entity}?", _clean_answer(m.group(1)), "person"))

    m = re.search(r"\bmascot(?:\s+name)?\s+(?:is|was)\s+([^.;\n]{2,80})", s, flags=re.IGNORECASE)
    if m:
        out.append((f"What is the mascot of {entity}?", _clean_answer(m.group(1)), "entity"))

    return out


def generate_examples(documents_jsonl, max_examples=300):
    docs = []
    with Path(documents_jsonl).open("r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if _is_scraped_doc(d):
                docs.append(d)

    questions = []
    answers = []
    seen_q = set()
    seen_pair = set()

    for d in docs:
        entity = _entity_from_doc(d)
        if not _looks_good_entity(entity):
            continue
        sents = _split_sentences(d.get("text", ""))
        # Use early/middle sentences for better factual density and less boilerplate tails.
        focus = sents[:40] + sents[80:120]
        for s in focus:
            for q, a, _kind in _qa_patterns_for_sentence(entity, s):
                q = _normalize_ws(q)
                a = _clean_answer(a)
                if not _looks_good_answer(a):
                    continue
                if not q.endswith("?"):
                    q = q + "?"
                key = (q.lower(), a.lower())
                if q.lower() in seen_q:
                    continue
                if key in seen_pair:
                    continue
                seen_q.add(q.lower())
                seen_pair.add(key)
                questions.append(q)
                answers.append(a)
                if len(questions) >= max_examples:
                    return questions, answers
    return questions, answers


def main():
    p = argparse.ArgumentParser(description="Generate training questions/answers from scraped documents (not leaderboard queries).")
    p.add_argument("--documents", default="data/processed/documents.jsonl")
    p.add_argument("--out-dir", default="data/train")
    p.add_argument("--max-examples", type=int, default=300)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    questions, answers = generate_examples(args.documents, max_examples=args.max_examples)

    q_path = out_dir / "questions.txt"
    ref_path = out_dir / "reference_answers.json"

    q_path.write_text("\n".join(questions) + ("\n" if questions else ""), encoding="utf-8")
    refs = {str(i + 1): answers[i] for i in range(len(answers))}
    ref_path.write_text(json.dumps(refs, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {len(questions)} questions to {q_path}")
    print(f"Wrote {len(refs)} references to {ref_path}")


if __name__ == "__main__":
    main()
