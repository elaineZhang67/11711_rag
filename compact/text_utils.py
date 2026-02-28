from __future__ import annotations

import re
import unicodedata
from collections import Counter

_WS_RE = re.compile(r"\s+")
_NONWORD_RE = re.compile(r"[^A-Za-z0-9]+")


def normalize_unicode(text) :
    return unicodedata.normalize("NFKC", text)


def normalize_whitespace(text) :
    return _WS_RE.sub(" ", text).strip()


def clean_text_basic(text) :
    text = normalize_unicode(text)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def bm25_tokenize(text) :
    text = normalize_unicode(text).lower()
    text = _NONWORD_RE.sub(" ", text)
    return [t for t in text.split() if t]


def simple_sentence_split(text) :
    # Lightweight fallback without external tokenizers.
    text = text.strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", text)
    return [p.strip() for p in parts if p.strip()]


def word_count(text) :
    return len(text.split())


def dedupe_texts_exact(texts) :
    seen = Counter()
    keep = []
    for i, t in enumerate(texts):
        key = normalize_whitespace(t.lower())
        if seen[key] == 0:
            keep.append(i)
        seen[key] += 1
    return keep


def normalize_answer_for_eval(s) :
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return normalize_whitespace(s)

