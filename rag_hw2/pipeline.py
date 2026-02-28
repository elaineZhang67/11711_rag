from __future__ import annotations

import re
from typing import Any

from rag_hw2.reader.qa_reader import Reader, postprocess_answer
from rag_hw2.retrieval.hybrid import HybridRetriever
from rag_hw2.types import RetrievedChunk

_WS_RE = re.compile(r"\s+")
_YEAR_RE = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
_DATE_RE = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|"
    r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?\s+\d{1,2},?\s+\d{4}\b",
    re.IGNORECASE,
)
_NAME_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-zA-Z'.-]+){1,4})\b")
_STOPWORDS = {
    "what",
    "which",
    "when",
    "where",
    "who",
    "whom",
    "whose",
    "that",
    "this",
    "with",
    "from",
    "about",
    "into",
    "after",
    "before",
    "between",
    "their",
    "there",
    "these",
    "those",
    "have",
    "has",
    "were",
    "was",
    "been",
    "being",
    "and",
    "the",
    "for",
    "are",
    "but",
    "not",
    "did",
    "does",
    "was",
    "is",
    "at",
    "in",
    "on",
    "by",
    "of",
    "to",
}
_MONTHS = {
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
}
_ORG_TAIL_WORDS = {
    "University",
    "School",
    "Institute",
    "College",
    "Center",
    "Centre",
    "Department",
    "Hall",
    "Museum",
    "Hospital",
    "Library",
}
_LIST_NOISE_HINTS = [
    "list of",
    "/list",
    "directory",
    "/directory",
    "/people",
    "/faculty",
    "/staff",
]


class RetrievalConfig:
    def __init__(
        self,
        mode= "hybrid",
        top_k= 5,
        fetch_k_each= 30,
        fusion_method= "rrf",
        fusion_alpha= 0.5,
        rerank_fetch_k= None,
    ) :
        self.mode = mode  # sparse | dense | hybrid | closedbook
        self.top_k = top_k
        self.fetch_k_each = fetch_k_each
        self.fusion_method = fusion_method  # rrf | weighted
        self.fusion_alpha = fusion_alpha
        self.rerank_fetch_k = rerank_fetch_k

    def to_dict(self) :
        return {
            "mode": self.mode,
            "top_k": self.top_k,
            "fetch_k_each": self.fetch_k_each,
            "fusion_method": self.fusion_method,
            "fusion_alpha": self.fusion_alpha,
            "rerank_fetch_k": self.rerank_fetch_k,
        }


def _sanitize_answer(answer) :
    answer = answer.strip()
    return answer if answer else ""


def _normalize_ws(text) :
    return _WS_RE.sub(" ", (text or "")).strip()


def _tokenize_query_terms(question) :
    terms = []
    for t in re.findall(r"[a-z0-9]+", (question or "").lower()):
        if len(t) < 4:
            continue
        if t in _STOPWORDS:
            continue
        terms.append(t)
    return terms


def _score_shape_adjustment(question, chunk) :
    if chunk is None:
        return 0.0
    q_terms = _tokenize_query_terms(question)
    if not q_terms:
        return 0.0

    title = (chunk.title or "").lower()
    url = (chunk.source_url or "").lower()
    path = (chunk.source_path or "").lower()
    text = (chunk.text or "").lower()

    title_hits = 0
    url_hits = 0
    text_hits = 0
    for t in q_terms:
        if t in title:
            title_hits += 1
        if t in url or t in path:
            url_hits += 1
        if t in text[:1200]:
            text_hits += 1

    # Lightweight shaping: emphasize title/url matches, small boost for chunk text overlap.
    boost = min(0.24, 0.06 * title_hits) + min(0.12, 0.03 * url_hits) + min(0.12, 0.015 * text_hits)

    penalty = 0.0
    title_url = f"{title} {url} {path}"
    for hint in _LIST_NOISE_HINTS:
        if hint in title_url:
            penalty = max(penalty, 0.08)
    return boost - penalty


def _apply_score_shaping(question, retrieved) :
    if not retrieved:
        return []
    shaped = []
    for r in retrieved:
        adj = _score_shape_adjustment(question, r.chunk)
        parts = dict(r.component_scores) if r.component_scores else {}
        parts["pre_shape_score"] = float(r.score)
        parts["shape_adj"] = float(adj)
        shaped.append(
            RetrievedChunk(
                chunk_id=r.chunk_id,
                score=float(r.score) + float(adj),
                rank=0,
                source=r.source,
                chunk=r.chunk,
                component_scores=parts,
            )
        )
    shaped.sort(key=lambda x: x.score, reverse=True)
    for i, r in enumerate(shaped, start=1):
        r.rank = i
    return shaped


def _simple_sentence_split(text) :
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _question_terms(question) :
    return _tokenize_query_terms(question)


def _best_support_sentence(question, contexts) :
    terms = _question_terms(question)
    best = ""
    best_score = -1
    for c in contexts[:3]:
        for s in _simple_sentence_split(c.text):
            low = s.lower()
            overlap = 0
            for t in terms:
                if t in low:
                    overlap += 1
            score = overlap * 10 + min(4, len(s.split()) // 8)
            if score > best_score:
                best_score = score
                best = s
    if best:
        return best
    if contexts:
        sents = _simple_sentence_split(contexts[0].text)
        if sents:
            return sents[0]
    return ""


def _looks_year_question(question) :
    q = (question or "").lower()
    return "what year" in q or (q.startswith("when ") and "year" in q)


def _looks_when_question(question) :
    return (question or "").lower().startswith("when ")


def _looks_person_question(question) :
    q = (question or "").lower().strip()
    if q.startswith("who "):
        return True
    person_like = [
        "which notable artist",
        "which artist",
        "which alumnus",
        "which alumnus",
        "which person",
        "who is",
        "who was",
    ]
    for p in person_like:
        if p in q:
            return True
    return False


def _extract_name_from_sentence(sent) :
    names = [m.group(1).strip() for m in _NAME_RE.finditer(sent or "")]
    if not names:
        return ""
    filtered = []
    for n in names:
        parts = n.split()
        if not parts:
            continue
        if parts[0] in _MONTHS:
            continue
        if parts[-1] in _ORG_TAIL_WORDS:
            continue
        low = n.lower()
        if "carnegie mellon university" in low:
            continue
        filtered.append(n)
    if not filtered:
        return ""
    filtered.sort(key=lambda x: len(x.split()), reverse=True)
    return filtered[0]


def _extractive_first_answer(question, contexts) :
    if not contexts:
        return ""
    sent = _best_support_sentence(question, contexts)
    if not sent:
        return ""

    if _looks_year_question(question):
        m = _YEAR_RE.search(sent)
        if m:
            return m.group(1)
        for c in contexts[:3]:
            m = _YEAR_RE.search(c.text or "")
            if m:
                return m.group(1)
        return ""

    if _looks_when_question(question):
        m = _DATE_RE.search(sent)
        if m:
            return _normalize_ws(m.group(0))
        m = _YEAR_RE.search(sent)
        if m:
            return m.group(1)
        return ""

    if _looks_person_question(question):
        n = _extract_name_from_sentence(sent)
        if n:
            return n
        return ""

    return ""


class RAGPipeline:
    def __init__(self, retriever, reader, retrieval_cfg, reranker= None):
        self.retriever = retriever
        self.reader = reader
        self.cfg = retrieval_cfg
        self.reranker = reranker

    def retrieve(self, question) :
        mode = self.cfg.mode
        if mode == "closedbook":
            return []
        if not self.retriever:
            raise ValueError("Retriever not available.")
        target_top_k = self.cfg.top_k
        if self.reranker and self.cfg.rerank_fetch_k:
            target_top_k = max(int(self.cfg.top_k), int(self.cfg.rerank_fetch_k))
        if mode == "sparse":
            return self.retriever.retrieve_sparse(question, top_k=target_top_k)
        if mode == "dense":
            return self.retriever.retrieve_dense(question, top_k=target_top_k)
        if mode == "hybrid":
            return self.retriever.retrieve_hybrid(
                question,
                top_k=target_top_k,
                fetch_k_each=self.cfg.fetch_k_each,
                method=self.cfg.fusion_method,
                alpha=self.cfg.fusion_alpha,
            )
        raise ValueError(f"Unknown retrieval mode: {mode}")

    def answer_query(self, qid, question) :
        retrieved = self.retrieve(question)
        if retrieved:
            retrieved = _apply_score_shaping(question, retrieved)
        if self.reranker and retrieved:
            retrieved = self.reranker.rerank(question, retrieved, top_k=self.cfg.top_k)
        contexts = [r.chunk for r in retrieved if r.chunk is not None]
        extractive = _extractive_first_answer(question, contexts)
        if extractive:
            answer = _sanitize_answer(postprocess_answer(extractive, question=question))
        else:
            answer = _sanitize_answer(self.reader.answer(question, contexts))
        trace = []
        for r in retrieved:
            ch = r.chunk
            trace.append(
                {
                    "chunk_id": r.chunk_id,
                    "rank": r.rank,
                    "score": r.score,
                    "source": r.source,
                    "component_scores": r.component_scores,
                    "title": ch.title if ch else None,
                    "source_url": ch.source_url if ch else None,
                    "source_path": ch.source_path if ch else None,
                    "text_preview": (ch.text[:300] if ch else None),
                }
            )
        return {
            "id": qid,
            "question": question,
            "answer": answer,
            "retrieval": trace,
            "config": self.cfg.to_dict(),
        }
