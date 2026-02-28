from __future__ import annotations

import re

from rag_hw2.types import RetrievedChunk

_WS_RE = re.compile(r"\s+")
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
        rrf_k= 60,
        dense_weight= 0.5,
        sparse_weight= 0.5,
        multi_query= False,
        multi_query_max= 2,
        rerank_fetch_k= None,
    ) :
        self.mode = mode  # sparse | dense | hybrid | closedbook
        self.top_k = top_k
        self.fetch_k_each = fetch_k_each
        self.fusion_method = fusion_method  # rrf | weighted | weighted_rrf | combmnz
        self.fusion_alpha = fusion_alpha
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.multi_query = bool(multi_query)
        self.multi_query_max = int(multi_query_max)
        self.rerank_fetch_k = rerank_fetch_k

    def to_dict(self) :
        return {
            "mode": self.mode,
            "top_k": self.top_k,
            "fetch_k_each": self.fetch_k_each,
            "fusion_method": self.fusion_method,
            "fusion_alpha": self.fusion_alpha,
            "rrf_k": self.rrf_k,
            "dense_weight": self.dense_weight,
            "sparse_weight": self.sparse_weight,
            "multi_query": self.multi_query,
            "multi_query_max": self.multi_query_max,
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


def _rewrite_prefix(question, old_prefix, new_prefix) :
    q = _normalize_ws(question)
    if q.lower().startswith(old_prefix):
        return _normalize_ws(new_prefix + q[len(old_prefix) :])
    return ""


def _build_multi_queries(question, max_n= 2) :
    q = _normalize_ws(question)
    if not q:
        return []
    cands = []

    # Conservative rewrites only.
    r = _rewrite_prefix(q, "when was ", "in what year was ")
    if r:
        cands.append(r)
    r = _rewrite_prefix(q, "when did ", "in what year did ")
    if r:
        cands.append(r)
    r = _rewrite_prefix(q, "where is ", "what is the location of ")
    if r:
        cands.append(r)

    # Add one keyword-focused query.
    terms = _tokenize_query_terms(q)
    if len(terms) >= 2:
        cands.append(" ".join(terms[:8]))

    out = []
    seen = {q.lower()}
    for c in cands:
        if not c:
            continue
        key = c.lower().strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
        if len(out) >= max(0, int(max_n)):
            break
    return out


def _merge_multi_query_results(result_lists, keep_n) :
    merged = {}
    for results in result_lists:
        for r in results:
            rec = merged.get(r.chunk_id)
            if rec is None:
                merged[r.chunk_id] = {
                    "best": r,
                    "best_score": float(r.score),
                    "votes": 1,
                }
                continue
            rec["votes"] += 1
            if float(r.score) > rec["best_score"]:
                rec["best"] = r
                rec["best_score"] = float(r.score)

    out = []
    for cid, rec in merged.items():
        base = rec["best"]
        votes = int(rec["votes"])
        best_score = float(rec["best_score"])
        # Keep MQ vote bonus small so repeated generic pages do not dominate.
        score = best_score + 0.002 * min(3, max(0, votes - 1))
        parts = dict(base.component_scores) if base.component_scores else {}
        parts["mq_votes"] = votes
        parts["mq_best_score"] = best_score
        out.append(
            RetrievedChunk(
                chunk_id=cid,
                score=score,
                rank=0,
                source=base.source,
                chunk=base.chunk,
                component_scores=parts,
            )
        )
    out.sort(key=lambda x: x.score, reverse=True)
    out = out[: max(1, int(keep_n))]
    for i, r in enumerate(out, start=1):
        r.rank = i
    return out


def _preserve_primary_results(primary_results, merged_results, keep_n, preserve_n= 2) :
    keep_n = max(1, int(keep_n))
    preserve_n = max(0, min(int(preserve_n), keep_n))
    out = []
    seen = set()
    merged_by_id = {r.chunk_id: r for r in merged_results}

    # Always keep a few top docs from the original query to avoid rewrite drift.
    for p in primary_results[:preserve_n]:
        use = merged_by_id.get(p.chunk_id, p)
        if use.chunk_id in seen:
            continue
        out.append(use)
        seen.add(use.chunk_id)

    for r in merged_results:
        if len(out) >= keep_n:
            break
        if r.chunk_id in seen:
            continue
        out.append(r)
        seen.add(r.chunk_id)

    for p in primary_results:
        if len(out) >= keep_n:
            break
        if p.chunk_id in seen:
            continue
        out.append(p)
        seen.add(p.chunk_id)

    for i, r in enumerate(out, start=1):
        r.rank = i
    return out


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


class RAGPipeline:
    def __init__(self, retriever, reader, retrieval_cfg, reranker= None):
        self.retriever = retriever
        self.reader = reader
        self.cfg = retrieval_cfg
        self.reranker = reranker

    def _retrieve_single(self, question, mode, target_top_k) :
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
                rrf_k=self.cfg.rrf_k,
                dense_weight=self.cfg.dense_weight,
                sparse_weight=self.cfg.sparse_weight,
            )
        raise ValueError(f"Unknown retrieval mode: {mode}")

    def retrieve(self, question) :
        mode = self.cfg.mode
        if mode == "closedbook":
            return []
        if not self.retriever:
            raise ValueError("Retriever not available.")
        target_top_k = self.cfg.top_k
        if self.reranker and self.cfg.rerank_fetch_k:
            target_top_k = max(int(self.cfg.top_k), int(self.cfg.rerank_fetch_k))

        if not self.cfg.multi_query:
            return self._retrieve_single(question, mode, target_top_k)

        base_results = self._retrieve_single(question, mode, target_top_k)
        alt_queries = _build_multi_queries(question, max_n=self.cfg.multi_query_max)
        if not alt_queries:
            return base_results
        result_lists = [base_results]
        for q in alt_queries:
            result_lists.append(self._retrieve_single(q, mode, target_top_k))
        merged = _merge_multi_query_results(result_lists, keep_n=target_top_k)
        preserve_n = min(3, max(1, int(self.cfg.top_k)))
        return _preserve_primary_results(base_results, merged, keep_n=target_top_k, preserve_n=preserve_n)

    def answer_query(self, qid, question) :
        retrieved = self.retrieve(question)
        if retrieved:
            retrieved = _apply_score_shaping(question, retrieved)
        if self.reranker and retrieved:
            retrieved = self.reranker.rerank(question, retrieved, top_k=self.cfg.top_k)
        contexts = [r.chunk for r in retrieved if r.chunk is not None]
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
