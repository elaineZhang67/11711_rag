from __future__ import annotations

import re

from rag_hw2.types import Chunk, RetrievedChunk

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
        hyde= False,
        hyde_max_new_tokens= 64,
        hyde_weight= 1.0,
        retrieval_augment= False,
        retrieval_augment_max= 2,
        rerank_fetch_k= None,
        query_routing= False,
        factoid_top_k= None,
        factoid_fetch_k_each= None,
        factoid_rerank_fetch_k= None,
        factoid_context_mode= None,
        explanatory_top_k= None,
        explanatory_fetch_k_each= None,
        explanatory_rerank_fetch_k= None,
        explanatory_context_mode= None,
        temporal_top_k= None,
        temporal_fetch_k_each= None,
        temporal_rerank_fetch_k= None,
        temporal_context_mode= None,
        confidence_fallback= False,
        fallback_min_top1= -1.0,
        fallback_min_gap= 0.05,
        fallback_top_k= None,
        fallback_fetch_k_each= None,
        fallback_rerank_fetch_k= None,
        fallback_context_mode= None,
        diversify_docs= False,
        doc_cap= 2,
        context_mode= "child",
        parent_window= 1,
        parent_min_hits= 2,
        parent_max_contexts= 3,
        parent_max_chars= 5000,
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
        self.hyde = bool(hyde)
        self.hyde_max_new_tokens = int(hyde_max_new_tokens)
        self.hyde_weight = float(hyde_weight)
        self.retrieval_augment = bool(retrieval_augment)
        self.retrieval_augment_max = int(retrieval_augment_max)
        self.rerank_fetch_k = rerank_fetch_k
        self.query_routing = bool(query_routing)
        self.factoid_top_k = factoid_top_k
        self.factoid_fetch_k_each = factoid_fetch_k_each
        self.factoid_rerank_fetch_k = factoid_rerank_fetch_k
        self.factoid_context_mode = factoid_context_mode
        self.explanatory_top_k = explanatory_top_k
        self.explanatory_fetch_k_each = explanatory_fetch_k_each
        self.explanatory_rerank_fetch_k = explanatory_rerank_fetch_k
        self.explanatory_context_mode = explanatory_context_mode
        self.temporal_top_k = temporal_top_k
        self.temporal_fetch_k_each = temporal_fetch_k_each
        self.temporal_rerank_fetch_k = temporal_rerank_fetch_k
        self.temporal_context_mode = temporal_context_mode
        self.confidence_fallback = bool(confidence_fallback)
        self.fallback_min_top1 = float(fallback_min_top1)
        self.fallback_min_gap = float(fallback_min_gap)
        self.fallback_top_k = fallback_top_k
        self.fallback_fetch_k_each = fallback_fetch_k_each
        self.fallback_rerank_fetch_k = fallback_rerank_fetch_k
        self.fallback_context_mode = fallback_context_mode
        self.diversify_docs = bool(diversify_docs)
        self.doc_cap = int(doc_cap)
        self.context_mode = context_mode  # child | parent_merge
        self.parent_window = int(parent_window)
        self.parent_min_hits = int(parent_min_hits)
        self.parent_max_contexts = int(parent_max_contexts)
        self.parent_max_chars = int(parent_max_chars)

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
            "hyde": self.hyde,
            "hyde_max_new_tokens": self.hyde_max_new_tokens,
            "hyde_weight": self.hyde_weight,
            "retrieval_augment": self.retrieval_augment,
            "retrieval_augment_max": self.retrieval_augment_max,
            "rerank_fetch_k": self.rerank_fetch_k,
            "query_routing": self.query_routing,
            "factoid_top_k": self.factoid_top_k,
            "factoid_fetch_k_each": self.factoid_fetch_k_each,
            "factoid_rerank_fetch_k": self.factoid_rerank_fetch_k,
            "factoid_context_mode": self.factoid_context_mode,
            "explanatory_top_k": self.explanatory_top_k,
            "explanatory_fetch_k_each": self.explanatory_fetch_k_each,
            "explanatory_rerank_fetch_k": self.explanatory_rerank_fetch_k,
            "explanatory_context_mode": self.explanatory_context_mode,
            "temporal_top_k": self.temporal_top_k,
            "temporal_fetch_k_each": self.temporal_fetch_k_each,
            "temporal_rerank_fetch_k": self.temporal_rerank_fetch_k,
            "temporal_context_mode": self.temporal_context_mode,
            "confidence_fallback": self.confidence_fallback,
            "fallback_min_top1": self.fallback_min_top1,
            "fallback_min_gap": self.fallback_min_gap,
            "fallback_top_k": self.fallback_top_k,
            "fallback_fetch_k_each": self.fallback_fetch_k_each,
            "fallback_rerank_fetch_k": self.fallback_rerank_fetch_k,
            "fallback_context_mode": self.fallback_context_mode,
            "diversify_docs": self.diversify_docs,
            "doc_cap": self.doc_cap,
            "context_mode": self.context_mode,
            "parent_window": self.parent_window,
            "parent_min_hits": self.parent_min_hits,
            "parent_max_contexts": self.parent_max_contexts,
            "parent_max_chars": self.parent_max_chars,
        }


def _sanitize_answer(answer) :
    answer = answer.strip()
    return answer if answer else ""


def _normalize_ws(text) :
    return _WS_RE.sub(" ", (text or "")).strip()


def _looks_explanatory_question(question) :
    q = _normalize_ws(question).lower()
    if not q:
        return False
    if q.startswith("how many ") or q.startswith("how much "):
        return False
    if q.startswith("why ") or q.startswith("how "):
        return True
    markers = [
        "what happened",
        "describe",
        "explain",
        "significance",
        "importance",
        "impact",
        "role",
        "history of",
        "how did",
        "how does",
    ]
    return any(m in q for m in markers)


def _looks_temporal_question(question) :
    q = _normalize_ws(question).lower()
    if not q:
        return False
    temporal_markers = [
        "when ",
        "what year",
        "what date",
        "what month",
        "what day",
        "schedule",
        "deadline",
        "due",
        "quarter",
        "typically held",
        "usually occur",
    ]
    return any(m in q for m in temporal_markers)


def _classify_question_type(question) :
    if _looks_temporal_question(question):
        return "temporal"
    if _looks_explanatory_question(question):
        return "explanatory"
    return "factoid"


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


def _build_retrieval_aug_queries(question, max_n= 2) :
    q = _normalize_ws(question)
    if not q:
        return []

    cands = []

    # Query form without leading wh-prefix.
    stem = re.sub(
        r"^(what|when|where|who|which|how many|how much)\s+(is|was|are|were|did|does|do)?\s*",
        "",
        q,
        flags=re.IGNORECASE,
    )
    stem = stem.strip(" ?")
    if stem:
        cands.append(stem)

    # Entity-focused phrase extraction (capitalized phrases).
    for m in re.finditer(r"\b(?:[A-Z][A-Za-z0-9&.\-']*(?:\s+[A-Z][A-Za-z0-9&.\-']*){0,6})\b", q):
        phrase = m.group(0).strip(" ?")
        if not phrase:
            continue
        if phrase.lower() in {"what", "when", "where", "who", "which", "how"}:
            continue
        cands.append(phrase)

    # Keyword-focused query for sparse retrieval.
    terms = _tokenize_query_terms(q)
    if len(terms) >= 2:
        cands.append(" ".join(terms[:10]))

    out = []
    seen = {q.lower()}
    for c in cands:
        key = c.lower().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(c)
        if len(out) >= max(0, int(max_n)):
            break
    return out


def _annotate_variant_results(results, variant_name) :
    out = []
    for r in results:
        parts = dict(r.component_scores) if r.component_scores else {}
        parts["query_variant"] = variant_name
        out.append(
            RetrievedChunk(
                chunk_id=r.chunk_id,
                score=float(r.score),
                rank=r.rank,
                source=r.source,
                chunk=r.chunk,
                component_scores=parts,
            )
        )
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


def _doc_group_key(r) :
    ch = r.chunk
    if ch is not None:
        doc_id = getattr(ch, "doc_id", None)
        if doc_id:
            return f"doc:{doc_id}"
        source_path = getattr(ch, "source_path", None)
        if source_path:
            return f"path:{source_path}"
        source_url = getattr(ch, "source_url", None)
        if source_url:
            return f"url:{source_url}"
    return f"chunk:{r.chunk_id}"


def _apply_doc_diversification(ranked, top_k, doc_cap= 2) :
    if not ranked:
        return []
    k = max(1, int(top_k))
    cap = max(1, int(doc_cap))

    out = []
    used = set()
    per_doc = {}

    for r in ranked:
        if len(out) >= k:
            break
        g = _doc_group_key(r)
        cnt = per_doc.get(g, 0)
        if cnt >= cap:
            continue
        out.append(r)
        used.add(r.chunk_id)
        per_doc[g] = cnt + 1

    # Backfill if cap is too restrictive for the available candidates.
    if len(out) < k:
        for r in ranked:
            if len(out) >= k:
                break
            if r.chunk_id in used:
                continue
            out.append(r)
            used.add(r.chunk_id)

    for i, r in enumerate(out, start=1):
        r.rank = i
        parts = dict(r.component_scores) if r.component_scores else {}
        parts["doc_diversified"] = 1.0
        parts["doc_group"] = _doc_group_key(r)
        parts["doc_cap"] = float(cap)
        r.component_scores = parts
    return out


class RAGPipeline:
    def __init__(self, retriever, reader, retrieval_cfg, reranker= None):
        self.retriever = retriever
        self.reader = reader
        self.cfg = retrieval_cfg
        self.reranker = reranker
        self._doc_chunks = {}
        chunk_store = getattr(retriever, "chunk_store", None) if retriever is not None else None
        if chunk_store is not None:
            for ch in chunk_store.chunks:
                self._doc_chunks.setdefault(ch.doc_id, []).append(ch)
            for doc_id in self._doc_chunks:
                self._doc_chunks[doc_id].sort(key=lambda x: int(x.chunk_index) if x.chunk_index is not None else 0)

    def _base_runtime_params(self) :
        return {
            "mode": self.cfg.mode,
            "top_k": int(self.cfg.top_k),
            "fetch_k_each": int(self.cfg.fetch_k_each),
            "fusion_method": self.cfg.fusion_method,
            "fusion_alpha": float(self.cfg.fusion_alpha),
            "rrf_k": int(self.cfg.rrf_k),
            "dense_weight": float(self.cfg.dense_weight),
            "sparse_weight": float(self.cfg.sparse_weight),
            "multi_query": bool(self.cfg.multi_query),
            "multi_query_max": int(self.cfg.multi_query_max),
            "hyde": bool(self.cfg.hyde),
            "hyde_max_new_tokens": int(self.cfg.hyde_max_new_tokens),
            "hyde_weight": float(self.cfg.hyde_weight),
            "retrieval_augment": bool(self.cfg.retrieval_augment),
            "retrieval_augment_max": int(self.cfg.retrieval_augment_max),
            "rerank_fetch_k": (None if self.cfg.rerank_fetch_k is None else int(self.cfg.rerank_fetch_k)),
            "context_mode": self.cfg.context_mode,
        }

    def _apply_runtime_overrides(self, params, top_k= None, fetch_k_each= None, rerank_fetch_k= None, context_mode= None) :
        if top_k is not None:
            params["top_k"] = int(top_k)
        if fetch_k_each is not None:
            params["fetch_k_each"] = int(fetch_k_each)
        if rerank_fetch_k is not None:
            params["rerank_fetch_k"] = int(rerank_fetch_k)
        if context_mode in {"child", "parent_merge"}:
            params["context_mode"] = context_mode
        return params

    def _build_params_for_question(self, question) :
        params = self._base_runtime_params()
        route_type = "default"
        if not self.cfg.query_routing:
            return params, route_type

        route_type = _classify_question_type(question)
        if route_type == "factoid":
            self._apply_runtime_overrides(
                params,
                top_k=self.cfg.factoid_top_k,
                fetch_k_each=self.cfg.factoid_fetch_k_each,
                rerank_fetch_k=self.cfg.factoid_rerank_fetch_k,
                context_mode=self.cfg.factoid_context_mode,
            )
        elif route_type == "explanatory":
            self._apply_runtime_overrides(
                params,
                top_k=self.cfg.explanatory_top_k,
                fetch_k_each=self.cfg.explanatory_fetch_k_each,
                rerank_fetch_k=self.cfg.explanatory_rerank_fetch_k,
                context_mode=self.cfg.explanatory_context_mode,
            )
        elif route_type == "temporal":
            self._apply_runtime_overrides(
                params,
                top_k=self.cfg.temporal_top_k,
                fetch_k_each=self.cfg.temporal_fetch_k_each,
                rerank_fetch_k=self.cfg.temporal_rerank_fetch_k,
                context_mode=self.cfg.temporal_context_mode,
            )
        return params, route_type

    def _build_fallback_params(self, params) :
        out = dict(params)
        self._apply_runtime_overrides(
            out,
            top_k=self.cfg.fallback_top_k,
            fetch_k_each=self.cfg.fallback_fetch_k_each,
            rerank_fetch_k=self.cfg.fallback_rerank_fetch_k,
            context_mode=self.cfg.fallback_context_mode,
        )
        return out

    def _retrieve_single(self, question, params, target_top_k) :
        mode = params["mode"]
        if mode == "sparse":
            return self.retriever.retrieve_sparse(question, top_k=target_top_k)
        if mode == "dense":
            return self.retriever.retrieve_dense(question, top_k=target_top_k)
        if mode == "hybrid":
            return self.retriever.retrieve_hybrid(
                question,
                top_k=target_top_k,
                fetch_k_each=int(params["fetch_k_each"]),
                method=params["fusion_method"],
                alpha=float(params["fusion_alpha"]),
                rrf_k=int(params["rrf_k"]),
                dense_weight=float(params["dense_weight"]),
                sparse_weight=float(params["sparse_weight"]),
            )
        raise ValueError(f"Unknown retrieval mode: {mode}")

    def _retrieve_hyde_dense(self, question, target_top_k, params) :
        if not bool(params.get("hyde", False)):
            return [], ""
        if params.get("mode") not in {"dense", "hybrid"}:
            return [], ""
        if not self.retriever or not getattr(self.retriever, "dense", None):
            return [], ""
        generate_hypothesis = getattr(self.reader, "generate_hypothesis", None)
        if not callable(generate_hypothesis):
            return [], ""
        hypo = _normalize_ws(
            generate_hypothesis(
                question,
                max_new_tokens=int(params.get("hyde_max_new_tokens", 64)),
            )
        )
        if not hypo:
            return [], ""
        hyde_results = self.retriever.retrieve_dense(hypo, top_k=target_top_k)
        hyde_weight = min(1.0, max(0.0, float(params.get("hyde_weight", 1.0))))
        out = []
        for r in hyde_results:
            parts = dict(r.component_scores) if r.component_scores else {}
            parts["hyde"] = 1.0
            parts["hyde_len"] = float(len(hypo.split()))
            parts["hyde_weight"] = float(hyde_weight)
            parts["hyde_raw_score"] = float(r.score)
            out.append(
                RetrievedChunk(
                    chunk_id=r.chunk_id,
                    score=float(r.score) * hyde_weight,
                    rank=r.rank,
                    source=r.source,
                    chunk=r.chunk,
                    component_scores=parts,
                )
            )
        return out, hypo

    def _retrieve_with_params(self, question, params) :
        mode = params["mode"]
        if mode == "closedbook":
            return []
        if not self.retriever:
            raise ValueError("Retriever not available.")
        target_top_k = int(params["top_k"])
        rerank_fetch_k = params.get("rerank_fetch_k")
        if self.reranker and rerank_fetch_k is not None:
            target_top_k = max(target_top_k, int(rerank_fetch_k))

        base_results = self._retrieve_single(question, params, target_top_k)
        base_results = _annotate_variant_results(base_results, "original_query")
        result_lists = [base_results]

        hyde_results, _ = self._retrieve_hyde_dense(question, target_top_k, params)
        if hyde_results:
            result_lists.append(hyde_results)

        extra_queries = []
        if bool(params.get("retrieval_augment", False)):
            extra_queries.extend(
                _build_retrieval_aug_queries(question, max_n=int(params.get("retrieval_augment_max", 0)))
            )
        if bool(params.get("multi_query", False)):
            extra_queries.extend(_build_multi_queries(question, max_n=int(params.get("multi_query_max", 0))))

        deduped_queries = []
        seen_queries = set()
        for q in extra_queries:
            key = _normalize_ws(q).lower()
            if not key or key in seen_queries:
                continue
            seen_queries.add(key)
            deduped_queries.append(q)

        for q in deduped_queries:
            tagged = _annotate_variant_results(
                self._retrieve_single(q, params, target_top_k),
                f"augmented_query:{q}",
            )
            result_lists.append(tagged)

        if len(result_lists) == 1:
            return base_results

        merged = _merge_multi_query_results(result_lists, keep_n=target_top_k)
        preserve_n = min(3, max(1, int(params["top_k"])))
        return _preserve_primary_results(base_results, merged, keep_n=target_top_k, preserve_n=preserve_n)

    def retrieve(self, question) :
        params, _ = self._build_params_for_question(question)
        return self._retrieve_with_params(question, params)

    def _retrieve_and_rerank(self, question, params) :
        retrieved = self._retrieve_with_params(question, params)
        if retrieved:
            retrieved = _apply_score_shaping(question, retrieved)
        if self.reranker and retrieved:
            retrieved = self.reranker.rerank(question, retrieved, top_k=int(params["top_k"]))
        if retrieved and self.cfg.diversify_docs:
            retrieved = _apply_doc_diversification(
                retrieved,
                top_k=int(params["top_k"]),
                doc_cap=self.cfg.doc_cap,
            )
        return retrieved

    def _should_trigger_fallback(self, retrieved) :
        if not self.cfg.confidence_fallback:
            return False
        if not retrieved:
            return True
        top1 = float(retrieved[0].score)
        if top1 < float(self.cfg.fallback_min_top1):
            return True
        if len(retrieved) >= 2:
            gap = float(retrieved[0].score) - float(retrieved[1].score)
            if gap < float(self.cfg.fallback_min_gap):
                return True
        return False

    def _build_parent_contexts(self, retrieved, context_mode= None) :
        child_contexts = [r.chunk for r in retrieved if r.chunk is not None]
        mode = context_mode or self.cfg.context_mode
        if mode != "parent_merge":
            return child_contexts
        if not child_contexts or not self._doc_chunks:
            return child_contexts

        per_doc = {}
        for r in retrieved:
            ch = r.chunk
            if ch is None or ch.doc_id is None:
                continue
            try:
                idx = int(ch.chunk_index)
            except Exception:
                continue
            rec = per_doc.get(ch.doc_id)
            if rec is None:
                rec = {"best_score": float(r.score), "hits": 0, "seed_indices": set()}
                per_doc[ch.doc_id] = rec
            rec["hits"] += 1
            rec["seed_indices"].add(idx)
            if float(r.score) > rec["best_score"]:
                rec["best_score"] = float(r.score)

        min_hits = max(1, int(self.cfg.parent_min_hits))
        window = max(0, int(self.cfg.parent_window))
        max_chars = max(300, int(self.cfg.parent_max_chars))
        merged = []

        for doc_id, rec in per_doc.items():
            if rec["hits"] < min_hits:
                continue
            doc_chunks = self._doc_chunks.get(doc_id, [])
            if not doc_chunks:
                continue

            wanted = set()
            for idx in rec["seed_indices"]:
                for j in range(idx - window, idx + window + 1):
                    wanted.add(j)

            selected = []
            for ch in doc_chunks:
                try:
                    ci = int(ch.chunk_index)
                except Exception:
                    continue
                if ci in wanted:
                    selected.append(ch)
            if not selected:
                continue

            selected.sort(key=lambda x: int(x.chunk_index) if x.chunk_index is not None else 0)
            text = _normalize_ws(" ".join((c.text or "").strip() for c in selected))
            if len(text) > max_chars:
                text = text[:max_chars].rsplit(" ", 1)[0]

            first = selected[0]
            last = selected[-1]
            try:
                start_idx = int(first.chunk_index)
            except Exception:
                start_idx = 0
            try:
                end_idx = int(last.chunk_index)
            except Exception:
                end_idx = start_idx
            parent_chunk = Chunk(
                chunk_id=f"parent:{doc_id}:{start_idx}-{end_idx}",
                doc_id=doc_id,
                text=text,
                title=first.title,
                source_path=first.source_path,
                source_url=first.source_url,
                start_char=first.start_char,
                end_char=last.end_char,
                chunk_index=start_idx,
                metadata=first.metadata,
            )
            merged.append((rec["best_score"], parent_chunk))

        if not merged:
            return child_contexts

        merged.sort(key=lambda x: x[0], reverse=True)
        max_ctx = max(1, int(self.cfg.parent_max_contexts))
        contexts = [x[1] for x in merged[:max_ctx]]
        return contexts if contexts else child_contexts

    def answer_query(self, qid, question) :
        params, route_type = self._build_params_for_question(question)
        retrieved = self._retrieve_and_rerank(question, params)

        fallback_used = False
        if self._should_trigger_fallback(retrieved):
            fallback_params = self._build_fallback_params(params)
            fallback_retrieved = self._retrieve_and_rerank(question, fallback_params)
            if fallback_retrieved:
                retrieved = fallback_retrieved
                params = fallback_params
                fallback_used = True

        contexts = self._build_parent_contexts(retrieved, context_mode=params.get("context_mode"))
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
            "route_type": route_type,
            "fallback_used": fallback_used,
            "runtime_retrieval": params,
            "config": self.cfg.to_dict(),
        }
