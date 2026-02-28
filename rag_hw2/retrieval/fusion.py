from __future__ import annotations

from collections import defaultdict


def reciprocal_rank_fusion(
    rankings,
    k= 60,
    top_n= 10,
) :
    scores = defaultdict(float)
    parts = defaultdict(dict)
    for source_name, ranked in rankings.items():
        for rank_idx, (doc_id, score) in enumerate(ranked, start=1):
            rrf = 1.0 / (k + rank_idx)
            scores[doc_id] += rrf
            parts[doc_id][f"{source_name}_raw"] = float(score)
            parts[doc_id][f"{source_name}_rrf"] = rrf
    out = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [(doc_id, score, parts.get(doc_id, {})) for doc_id, score in out]


def weighted_score_fusion(
    dense,
    sparse,
    alpha= 0.5,
    top_n= 10,
) :
    def _norm(values, key) :
        if not values:
            return {}
        scores = [v for _, v in values]
        lo, hi = min(scores), max(scores)
        denom = (hi - lo) if hi > lo else 1.0
        out = {}
        for doc_id, s in values:
            out[doc_id] = {f"{key}_raw": float(s), f"{key}_norm": float((s - lo) / denom)}
        return out

    d = _norm(dense, "dense")
    s = _norm(sparse, "sparse")
    ids = set(d) | set(s)
    fused = []
    for chunk_id in ids:
        dnorm = d.get(chunk_id, {}).get("dense_norm", 0.0)
        snorm = s.get(chunk_id, {}).get("sparse_norm", 0.0)
        score = alpha * dnorm + (1 - alpha) * snorm
        detail = {}
        detail.update(d.get(chunk_id, {}))
        detail.update(s.get(chunk_id, {}))
        fused.append((chunk_id, float(score), detail))
    fused.sort(key=lambda x: x[1], reverse=True)
    return fused[:top_n]


def weighted_rrf_fusion(
    dense,
    sparse,
    dense_weight= 0.5,
    sparse_weight= 0.5,
    k= 60,
    top_n= 10,
) :
    scores = defaultdict(float)
    parts = defaultdict(dict)

    for rank_idx, (doc_id, score) in enumerate(dense, start=1):
        comp = float(dense_weight) * (1.0 / (k + rank_idx))
        scores[doc_id] += comp
        parts[doc_id]["dense_raw"] = float(score)
        parts[doc_id]["dense_rrf"] = float(1.0 / (k + rank_idx))
        parts[doc_id]["dense_weight"] = float(dense_weight)

    for rank_idx, (doc_id, score) in enumerate(sparse, start=1):
        comp = float(sparse_weight) * (1.0 / (k + rank_idx))
        scores[doc_id] += comp
        parts[doc_id]["sparse_raw"] = float(score)
        parts[doc_id]["sparse_rrf"] = float(1.0 / (k + rank_idx))
        parts[doc_id]["sparse_weight"] = float(sparse_weight)

    out = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [(doc_id, score, parts.get(doc_id, {})) for doc_id, score in out]


def combmnz_fusion(
    dense,
    sparse,
    alpha= 0.5,
    top_n= 10,
) :
    def _norm(values, key) :
        if not values:
            return {}
        vals = [v for _, v in values]
        lo, hi = min(vals), max(vals)
        denom = (hi - lo) if hi > lo else 1.0
        out = {}
        for doc_id, s in values:
            out[doc_id] = {
                f"{key}_raw": float(s),
                f"{key}_norm": float((s - lo) / denom),
            }
        return out

    d = _norm(dense, "dense")
    s = _norm(sparse, "sparse")
    ids = set(d) | set(s)
    fused = []
    for chunk_id in ids:
        dnorm = d.get(chunk_id, {}).get("dense_norm", 0.0)
        snorm = s.get(chunk_id, {}).get("sparse_norm", 0.0)
        combsum = float(alpha) * dnorm + (1.0 - float(alpha)) * snorm
        present = int(chunk_id in d) + int(chunk_id in s)
        score = combsum * present
        detail = {}
        detail.update(d.get(chunk_id, {}))
        detail.update(s.get(chunk_id, {}))
        detail["combmnz_present"] = float(present)
        fused.append((chunk_id, float(score), detail))
    fused.sort(key=lambda x: x[1], reverse=True)
    return fused[:top_n]
