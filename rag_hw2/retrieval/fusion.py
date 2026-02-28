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

