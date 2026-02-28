from __future__ import annotations

from sentence_transformers import CrossEncoder

from rag_hw2.types import RetrievedChunk


def _build_reranker_text(chunk):
    if chunk is None:
        return ""
    parts = []
    if getattr(chunk, "title", None):
        parts.append(str(chunk.title).strip())
    text = getattr(chunk, "text", "") or ""
    if text:
        parts.append(text.strip())
    return "\n".join([p for p in parts if p])


class CrossEncoderReranker:
    def __init__(
        self,
        model_name= "cross-encoder/ms-marco-MiniLM-L-12-v2",
        device= None,
        batch_size= 16,
        max_length= 512,
    ) :
        self.model_name = model_name
        self.device = device
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        kwargs = {
            "model_name": self.model_name,
            "max_length": self.max_length,
        }
        if self.device is not None:
            kwargs["device"] = self.device
        self.model = CrossEncoder(**kwargs)

    def rerank(self, query, retrieved, top_k= None) :
        if not retrieved:
            return []
        pairs = []
        kept = []
        for r in retrieved:
            text = _build_reranker_text(r.chunk)
            if not text:
                continue
            pairs.append((query, text))
            kept.append(r)
        if not kept:
            return retrieved[: top_k or len(retrieved)]

        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        combined = []
        for r, score in zip(kept, scores):
            combined.append((r, float(score)))
        combined.sort(key=lambda x: x[1], reverse=True)

        limit = len(combined) if top_k is None else max(1, int(top_k))
        out = []
        for rank, (r, rerank_score) in enumerate(combined[:limit], start=1):
            parts = dict(r.component_scores) if r.component_scores else {}
            parts["retrieval_score"] = float(r.score)
            parts["retrieval_rank"] = int(r.rank)
            parts["reranker_score"] = float(rerank_score)
            out.append(
                RetrievedChunk(
                    chunk_id=r.chunk_id,
                    score=float(rerank_score),
                    rank=rank,
                    source=r.source,
                    chunk=r.chunk,
                    component_scores=parts,
                )
            )
        return out

