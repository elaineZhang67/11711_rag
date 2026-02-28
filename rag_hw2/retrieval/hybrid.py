from __future__ import annotations

from pathlib import Path

from rag_hw2.types import RetrievedChunk
from rag_hw2.retrieval.chunk_store import ChunkStore, load_chunk_store_from_jsonl
from rag_hw2.retrieval.dense_faiss import DenseFAISSIndex, load_dense_faiss_index
from rag_hw2.retrieval.fusion import reciprocal_rank_fusion, weighted_score_fusion
from rag_hw2.retrieval.sparse_bm25 import SparseBM25Index, load_sparse_bm25_index


class HybridRetriever:
    def __init__(
        self,
        chunk_store,
        sparse_index= None,
        dense_index= None,
    ):
        self.chunk_store = chunk_store
        self.sparse = sparse_index
        self.dense = dense_index

    def retrieve_sparse(self, query, top_k= 10) :
        if not self.sparse:
            raise ValueError("Sparse index not loaded.")
        return self.sparse.retrieve(query, top_k=top_k)

    def retrieve_dense(self, query, top_k= 10) :
        if not self.dense:
            raise ValueError("Dense index not loaded.")
        return self.dense.retrieve(query, top_k=top_k)

    def retrieve_hybrid(
        self,
        query,
        top_k= 10,
        fetch_k_each= 50,
        method= "rrf",
        alpha= 0.5,
    ) :
        if not self.sparse or not self.dense:
            raise ValueError("Both sparse and dense indices are required for hybrid retrieval.")
        sparse = self.sparse.retrieve(query, top_k=fetch_k_each)
        dense = self.dense.retrieve(query, top_k=fetch_k_each)
        sparse_pairs = [(r.chunk_id, r.score) for r in sparse]
        dense_pairs = [(r.chunk_id, r.score) for r in dense]
        if method == "weighted":
            fused = weighted_score_fusion(dense=dense_pairs, sparse=sparse_pairs, alpha=alpha, top_n=top_k)
        else:
            fused = reciprocal_rank_fusion({"dense": dense_pairs, "sparse": sparse_pairs}, top_n=top_k)
        out = []
        for rank, (cid, score, parts) in enumerate(fused, start=1):
            out.append(
                RetrievedChunk(
                    chunk_id=cid,
                    score=float(score),
                    rank=rank,
                    source="hybrid",
                    chunk=self.chunk_store.by_id.get(cid),
                    component_scores=parts,
                )
            )
        return out


def build_hybrid_retriever_from_dirs(
    chunks_jsonl,
    sparse_dir= None,
    dense_dir= None,
) :
    chunk_store = load_chunk_store_from_jsonl(chunks_jsonl)
    sparse = load_sparse_bm25_index(sparse_dir, chunk_store=chunk_store) if sparse_dir else None
    dense = load_dense_faiss_index(dense_dir, chunk_store=chunk_store) if dense_dir else None
    chunk_ids = set(chunk_store.by_id.keys())
    if sparse:
        sparse_missing = [cid for cid in sparse.artifacts.chunk_ids[:200] if cid not in chunk_ids]
        if sparse_missing:
            raise ValueError(
                "Sparse index and chunks file do not match. "
                f"Example missing chunk id: {sparse_missing[0]!r}. "
                "Rebuild sparse index with the same chunks file you pass to answer_queries.py."
            )
    if dense:
        dense_missing = [cid for cid in dense.artifacts.chunk_ids[:200] if cid not in chunk_ids]
        if dense_missing:
            raise ValueError(
                "Dense index and chunks file do not match. "
                f"Example missing chunk id: {dense_missing[0]!r}. "
                "Rebuild dense index with the same chunks file you pass to answer_queries.py."
            )
    return HybridRetriever(chunk_store=chunk_store, sparse_index=sparse, dense_index=dense)
