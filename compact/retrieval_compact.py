from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from compact.io_utils import load_pickle, read_jsonl, save_pickle
from compact.text_utils import bm25_tokenize
from rag_hw2.types import Chunk, RetrievedChunk


class ChunkStore:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.by_id = {c.chunk_id: c for c in chunks}

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "ChunkStore":
        chunks = [Chunk.from_dict(x) for x in read_jsonl(path)]
        return cls(chunks)

    def ids(self) -> list[str]:
        return [c.chunk_id for c in self.chunks]

    def texts(self) -> list[str]:
        return [c.text for c in self.chunks]


class SparseIndexArtifacts:
    def __init__(
        self,
        chunk_ids: list[str],
        tokenized_texts: list[list[str]],
        model_name: str = "rank_bm25",
    ) -> None:
        self.chunk_ids = list(chunk_ids)
        self.tokenized_texts = [list(toks) for toks in tokenized_texts]
        self.model_name = model_name


class SparseBM25Index:
    def __init__(self, bm25, artifacts: SparseIndexArtifacts, chunk_store: ChunkStore | None = None):
        self.bm25 = bm25
        self.artifacts = artifacts
        self.chunk_store = chunk_store

    @classmethod
    def build(cls, chunk_store: ChunkStore) -> "SparseBM25Index":
        tokenized = [bm25_tokenize(c.text) for c in chunk_store.chunks]
        bm25 = BM25Okapi(tokenized)
        return cls(bm25, SparseIndexArtifacts(chunk_store.ids(), tokenized), chunk_store=chunk_store)

    def save(self, out_dir: str | Path) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        save_pickle(self.bm25, out / "bm25_model.pkl")
        save_pickle(self.artifacts, out / "bm25_artifacts.pkl")

    @classmethod
    def load(cls, out_dir: str | Path, chunk_store: ChunkStore | None = None) -> "SparseBM25Index":
        out = Path(out_dir)
        return cls(
            load_pickle(out / "bm25_model.pkl"),
            load_pickle(out / "bm25_artifacts.pkl"),
            chunk_store=chunk_store,
        )

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievedChunk]:
        q_tokens = bm25_tokenize(query)
        scores = np.asarray(self.bm25.get_scores(q_tokens), dtype=float)
        if scores.size == 0:
            return []
        k = min(top_k, scores.size)
        idx = np.argpartition(-scores, kth=k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        out: list[RetrievedChunk] = []
        for rank, i in enumerate(idx.tolist(), start=1):
            cid = self.artifacts.chunk_ids[i]
            out.append(
                RetrievedChunk(
                    chunk_id=cid,
                    score=float(scores[i]),
                    rank=rank,
                    source="sparse",
                    chunk=self.chunk_store.by_id.get(cid) if self.chunk_store else None,
                    component_scores={"sparse": float(scores[i])},
                )
            )
        return out


class DenseIndexArtifacts:
    def __init__(
        self,
        chunk_ids: list[str],
        embedding_model: str,
        normalize_embeddings: bool = True,
    ) -> None:
        self.chunk_ids = list(chunk_ids)
        self.embedding_model = embedding_model
        self.normalize_embeddings = normalize_embeddings


class DenseFAISSIndex:
    def __init__(self, index, encoder, artifacts: DenseIndexArtifacts, chunk_store: ChunkStore | None = None):
        self.index = index
        self.encoder = encoder
        self.artifacts = artifacts
        self.chunk_store = chunk_store

    @classmethod
    def build(
        cls,
        chunk_store: ChunkStore,
        model_name: str = "BAAI/bge-large-en-v1.5",
        batch_size: int = 64,
        normalize_embeddings: bool = True,
        verbose: bool = False,
    ) -> "DenseFAISSIndex":
        encoder = SentenceTransformer(model_name)
        texts = [c.text for c in chunk_store.chunks]
        if verbose:
            print(f"[dense] embedding {len(texts)} chunks with {model_name}")
        emb = encoder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=verbose,
            convert_to_numpy=True,
            normalize_embeddings=normalize_embeddings,
        )
        emb = np.asarray(emb, dtype=np.float32)
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        return cls(
            index=index,
            encoder=encoder,
            artifacts=DenseIndexArtifacts(
                chunk_ids=chunk_store.ids(),
                embedding_model=model_name,
                normalize_embeddings=normalize_embeddings,
            ),
            chunk_store=chunk_store,
        )

    def save(self, out_dir: str | Path) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(out / "dense.index"))
        save_pickle(self.artifacts, out / "dense_artifacts.pkl")

    @classmethod
    def load(cls, out_dir: str | Path, chunk_store: ChunkStore | None = None) -> "DenseFAISSIndex":
        out = Path(out_dir)
        artifacts: DenseIndexArtifacts = load_pickle(out / "dense_artifacts.pkl")
        index = faiss.read_index(str(out / "dense.index"))
        encoder = SentenceTransformer(artifacts.embedding_model)
        return cls(index=index, encoder=encoder, artifacts=artifacts, chunk_store=chunk_store)

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievedChunk]:
        q = self.encoder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=self.artifacts.normalize_embeddings,
        )
        q = np.asarray(q, dtype=np.float32)
        scores, ids = self.index.search(q, top_k)
        out: list[RetrievedChunk] = []
        for rank, (idx, score) in enumerate(zip(ids[0].tolist(), scores[0].tolist()), start=1):
            if idx < 0:
                continue
            cid = self.artifacts.chunk_ids[idx]
            out.append(
                RetrievedChunk(
                    chunk_id=cid,
                    score=float(score),
                    rank=rank,
                    source="dense",
                    chunk=self.chunk_store.by_id.get(cid) if self.chunk_store else None,
                    component_scores={"dense": float(score)},
                )
            )
        return out


def reciprocal_rank_fusion(
    rankings: dict[str, list[tuple[str, float]]],
    k: int = 60,
    top_n: int = 10,
) -> list[tuple[str, float, dict[str, float]]]:
    scores: dict[str, float] = defaultdict(float)
    parts: dict[str, dict[str, float]] = defaultdict(dict)
    for source_name, ranked in rankings.items():
        for rank_idx, (doc_id, score) in enumerate(ranked, start=1):
            rrf = 1.0 / (k + rank_idx)
            scores[doc_id] += rrf
            parts[doc_id][f"{source_name}_raw"] = float(score)
            parts[doc_id][f"{source_name}_rrf"] = rrf
    out = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [(doc_id, score, parts.get(doc_id, {})) for doc_id, score in out]


def weighted_score_fusion(
    dense: list[tuple[str, float]],
    sparse: list[tuple[str, float]],
    alpha: float = 0.5,
    top_n: int = 10,
) -> list[tuple[str, float, dict[str, float]]]:
    def _norm(values: list[tuple[str, float]], key: str) -> dict[str, dict[str, float]]:
        if not values:
            return {}
        vals = [v for _, v in values]
        lo, hi = min(vals), max(vals)
        denom = (hi - lo) if hi > lo else 1.0
        out: dict[str, dict[str, float]] = {}
        for doc_id, s in values:
            out[doc_id] = {f"{key}_raw": float(s), f"{key}_norm": float((s - lo) / denom)}
        return out

    d = _norm(dense, "dense")
    s = _norm(sparse, "sparse")
    fused: list[tuple[str, float, dict[str, float]]] = []
    for cid in set(d) | set(s):
        score = alpha * d.get(cid, {}).get("dense_norm", 0.0) + (1 - alpha) * s.get(cid, {}).get("sparse_norm", 0.0)
        detail: dict[str, float] = {}
        detail.update(d.get(cid, {}))
        detail.update(s.get(cid, {}))
        fused.append((cid, float(score), detail))
    fused.sort(key=lambda x: x[1], reverse=True)
    return fused[:top_n]


class HybridRetriever:
    def __init__(self, chunk_store: ChunkStore, sparse_index: SparseBM25Index | None = None, dense_index: DenseFAISSIndex | None = None):
        self.chunk_store = chunk_store
        self.sparse = sparse_index
        self.dense = dense_index

    @classmethod
    def from_dirs(
        cls,
        chunks_jsonl: str | Path,
        sparse_dir: str | Path | None = None,
        dense_dir: str | Path | None = None,
    ) -> "HybridRetriever":
        store = ChunkStore.from_jsonl(chunks_jsonl)
        sparse = SparseBM25Index.load(sparse_dir, chunk_store=store) if sparse_dir else None
        dense = DenseFAISSIndex.load(dense_dir, chunk_store=store) if dense_dir else None
        return cls(store, sparse, dense)

    def retrieve_sparse(self, query: str, top_k: int = 10) -> list[RetrievedChunk]:
        if not self.sparse:
            raise ValueError("Sparse index not loaded.")
        return self.sparse.retrieve(query, top_k=top_k)

    def retrieve_dense(self, query: str, top_k: int = 10) -> list[RetrievedChunk]:
        if not self.dense:
            raise ValueError("Dense index not loaded.")
        return self.dense.retrieve(query, top_k=top_k)

    def retrieve_hybrid(
        self,
        query: str,
        top_k: int = 10,
        fetch_k_each: int = 50,
        method: str = "rrf",
        alpha: float = 0.5,
    ) -> list[RetrievedChunk]:
        if not self.sparse or not self.dense:
            raise ValueError("Both sparse and dense indices are required for hybrid retrieval.")
        sparse = self.sparse.retrieve(query, top_k=fetch_k_each)
        dense = self.dense.retrieve(query, top_k=fetch_k_each)
        sparse_pairs = [(r.chunk_id, r.score) for r in sparse]
        dense_pairs = [(r.chunk_id, r.score) for r in dense]
        if method == "weighted":
            fused = weighted_score_fusion(dense_pairs, sparse_pairs, alpha=alpha, top_n=top_k)
        else:
            fused = reciprocal_rank_fusion({"dense": dense_pairs, "sparse": sparse_pairs}, top_n=top_k)
        out: list[RetrievedChunk] = []
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
