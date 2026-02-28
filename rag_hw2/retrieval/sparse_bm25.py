from __future__ import annotations

import pickle
import re
import unicodedata
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from rag_hw2.types import RetrievedChunk
from rag_hw2.retrieval.chunk_store import ChunkStore


class SparseIndexArtifacts:
    def __init__(
        self,
        chunk_ids,
        tokenized_texts,
        model_name= "rank_bm25",
    ) :
        self.chunk_ids = list(chunk_ids)
        self.tokenized_texts = [list(tokens) for tokens in tokenized_texts]
        self.model_name = model_name


_NONWORD_RE = re.compile(r"[^A-Za-z0-9]+")


def _bm25_tokenize(text) :
    text = unicodedata.normalize("NFKC", text).lower()
    text = _NONWORD_RE.sub(" ", text)
    return [t for t in text.split() if t]


def _save_pickle(obj, path) :
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
        pickle.dump(obj, f)


def _load_pickle(path):
    with Path(path).open("rb") as f:
        return pickle.load(f)


class SparseBM25Index:
    def __init__(self, bm25, artifacts, chunk_store= None):
        self.bm25 = bm25
        self.artifacts = artifacts
        self.chunk_store = chunk_store
        self._id_to_pos = {cid: i for i, cid in enumerate(self.artifacts.chunk_ids)}

    def save(self, out_dir) :
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        _save_pickle(self.bm25, out / "bm25_model.pkl")
        _save_pickle(self.artifacts, out / "bm25_artifacts.pkl")

    def retrieve(self, query, top_k= 10) :
        q_tokens = _bm25_tokenize(query)
        scores = np.asarray(self.bm25.get_scores(q_tokens), dtype=float)
        if scores.size == 0:
            return []
        k = min(top_k, scores.size)
        idx = np.argpartition(-scores, kth=k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        results = []
        for rank, i in enumerate(idx.tolist(), start=1):
            cid = self.artifacts.chunk_ids[i]
            ch = None
            if self.chunk_store:
                ch = self.chunk_store.by_id.get(cid)
                if ch is None:
                    raise KeyError(
                        f"Chunk id {cid!r} not found in loaded chunk store. "
                        "This usually means the sparse index was built from a different chunks.jsonl file. "
                        "Rebuild the sparse index (and dense index, if using hybrid) from the same chunks file passed to answer_queries.py."
                    )
            results.append(
                RetrievedChunk(
                    chunk_id=cid,
                    score=float(scores[i]),
                    rank=rank,
                    source="sparse",
                    chunk=ch,
                    component_scores={"sparse": float(scores[i])},
                )
            )
        return results


def build_sparse_bm25_index(chunk_store) :
    tokenized = [_bm25_tokenize(c.text) for c in chunk_store.chunks]
    bm25 = BM25Okapi(tokenized)
    return SparseBM25Index(
        bm25=bm25,
        artifacts=SparseIndexArtifacts(chunk_ids=chunk_store.ids(), tokenized_texts=tokenized),
        chunk_store=chunk_store,
    )


def load_sparse_bm25_index(out_dir, chunk_store= None) :
    out = Path(out_dir)
    bm25 = _load_pickle(out / "bm25_model.pkl")
    artifacts = _load_pickle(out / "bm25_artifacts.pkl")
    return SparseBM25Index(bm25=bm25, artifacts=artifacts, chunk_store=chunk_store)
