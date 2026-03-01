from __future__ import annotations

import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from rag_hw2.types import RetrievedChunk
from rag_hw2.retrieval.chunk_store import ChunkStore


class DenseIndexArtifacts:
    def __init__(
        self,
        chunk_ids,
        embedding_model,
        normalize_embeddings= True,
        embedding_text_mode= "chunk",
    ) :
        self.chunk_ids = list(chunk_ids)
        self.embedding_model = embedding_model
        self.normalize_embeddings = normalize_embeddings
        self.embedding_text_mode = embedding_text_mode


def _save_pickle(obj, path) :
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
        pickle.dump(obj, f)


def _load_pickle(path):
    with Path(path).open("rb") as f:
        return pickle.load(f)


class DenseFAISSIndex:
    def __init__(
        self,
        index,
        encoder,
        artifacts,
        chunk_store= None,
    ):
        self.index = index
        self.encoder = encoder
        self.artifacts = artifacts
        self.chunk_store = chunk_store

    def save(self, out_dir) :
        faiss_lib = _load_faiss_lib()
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        faiss_lib.write_index(self.index, str(out / "dense.index"))
        _save_pickle(self.artifacts, out / "dense_artifacts.pkl")

    def retrieve(self, query, top_k= 10) :
        q = self.encoder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=self.artifacts.normalize_embeddings,
        )
        q = np.asarray(q, dtype=np.float32)
        scores, ids = self.index.search(q, top_k)
        results = []
        for rank, (idx, score) in enumerate(zip(ids[0].tolist(), scores[0].tolist()), start=1):
            if idx < 0:
                continue
            cid = self.artifacts.chunk_ids[idx]
            ch = None
            if self.chunk_store:
                ch = self.chunk_store.by_id.get(cid)
                if ch is None:
                    raise KeyError(
                        f"Chunk id {cid!r} not found in loaded chunk store. "
                        "This usually means the dense index was built from a different chunks.jsonl file. "
                        "Rebuild the dense index from the same chunks file passed to answer_queries.py."
                    )
            results.append(
                RetrievedChunk(
                    chunk_id=cid,
                    score=float(score),
                    rank=rank,
                    source="dense",
                    chunk=ch,
                    component_scores={"dense": float(score)},
                )
            )
        return results


def _load_faiss_lib():
    return faiss


def _load_encoder_model(model_name):
    return SentenceTransformer(model_name)


def build_dense_faiss_index(
    chunk_store,
    model_name= "BAAI/bge-large-en-v1.5",
    batch_size= 64,
    normalize_embeddings= True,
    embed_texts_by_chunk_id= None,
    embedding_text_mode= "chunk",
    verbose= False,
) :
    faiss_lib = _load_faiss_lib()
    encoder = _load_encoder_model(model_name)
    use_alt = bool(embed_texts_by_chunk_id)
    alt_used = 0
    texts = []
    for c in chunk_store.chunks:
        txt = c.text
        if use_alt:
            alt = str(embed_texts_by_chunk_id.get(c.chunk_id, "")).strip()
            if alt:
                txt = alt
                alt_used += 1
        texts.append(txt)
    if verbose:
        print(f"[dense] embedding {len(texts)} chunks with {model_name} (mode={embedding_text_mode})")
        if use_alt:
            print(f"[dense] using alternate embed text for {alt_used}/{len(texts)} chunks")
    embeddings = encoder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=verbose,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    dim = embeddings.shape[1]
    index = faiss_lib.IndexFlatIP(dim)
    index.add(embeddings)
    return DenseFAISSIndex(
        index=index,
        encoder=encoder,
        artifacts=DenseIndexArtifacts(
            chunk_ids=chunk_store.ids(),
            embedding_model=model_name,
            normalize_embeddings=normalize_embeddings,
            embedding_text_mode=embedding_text_mode,
        ),
        chunk_store=chunk_store,
    )


def load_dense_faiss_index(out_dir, chunk_store= None) :
    faiss_lib = _load_faiss_lib()
    out = Path(out_dir)
    artifacts = _load_pickle(out / "dense_artifacts.pkl")
    index = faiss_lib.read_index(str(out / "dense.index"))
    encoder = _load_encoder_model(artifacts.embedding_model)
    return DenseFAISSIndex(index=index, encoder=encoder, artifacts=artifacts, chunk_store=chunk_store)
