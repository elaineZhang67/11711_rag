from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from rag_hw2.types import Chunk, Document, document_from_dict


class ChunkingConfig:
    def __init__(
        self,
        strategy= "fixed",
        chunk_size_words= 250,
        overlap_words= 50,
        min_chunk_words= 20,
    ) :
        self.strategy = strategy  # fixed | sentence
        self.chunk_size_words = chunk_size_words
        self.overlap_words = overlap_words
        self.min_chunk_words = min_chunk_words


def _read_jsonl(path) :
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(records, path) :
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _simple_sentence_split(text) :
    text = text.strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", text)
    return [p.strip() for p in parts if p.strip()]


def word_count(text) :
    return len(text.split())


def _chunk_id(doc_id, idx, text) :
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"{doc_id}_{idx}_{h}"


def _find_span(text, chunk_text, start_hint= 0) :
    pos = text.find(chunk_text, start_hint)
    if pos < 0:
        return None, None
    return pos, pos + len(chunk_text)


def _fixed_word_chunks(text, cfg) :
    words = text.split()
    if not words:
        return []
    size = max(1, cfg.chunk_size_words)
    overlap = max(0, min(cfg.overlap_words, size - 1))
    step = max(1, size - overlap)
    chunks = []
    for start in range(0, len(words), step):
        piece = words[start : start + size]
        if not piece:
            continue
        if chunks and len(piece) < cfg.min_chunk_words:
            chunks[-1] = chunks[-1] + " " + " ".join(piece)
            break
        chunks.append(" ".join(piece))
        if start + size >= len(words):
            break
    return chunks


def _sentence_aware_chunks(text, cfg) :
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    units = []
    for p in paragraphs:
        if word_count(p) <= cfg.chunk_size_words:
            units.append(p)
        else:
            sents = _simple_sentence_split(p)
            if not sents:
                units.extend(_fixed_word_chunks(p, cfg))
            else:
                units.extend(sents)

    chunks = []
    cur = []
    cur_words = 0
    target = cfg.chunk_size_words
    overlap_words = cfg.overlap_words
    for unit in units:
        uw = word_count(unit)
        if cur and cur_words + uw > target:
            chunks.append(" ".join(cur).strip())
            if overlap_words > 0:
                # Rebuild overlap from the tail of current chunk.
                tail_words = " ".join(cur).split()[-overlap_words:]
                cur = [" ".join(tail_words)] if tail_words else []
                cur_words = len(tail_words)
            else:
                cur = []
                cur_words = 0
        cur.append(unit)
        cur_words += uw
    if cur:
        chunks.append(" ".join(cur).strip())
    # Merge tiny tail
    if len(chunks) >= 2 and word_count(chunks[-1]) < cfg.min_chunk_words:
        chunks[-2] = chunks[-2] + " " + chunks[-1]
        chunks.pop()
    return [c for c in chunks if c]


def chunk_document(doc, cfg) :
    if cfg.strategy == "sentence":
        pieces = _sentence_aware_chunks(doc.text, cfg)
    else:
        pieces = _fixed_word_chunks(doc.text, cfg)
    chunks = []
    cursor = 0
    for i, piece in enumerate(pieces):
        piece = piece.strip()
        if not piece:
            continue
        start, end = _find_span(doc.text, piece, cursor)
        if start is not None:
            cursor = start + 1
        chunks.append(
            Chunk(
                chunk_id=_chunk_id(doc.doc_id, i, piece),
                doc_id=doc.doc_id,
                text=piece,
                title=doc.title,
                source_path=doc.source_path,
                source_url=doc.source_url,
                start_char=start,
                end_char=end,
                chunk_index=i,
                metadata={"doc_type": doc.doc_type, **(doc.metadata or {})},
            )
        )
    return chunks


def build_chunks(
    documents_jsonl,
    output_jsonl,
    cfg,
    verbose= False,
) :
    docs = [document_from_dict(x) for x in _read_jsonl(documents_jsonl)]
    all_chunks = []
    for doc in docs:
        chunks = chunk_document(doc, cfg)
        all_chunks.extend(chunks)
        if verbose:
            print(f"[chunk] {doc.doc_id} -> {len(chunks)} chunks")
    _write_jsonl([c.to_dict() for c in all_chunks], output_jsonl)
    return all_chunks
