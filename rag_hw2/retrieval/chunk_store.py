from __future__ import annotations

import json
from pathlib import Path

from rag_hw2.types import Chunk, chunk_from_dict


class ChunkStore:
    def __init__(self, chunks):
        self.chunks = chunks
        self.by_id = {c.chunk_id: c for c in chunks}

    def texts(self):
        return [c.text for c in self.chunks]

    def ids(self):
        return [c.chunk_id for c in self.chunks]


def load_chunk_store_from_jsonl(path):
    p = Path(path)
    rows = []
    if p.exists():
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    chunks = [chunk_from_dict(x) for x in rows]
    return ChunkStore(chunks)
