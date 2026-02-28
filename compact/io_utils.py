from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Iterable


def ensure_parent(path) :
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dir(path) :
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path) :
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(obj, path, indent= 2) :
    path = ensure_parent(path)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=indent), encoding="utf-8")


def read_jsonl(path) :
    records = []
    p = Path(path)
    if not p.exists():
        return records
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(records, path) :
    p = ensure_parent(path)
    with p.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def append_jsonl(records, path) :
    p = ensure_parent(path)
    with p.open("a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def save_pickle(obj, path) :
    p = ensure_parent(path)
    with p.open("wb") as f:
        pickle.dump(obj, f)


def load_pickle(path) :
    with Path(path).open("rb") as f:
        return pickle.load(f)

