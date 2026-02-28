from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from rag_hw2.types import QueryItem, query_item_from_dict


def load_queries(path) :
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    items = []
    if isinstance(data, list):
        for i, item in enumerate(data, start=1):
            if isinstance(item, str):
                items.append(QueryItem(qid=str(i), question=item))
            else:
                items.append(query_item_from_dict(item))
    elif isinstance(data, dict):
        for qid, question in data.items():
            if qid == "andrewid":
                continue
            items.append(QueryItem(qid=str(qid), question=str(question)))
    else:
        raise ValueError(f"Unsupported query format in {path}")
    return items


def build_submission_json(
    answers,
    andrewid= None,
) :
    out = {}
    if andrewid:
        out["andrewid"] = andrewid
    for qid in sorted(answers, key=lambda x: int(x) if str(x).isdigit() else str(x)):
        out[str(qid)] = answers[qid]
    return out
