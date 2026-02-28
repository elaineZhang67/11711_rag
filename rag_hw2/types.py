from __future__ import annotations

from typing import Any


class Document:
    def __init__(
        self,
        doc_id,
        text,
        title= None,
        source_path= None,
        source_url= None,
        doc_type= None,
        metadata= None,
    ) :
        self.doc_id = doc_id
        self.text = text
        self.title = title
        self.source_path = source_path
        self.source_url = source_url
        self.doc_type = doc_type
        self.metadata = dict(metadata) if metadata else {}

    def to_dict(self) :
        return {
            "doc_id": self.doc_id,
            "text": self.text,
            "title": self.title,
            "source_path": self.source_path,
            "source_url": self.source_url,
            "doc_type": self.doc_type,
            "metadata": dict(self.metadata),
        }

class Chunk:
    def __init__(
        self,
        chunk_id,
        doc_id,
        text,
        title= None,
        source_path= None,
        source_url= None,
        start_char= None,
        end_char= None,
        chunk_index= 0,
        metadata= None,
    ) :
        self.chunk_id = chunk_id
        self.doc_id = doc_id
        self.text = text
        self.title = title
        self.source_path = source_path
        self.source_url = source_url
        self.start_char = start_char
        self.end_char = end_char
        self.chunk_index = chunk_index
        self.metadata = dict(metadata) if metadata else {}

    def to_dict(self) :
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "title": self.title,
            "source_path": self.source_path,
            "source_url": self.source_url,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "chunk_index": self.chunk_index,
            "metadata": dict(self.metadata),
        }

class RetrievedChunk:
    def __init__(
        self,
        chunk_id,
        score,
        rank,
        source,
        chunk= None,
        component_scores= None,
    ) :
        self.chunk_id = chunk_id
        self.score = score
        self.rank = rank
        self.source = source  # "sparse" / "dense" / "hybrid"
        self.chunk = chunk
        self.component_scores = dict(component_scores) if component_scores else {}


class QueryItem:
    def __init__(self, qid, question) :
        self.qid = qid
        self.question = question

def document_from_dict(value) :
    return Document(**value)


def chunk_from_dict(value) :
    return Chunk(**value)


def query_item_from_dict(value) :
    qid = str(value.get("id", value.get("qid", "")))
    question = str(value["question"])
    return QueryItem(qid=qid, question=question)
