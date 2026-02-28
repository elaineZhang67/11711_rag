from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Protocol

from transformers import pipeline as hf_pipeline

from compact.text_utils import normalize_answer_for_eval, normalize_whitespace, simple_sentence_split
from rag_hw2.types import Chunk, QueryItem, RetrievedChunk


def load_queries(path: str | Path) -> list[QueryItem]:
    data: Any = json.loads(Path(path).read_text(encoding="utf-8"))
    items: list[QueryItem] = []
    if isinstance(data, list):
        for i, item in enumerate(data, start=1):
            if isinstance(item, str):
                items.append(QueryItem(qid=str(i), question=item))
            else:
                items.append(QueryItem.from_dict(item))
    elif isinstance(data, dict):
        for qid, question in data.items():
            if qid == "andrewid":
                continue
            items.append(QueryItem(qid=str(qid), question=str(question)))
    else:
        raise ValueError(f"Unsupported query format in {path}")
    return items


def build_submission_json(answers: dict[str, str], andrewid: str | None = None) -> dict[str, str]:
    out: dict[str, str] = {}
    if andrewid:
        out["andrewid"] = andrewid
    for qid in sorted(answers, key=lambda x: int(x) if str(x).isdigit() else str(x)):
        out[str(qid)] = answers[qid]
    return out


def build_rag_prompt(question: str, contexts: list[Chunk], max_context_chars: int = 5000) -> str:
    sections: list[str] = []
    used = 0
    for i, c in enumerate(contexts, start=1):
        header_parts = [f"[{i}]"]
        if c.title:
            header_parts.append(c.title)
        if c.source_url:
            header_parts.append(c.source_url)
        snippet = c.text.strip()
        room = max_context_chars - used
        if room <= 0:
            break
        snippet = snippet[:room]
        used += len(snippet)
        sections.append(f"{' | '.join(header_parts)}\n{snippet}")
    context_block = "\n\n".join(sections) if sections else "(no retrieved context provided)"
    return (
        "You are answering factual questions about Pittsburgh and Carnegie Mellon University.\n"
        "Use the provided context. If the answer is in the context, answer concisely with only the answer phrase.\n"
        "If multiple answers are explicitly correct, separate them with semicolons.\n"
        "If the context is insufficient, give your best short answer and avoid long explanations.\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{context_block}\n\n"
        "Answer:"
    )


def postprocess_answer(text: str) -> str:
    text = text.strip()
    for prefix in [
        "Answer:",
        "The answer is",
        "Based on the context,",
        "Based on the provided context,",
    ]:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix) :].strip(" :,-")
    text = text.splitlines()[0].strip() if text else text
    if len(text.split()) <= 12:
        text = text.rstrip(" .")
    return normalize_whitespace(text)


class Reader(Protocol):
    def answer(self, question: str, contexts: list[Chunk]) -> str:
        ...


class HeuristicReader:
    def __init__(self) -> None:
        pass

    def answer(self, question: str, contexts: list[Chunk]) -> str:
        if not contexts:
            return ""
        top = contexts[0].text.strip()
        sents = simple_sentence_split(top)
        cand = sents[0] if sents else top.split("\n")[0]
        return postprocess_answer(cand[:240])


class TransformersReader:
    def __init__(
        self,
        model_name: str,
        task: str = "text2text-generation",
        max_new_tokens: int = 48,
        temperature: float = 0.0,
        device: str | int | None = None,
        max_context_chars: int = 5000,
    ) -> None:
        self.model_name = model_name
        self.task = task
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device
        self.max_context_chars = max_context_chars
        self.__post_init__()

    def __post_init__(self) -> None:
        kwargs = {"task": self.task, "model": self.model_name, "tokenizer": self.model_name}
        if self.device is not None:
            kwargs["device"] = self.device
        self.pipe = hf_pipeline(**kwargs)

    def answer(self, question: str, contexts: list[Chunk]) -> str:
        prompt = build_rag_prompt(question, contexts, max_context_chars=self.max_context_chars)
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
            "temperature": self.temperature if self.temperature > 0 else None,
        }
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
        out = self.pipe(prompt, **gen_kwargs)
        if not out:
            return ""
        raw = out[0].get("generated_text") or out[0].get("summary_text") or ""
        if self.task == "text-generation" and raw.startswith(prompt):
            raw = raw[len(prompt) :]
        return postprocess_answer(raw)


def make_reader(
    backend: str,
    model_name: str | None = None,
    task: str = "text2text-generation",
    max_new_tokens: int = 48,
    temperature: float = 0.0,
    device: str | int | None = None,
    max_context_chars: int = 5000,
) -> Reader:
    if backend == "heuristic":
        return HeuristicReader()
    if backend == "transformers":
        if not model_name:
            raise ValueError("model_name is required for transformers reader")
        return TransformersReader(model_name, task, max_new_tokens, temperature, device, max_context_chars)
    raise ValueError(f"Unknown reader backend: {backend}")


class RetrievalConfig:
    def __init__(
        self,
        mode: str = "hybrid",
        top_k: int = 5,
        fetch_k_each: int = 30,
        fusion_method: str = "rrf",
        fusion_alpha: float = 0.5,
    ) -> None:
        self.mode = mode
        self.top_k = top_k
        self.fetch_k_each = fetch_k_each
        self.fusion_method = fusion_method
        self.fusion_alpha = fusion_alpha

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "top_k": self.top_k,
            "fetch_k_each": self.fetch_k_each,
            "fusion_method": self.fusion_method,
            "fusion_alpha": self.fusion_alpha,
        }


class RAGPipeline:
    def __init__(self, retriever, reader: Reader, retrieval_cfg: RetrievalConfig):
        self.retriever = retriever
        self.reader = reader
        self.cfg = retrieval_cfg

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        mode = self.cfg.mode
        if mode == "closedbook":
            return []
        if self.retriever is None:
            raise ValueError("Retriever not available.")
        if mode == "sparse":
            return self.retriever.retrieve_sparse(question, top_k=self.cfg.top_k)
        if mode == "dense":
            return self.retriever.retrieve_dense(question, top_k=self.cfg.top_k)
        if mode == "hybrid":
            return self.retriever.retrieve_hybrid(
                question,
                top_k=self.cfg.top_k,
                fetch_k_each=self.cfg.fetch_k_each,
                method=self.cfg.fusion_method,
                alpha=self.cfg.fusion_alpha,
            )
        raise ValueError(f"Unknown retrieval mode: {mode}")

    def answer_query(self, qid: str, question: str) -> dict[str, Any]:
        retrieved = self.retrieve(question)
        contexts = [r.chunk for r in retrieved if r.chunk is not None]
        answer = (self.reader.answer(question, contexts) or "").strip()
        trace: list[dict[str, Any]] = []
        for r in retrieved:
            ch = r.chunk
            trace.append(
                {
                    "chunk_id": r.chunk_id,
                    "rank": r.rank,
                    "score": r.score,
                    "source": r.source,
                    "component_scores": r.component_scores,
                    "title": ch.title if ch else None,
                    "source_url": ch.source_url if ch else None,
                    "source_path": ch.source_path if ch else None,
                    "text_preview": (ch.text[:300] if ch else None),
                }
            )
        return {"id": qid, "question": question, "answer": answer, "retrieval": trace, "config": self.cfg.to_dict()}


def exact_match(pred: str, gold: str) -> float:
    return 1.0 if normalize_answer_for_eval(pred) == normalize_answer_for_eval(gold) else 0.0


def token_f1(pred: str, gold: str) -> float:
    p_toks = normalize_answer_for_eval(pred).split()
    g_toks = normalize_answer_for_eval(gold).split()
    if not p_toks and not g_toks:
        return 1.0
    if not p_toks or not g_toks:
        return 0.0
    common = Counter(p_toks) & Counter(g_toks)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    precision = n_same / len(p_toks)
    recall = n_same / len(g_toks)
    return 2 * precision * recall / (precision + recall)


def _lcs_len(a: list[str], b: list[str]) -> int:
    dp = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        prev = 0
        for j in range(1, len(b) + 1):
            tmp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = tmp
    return dp[-1]


def rouge_l_f1(pred: str, gold: str) -> float:
    p = normalize_answer_for_eval(pred).split()
    g = normalize_answer_for_eval(gold).split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    lcs = _lcs_len(p, g)
    if lcs == 0:
        return 0.0
    prec = lcs / len(p)
    rec = lcs / len(g)
    return 2 * prec * rec / (prec + rec)


def _to_answer_list(gold: str | list[str]) -> list[str]:
    if isinstance(gold, list):
        return [str(x) for x in gold]
    if isinstance(gold, str) and ";" in gold:
        return [x.strip() for x in gold.split(";") if x.strip()]
    return [str(gold)]


class EvalSummary:
    def __init__(self, n: int, em: float, f1: float, rouge_l: float) -> None:
        self.n = n
        self.em = em
        self.f1 = f1
        self.rouge_l = rouge_l


def score_predictions(preds: dict[str, str], refs: dict[str, str | list[str]]) -> tuple[EvalSummary, list[dict]]:
    rows: list[dict] = []
    ems: list[float] = []
    f1s: list[float] = []
    rls: list[float] = []
    for qid, gold in refs.items():
        pred = preds.get(str(qid), "")
        answers = _to_answer_list(gold)
        em = max(exact_match(pred, g) for g in answers)
        f1 = max(token_f1(pred, g) for g in answers)
        rl = max(rouge_l_f1(pred, g) for g in answers)
        rows.append({"id": str(qid), "prediction": pred, "references": answers, "em": em, "f1": f1, "rouge_l": rl})
        ems.append(em)
        f1s.append(f1)
        rls.append(rl)
    n = len(rows)
    return (
        EvalSummary(n=n, em=sum(ems) / n if n else 0.0, f1=sum(f1s) / n if n else 0.0, rouge_l=sum(rls) / n if n else 0.0),
        rows,
    )
