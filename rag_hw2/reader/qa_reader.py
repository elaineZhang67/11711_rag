from __future__ import annotations

import re

from transformers import pipeline as hf_pipeline

from rag_hw2.types import Chunk


_WS_RE = re.compile(r"\s+")
_QUOTE_RE = re.compile(r'^[\"\'“”‘’`]+|[\"\'“”‘’`]+$')
_CITATION_TAIL_RE = re.compile(r"(?:\s*\[\d+\])+$")
_YEAR_RE = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
_DATE_RE = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|"
    r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?\s+\d{1,2},?\s+\d{4}\b",
    re.IGNORECASE,
)


def _normalize_whitespace(text) :
    return _WS_RE.sub(" ", text).strip()


def _simple_sentence_split(text) :
    text = text.strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", text)
    return [p.strip() for p in parts if p.strip()]


def build_rag_prompt(question, contexts, max_context_chars= 5000) :
    sections = []
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
        "Use the provided context first.\n"
        "When possible, copy the exact answer span from the context.\n"
        "Prefer one short canonical answer phrase (or one short sentence) when possible.\n"
        "If a short answer is sufficient (for example, a name, title, date, or place), answer it once only.\n"
        "For factual questions, answer directly and do not add extra explanation.\n"
        "Use explanation only for questions asking why/how/what happened/describe/explain/significance.\n"
        "Do not repeat aliases or alternate names unless the question explicitly asks for them.\n"
        "For date/year questions, return the exact date or year from the context when available.\n"
        "If the context is weak or missing, give your best answer based on your knowledge.\n"
        "Do your best to be concise, but include enough detail to answer correctly.\n\n"
        "Limit your final answer to at most 3 sentences.\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{context_block}\n\n"
        "Answer:"
    )


def _strip_outer_quotes(text) :
    text = text.strip()
    if not text:
        return text
    return _QUOTE_RE.sub("", text).strip()


def _dedupe_parts_keep_order(parts) :
    out = []
    seen = set()
    for p in parts:
        key = p.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _collapse_alias_list(text) :
    if ";" not in text:
        return text
    parts = [ _strip_outer_quotes(x.strip()) for x in text.split(";") if x.strip() ]
    parts = _dedupe_parts_keep_order(parts)
    if not parts:
        return text
    if len(parts) == 1:
        return parts[0]
    # If one answer is just a shorter alias/canonical form of another, keep the shorter one.
    lower_parts = [p.lower() for p in parts]
    for i, p in enumerate(parts):
        for j, q in enumerate(parts):
            if i == j:
                continue
            if lower_parts[i] and lower_parts[i] in lower_parts[j]:
                return p
    # Otherwise preserve the list (some questions may genuinely need multiple answers).
    return "; ".join(parts)


def _looks_plural_question(question) :
    if not question:
        return False
    q = question.lower().strip()
    plural_markers = [
        "what are",
        "who are",
        "which are",
        "list ",
        "name the",
        "what were",
        "who were",
        "which teams",
        "which events",
    ]
    return any(q.startswith(m) for m in plural_markers)


def _looks_singular_fact_question(question) :
    if not question:
        return False
    q = question.lower().strip()
    if _looks_plural_question(q):
        return False
    singular_markers = [
        "what is",
        "what was",
        "who is",
        "who was",
        "where is",
        "where was",
        "when was",
        "when did",
        "what year",
        "what date",
        "what month",
    ]
    return any(q.startswith(m) for m in singular_markers)


def _looks_year_question(question) :
    if not question:
        return False
    q = question.lower()
    return "what year" in q or (q.strip().startswith("when ") and "year" in q)


def _looks_when_question(question) :
    if not question:
        return False
    return question.lower().strip().startswith("when ")


def _looks_explanatory_question(question) :
    if not question:
        return False
    q = question.lower().strip()
    if q.startswith("why ") or q.startswith("how "):
        return True
    expl_markers = [
        "what happened",
        "describe",
        "explain",
        "significance",
        "importance",
        "impact",
        "role",
        "reason",
        "history of",
    ]
    return any(m in q for m in expl_markers)


def _collapse_semicolon_for_singular_question(text, question) :
    if ";" not in text:
        return text
    if not _looks_singular_fact_question(question):
        return text
    parts = [p.strip() for p in text.split(";") if p.strip()]
    if not parts:
        return text
    return parts[0]


def _collapse_parenthetical_alias(text) :
    text = text.strip()
    if "(" not in text or ")" not in text:
        return text
    m = re.match(r"^(.*?)\s*\((.*?)\)\s*$", text)
    if not m:
        return text
    left = _strip_outer_quotes(m.group(1).strip())
    inside = _strip_outer_quotes(m.group(2).strip())
    if not left or not inside:
        return text
    l = left.lower()
    r = inside.lower()
    if l in r:
        return left
    if r in l:
        return inside
    return text


def _normalize_date_or_year_for_question(text, question) :
    if not question:
        return text
    years = _YEAR_RE.findall(text)
    if _looks_year_question(question):
        if len(years) == 1:
            return years[0]
        return text
    if _looks_when_question(question):
        dates = _DATE_RE.findall(text)
        if len(dates) == 1:
            return _normalize_whitespace(dates[0])
        if len(years) == 1:
            return years[0]
    return text


def _remove_trailing_explanation(text) :
    # If model gives "short answer - explanation", keep the left side when it's clearly shorter.
    for sep in [" - ", " — ", " – ", " : "]:
        if sep in text:
            left, right = text.split(sep, 1)
            if 0 < len(left.split()) <= 10 and len(right.split()) > len(left.split()):
                return left.strip()
    for sep in [", because ", ", which ", ", who ", ", where ", ", and "]:
        if sep in text:
            left, right = text.split(sep, 1)
            if 0 < len(left.split()) <= 10 and len(right.split()) > len(left.split()):
                return left.strip()
    return text


def _keep_first_sentence_if_compact(text, question) :
    if not text or not _looks_singular_fact_question(question):
        return text
    sents = _simple_sentence_split(text)
    if len(sents) <= 1:
        return text
    first = sents[0].strip()
    if 0 < len(first.split()) <= 16:
        return first
    return text


def _truncate_to_max_sentences(text, max_sentences= 3) :
    sents = _simple_sentence_split(text)
    if len(sents) <= max_sentences:
        return text
    return " ".join(sents[:max_sentences]).strip()


def postprocess_answer(text, question= None) :
    text = text.strip()
    for prefix in [
        "Answer:",
        "The answer is",
        "Based on the context,",
        "Based on the provided context,",
    ]:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix) :].strip(" :,-")
    # Keep first line for concise leaderboard-friendly outputs.
    text = text.splitlines()[0].strip() if text else text
    # Trim wrapping quotes and boilerplate punctuation.
    text = _strip_outer_quotes(text)
    text = text.strip(" \t:,-")
    text = _CITATION_TAIL_RE.sub("", text).strip()
    # If the model outputs "answer + long alias", prefer the canonical shorter form.
    text = _collapse_alias_list(text)
    text = _collapse_semicolon_for_singular_question(text, question)
    text = _collapse_parenthetical_alias(text)
    text = _remove_trailing_explanation(text)
    text = _normalize_date_or_year_for_question(text, question)
    text = _keep_first_sentence_if_compact(text, question)
    max_sents = 3 if _looks_explanatory_question(question) else 2
    text = _truncate_to_max_sentences(text, max_sentences=max_sents)
    # Normalize some common non-answer fillers.
    low = text.lower()
    if low in {"unknown.", "unknown", "not found", "not provided", "insufficient information"}:
        text = "UNKNOWN"
    # Remove trailing sentence punctuation if answer looks like a short span.
    if len(text.split()) <= 12:
        text = text.rstrip(" .")
    return _normalize_whitespace(text)


class Reader:
    def answer(self, question, contexts) :
        raise NotImplementedError

    def generate_hypothesis(self, question, max_new_tokens= 64) :
        return ""


class TransformersReader:
    def __init__(
        self,
        model_name,
        task= "text2text-generation",
        max_new_tokens= 120,
        temperature= 0.0,
        device= None,
        max_context_chars= 5000,
    ) :
        self.model_name = model_name
        self.task = task  # text2text-generation or text-generation
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device
        self.max_context_chars = max_context_chars
        self.__post_init__()

    def __post_init__(self) :
        kwargs = {
            "task": self.task,
            "model": self.model_name,
            "tokenizer": self.model_name,
        }
        if self.device is not None:
            kwargs["device"] = self.device
        self.pipe = hf_pipeline(**kwargs)

    def answer(self, question, contexts) :
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
        return postprocess_answer(raw, question=question)

    def generate_hypothesis(self, question, max_new_tokens= 64) :
        prompt = (
            "Write a short factual passage that likely answers the question.\n"
            "Use 1-2 concise sentences with concrete entities, dates, and places when relevant.\n"
            "Do not mention sources or uncertainty.\n\n"
            f"Question: {question}\n"
            "Passage:"
        )
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
        }
        out = self.pipe(prompt, **gen_kwargs)
        if not out:
            return ""
        raw = out[0].get("generated_text") or out[0].get("summary_text") or ""
        if self.task == "text-generation" and raw.startswith(prompt):
            raw = raw[len(prompt) :]
        text = _normalize_whitespace(raw)
        sents = _simple_sentence_split(text)
        if len(sents) > 2:
            text = " ".join(sents[:2]).strip()
        words = text.split()
        if len(words) > 60:
            text = " ".join(words[:60]).strip()
        return text


def make_reader(
    backend,
    model_name= None,
    task= "text2text-generation",
    max_new_tokens= 120,
    temperature= 0.0,
    device= None,
    max_context_chars= 5000,
) :
    if backend == "transformers":
        if not model_name:
            raise ValueError("model_name is required for transformers reader")
        return TransformersReader(
            model_name=model_name,
            task=task,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            device=device,
            max_context_chars=max_context_chars,
        )
    raise ValueError(f"Unknown reader backend: {backend}")
