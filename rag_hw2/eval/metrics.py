from __future__ import annotations

from collections import Counter
import re
from typing import Iterable


_WS_RE = re.compile(r"\s+")


def _normalize_whitespace(text) :
    return _WS_RE.sub(" ", text).strip()


def _normalize_answer_for_eval(s) :
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return _normalize_whitespace(s)


def exact_match(pred, gold) :
    return 1.0 if _normalize_answer_for_eval(pred) == _normalize_answer_for_eval(gold) else 0.0


def token_f1(pred, gold) :
    p_toks = _normalize_answer_for_eval(pred).split()
    g_toks = _normalize_answer_for_eval(gold).split()
    if not p_toks and not g_toks:
        return 1.0
    if not p_toks or not g_toks:
        return 0.0
    common = Counter(p_toks) & Counter(g_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(p_toks)
    recall = num_same / len(g_toks)
    return 2 * precision * recall / (precision + recall)


def _lcs_len(a, b) :
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


def rouge_l_f1(pred, gold) :
    p = _normalize_answer_for_eval(pred).split()
    g = _normalize_answer_for_eval(gold).split()
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


def _to_answer_list(gold) :
    if isinstance(gold, list):
        return [str(x) for x in gold]
    if isinstance(gold, str):
        if ";" in gold:
            return [x.strip() for x in gold.split(";") if x.strip()]
        return [gold]
    return [str(gold)]


class EvalSummary:
    def __init__(self, n, em, f1, rouge_l) :
        self.n = n
        self.em = em
        self.f1 = f1
        self.rouge_l = rouge_l


def score_predictions(preds, refs) :
    rows = []
    ems = []
    f1s = []
    rls = []
    for qid, gold in refs.items():
        pred = preds.get(str(qid), "")
        answers = _to_answer_list(gold)
        em = max(exact_match(pred, g) for g in answers)
        f1 = max(token_f1(pred, g) for g in answers)
        rl = max(rouge_l_f1(pred, g) for g in answers)
        rows.append(
            {
                "id": str(qid),
                "prediction": pred,
                "references": answers,
                "em": em,
                "f1": f1,
                "rouge_l": rl,
            }
        )
        ems.append(em)
        f1s.append(f1)
        rls.append(rl)
    n = len(rows)
    summary = EvalSummary(
        n=n,
        em=sum(ems) / n if n else 0.0,
        f1=sum(f1s) / n if n else 0.0,
        rouge_l=sum(rls) / n if n else 0.0,
    )
    return summary, rows
