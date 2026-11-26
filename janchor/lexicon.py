"""Refusal and harmfulness lexicons.  Shared across models; each model contributes every single-token surface form."""
from __future__ import annotations

REFUSE = ["sorry", "Sorry", "apolog", "cannot", "Cannot", "can't", "unable", "Unable", "refuse", "decline", "won't",
          "抱歉", "无法", "不能", "拒绝", "无法满足", "不能满足", "对不起"]
APOLOGY = ["apologize", "apologise", "apologies"]
HARM = ["illegal", "unethical", "harmful", "dangerous", "malicious", "immoral", "unlawful", "inappropriate", "unsafe",
        "criminal", "wrong", "prohibited", "forbidden", "offensive", "unacceptable", "harm", "danger", "crime"]
HARM_STEMS = ["unsafe", "illegal", "harmful", "dangerous", "forbidden", "prohibit", "unethical", "unlawful", "malicious",
              "criminal", "illicit", "danger", "immoral", "不安全", "危险", "违法", "非法", "запрещ", "опасн"]


def _single(tok, words) -> set[int]:
    """Token ids of the words that are a single token, with and without a leading space and capitalisation."""
    out = set()
    for w in words:
        for v in (w, " " + w, w.capitalize(), " " + w.capitalize()):
            e = tok.encode(v, add_special_tokens=False)
            if len(e) == 1:
                out.add(e[0])
    return out


def refusal_ids(tok, vocab_size: int) -> list[int]:
    ids = _single(tok, REFUSE) | _single(tok, APOLOGY)
    ids |= {tok.encode(s, add_special_tokens=False)[-1] for s in (" can't", " can’t")}
    ids |= {i for i in range(vocab_size) if tok.decode([i]).strip().lower().startswith("apolog")}
    return sorted(ids)


def harm_ids(tok, vocab_size: int) -> list[int]:
    ids = _single(tok, HARM)
    for i in range(vocab_size):
        t = tok.decode([i]).strip().lower()
        if t and any(t.startswith(s) for s in HARM_STEMS):
            ids.add(i)
    return sorted(ids)
