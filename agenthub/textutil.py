"""Text helpers shared by retrieval, memory and the mock LLM (no third-party deps)."""
from __future__ import annotations

import re

_CJK = re.compile(r"[\u4e00-\u9fff]")
_LATIN = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    """Mixed zh/en tokenizer.

    * latin / digit runs -> lowercase words
    * CJK runs -> character bigrams (+ the single char for runs of length 1)

    Bigrams avoid a jieba dependency while still giving usable lexical recall for Chinese.
    """
    text = text.lower()
    tokens: list[str] = _LATIN.findall(text)
    run: list[str] = []

    def flush() -> None:
        if len(run) == 1:
            tokens.append(run[0])
        for i in range(len(run) - 1):
            tokens.append(run[i] + run[i + 1])
        run.clear()

    for ch in text:
        if _CJK.match(ch):
            run.append(ch)
        else:
            flush()
    flush()
    return tokens


def estimate_tokens(text: str) -> int:
    """Cheap token estimate: ~1 token per CJK char, ~1.3 per latin word."""
    cjk = len(_CJK.findall(text))
    words = len(re.findall(r"[A-Za-z0-9_]+", text))
    return cjk + int(words * 1.3) + 1
