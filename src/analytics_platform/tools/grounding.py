"""Checks that an LLM's quoted evidence really appears in the source text."""

import re

_QUOTES = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"'})


def normalise(text: str) -> str:
    text = text.translate(_QUOTES)
    return re.sub(r"\s+", " ", re.sub(r"[*_`]", "", text)).strip().lower().rstrip(".")


def quote_in(quote: str, texts: list[str]) -> bool:
    q = normalise(quote)
    return bool(q) and any(q in normalise(t) for t in texts)
