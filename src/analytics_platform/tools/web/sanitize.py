"""Treat web content as untrusted: drop hidden elements and instruction-like sentences."""

import re
from dataclasses import dataclass, field

import lxml.html

_HIDDEN_STYLE = re.compile(
    r"display\s*:\s*none|visibility\s*:\s*hidden|font-size\s*:\s*0|opacity\s*:\s*0(\.0+)?\s*(;|$)",
    re.I,
)
_INVISIBLE_CHARS = re.compile("[\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff]")

INJECTION_PATTERNS = {
    "override instructions": re.compile(
        r"\b(ignore|disregard|forget|override)\b[^.!?\n]{0,40}"
        r"\b(previous|prior|above|earlier|all|your|system)\b[^.!?\n]{0,20}"
        r"\b(instructions?|prompts?|rules|guidelines)\b",
        re.I,
    ),
    "role hijack": re.compile(
        r"\byou are now\b|\bact as\b[^.!?\n]{0,30}\b(assistant|ai|model)\b|\bdeveloper mode\b",
        re.I,
    ),
    "addressed to AI": re.compile(
        r"\b(note|message|instructions?) (to|for) (ai|llm|chatgpt|assistant|language model)s?\b",
        re.I,
    ),
    "fake system turn": re.compile(r"^\s*(system|assistant)\s*:|<\|?(im_start|system)\|?>", re.I),
}

_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass
class Sanitized:
    text: str
    flags: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)


def strip_hidden_html(html: str) -> tuple[str, list[str]]:
    """Remove elements a human reader would not see; attackers hide instructions there.

    Returns the cleaned HTML and the text of hidden elements (scripts and styles excluded).
    """
    try:
        root = lxml.html.fromstring(html)
    except (lxml.etree.ParserError, ValueError):
        return html, []
    hidden: list[str] = []
    for el in root.xpath(
        "//*[@hidden or @aria-hidden='true' or @style]|//script|//style|//noscript|//template"
    ):
        if el.tag in ("script", "style", "noscript", "template"):
            el.drop_tree()
        elif (
            el.get("hidden") is not None
            or el.get("aria-hidden") == "true"
            or _HIDDEN_STYLE.search(el.get("style") or "")
        ):
            hidden.append(el.text_content())
            el.drop_tree()
    return lxml.html.tostring(root, encoding="unicode"), hidden


def sanitize_text(text: str, max_chars: int) -> Sanitized:
    text = _INVISIBLE_CHARS.sub("", text)
    kept, removed, flags = [], [], set()
    for sentence in _SENTENCE.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        hits = [name for name, p in INJECTION_PATTERNS.items() if p.search(sentence)]
        if hits:
            flags.update(hits)
            removed.append(sentence)
        else:
            kept.append(sentence)
    clean = " ".join(kept)
    if len(clean) > max_chars:
        clean = clean[:max_chars].rsplit(" ", 1)[0] + " ..."
    return Sanitized(clean, sorted(flags), removed)
