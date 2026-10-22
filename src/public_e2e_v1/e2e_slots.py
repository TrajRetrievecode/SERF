#!/usr/bin/env python3
"""Query-text-only decomposition into claim slots, for datasets that have none.

The method defines the claim slots of a query as a function of the query text
alone.  On the evaluation set that function reads an annotated (A)/(B)
sub-question structure.  Public queries carry no such annotation, so the
function has to be implemented for them; this module is that implementation.

It is deliberately mechanical: split on sentence and clause delimiters and on
a fixed list of coordinators, keep the pieces that are long enough to carry
content, and emit at most two slots with the whole query as the fallback.
No model, no label and no tuning is involved, so the decomposition is fixed
before any record is scored and is identical for every method.
"""

from __future__ import annotations

import re
import unicodedata

MAX_SLOTS = 2
MIN_LATIN_TOKENS = 3
MIN_CJK_CHARS = 6

_CJK = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
_DELIMITERS = re.compile(
    r"(?:[?？;；]|,\s+and\s+|\s+and\s+|\s+as\s+well\s+as\s+|、|以及|并且|同时|，还有)",
    re.IGNORECASE,
)


def _is_cjk(text: str) -> bool:
    return bool(_CJK.search(text))


def _long_enough(piece: str) -> bool:
    piece = piece.strip()
    if not piece:
        return False
    if _is_cjk(piece):
        return len(_CJK.findall(piece)) >= MIN_CJK_CHARS
    return len(piece.split()) >= MIN_LATIN_TOKENS


def decompose(query: str) -> list[str]:
    """Return between one and MAX_SLOTS slot texts for a query."""
    normalized = unicodedata.normalize("NFKC", str(query)).strip()
    if not normalized:
        return [""]
    pieces = [p.strip() for p in _DELIMITERS.split(normalized) if p and p.strip()]
    pieces = [p for p in pieces if _long_enough(p)]
    if len(pieces) < 2:
        return [normalized]
    head = pieces[0]
    tail = " ".join(pieces[1:])
    slots = [head, tail][:MAX_SLOTS]
    # A slot that merely repeats the whole query adds nothing.
    if len(slots) == 2 and slots[0] == slots[1]:
        return [normalized]
    return slots


def slot_texts_for(records) -> tuple[list[str], list[tuple[int, int]]]:
    """Flatten every record's slots into one list plus per-record spans."""
    texts: list[str] = []
    spans: list[tuple[int, int]] = []
    for record in records:
        slots = decompose(record.query)
        spans.append((len(texts), len(texts) + len(slots)))
        texts.extend(slots)
    return texts, spans


def statistics(records) -> dict:
    counts: dict[int, int] = {}
    for record in records:
        n = len(decompose(record.query))
        counts[n] = counts.get(n, 0) + 1
    total = sum(counts.values()) or 1
    return {
        "slots_per_query": dict(sorted(counts.items())),
        "mean_slots": sum(k * v for k, v in counts.items()) / total,
        "max_slots": MAX_SLOTS,
        "rule": (
            "split on sentence/clause delimiters and a fixed coordinator list, "
            "drop pieces below the length floor, emit at most two slots, fall "
            "back to the whole query"
        ),
        "query_text_only": True,
        "uses_labels": False,
        "tuned": False,
    }
