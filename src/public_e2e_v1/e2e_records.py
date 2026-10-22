#!/usr/bin/env python3
"""Query records for the two public datasets, with gold evidence attached.

Gold fields are carried alongside the record but are only read by the
evaluator.  ``inference_view`` returns the strict subset a method may see.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from e2e_common import ROOT
from e2e_corpus import FINRAG_ROOT, VISDOM_ROOT, finrag_gold_pages

from fedresolve_core import load_finrag_records, load_visdom_records


@dataclass(frozen=True)
class E2ERecord:
    uid: str
    dataset: str
    lang: str
    query_id: str
    query: str
    answer: str
    category: str
    answer_type: str
    doc_keys: tuple[str, ...]
    gold_pages: tuple[str, ...]
    gold_unit: str  # "page" or "document"

    def inference_view(self) -> dict:
        """Exactly what any method is allowed to observe about the query."""
        return {
            "uid": self.uid,
            "dataset": self.dataset,
            "lang": self.lang,
            "query": self.query,
        }


def load_records(dataset: str) -> tuple[list[E2ERecord], dict, tuple[str, ...]]:
    if dataset == "finrag":
        raw, audit = load_finrag_records(FINRAG_ROOT)
        records = [
            E2ERecord(
                uid=r.uid,
                dataset="finrag",
                lang=r.lang,
                query_id=r.query_id,
                query=r.query,
                answer=r.answer,
                category=r.category,
                answer_type=r.answer_type,
                doc_keys=tuple(r.doc_keys),
                gold_pages=tuple(finrag_gold_pages(r.corpus_ids, r.doc_keys)),
                gold_unit="page",
            )
            for r in raw
        ]
        inventory: tuple[str, ...] = ()
        return records, audit, inventory
    if dataset == "visdom":
        raw, audit, inventory = load_visdom_records(VISDOM_ROOT)
        records = [
            E2ERecord(
                uid=r.uid,
                dataset="visdom",
                lang="en",
                query_id=r.query_id,
                query=r.query,
                answer=r.answer,
                category=r.category,
                answer_type=r.answer_type,
                doc_keys=tuple(r.doc_keys),
                gold_pages=(),
                gold_unit="document",
            )
            for r in raw
        ]
        return records, audit, inventory
    raise ValueError(f"unknown dataset: {dataset}")
