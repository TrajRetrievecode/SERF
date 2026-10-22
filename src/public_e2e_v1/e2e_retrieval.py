#!/usr/bin/env python3
"""Shared page retrieval and evidence packaging.

Every method reads the same frozen page index, so any difference between
methods comes from which clients were called, not from retrieval quality.

Packaging follows the protocol of the Gold evaluation set: a called
client returns its single best-scoring page, the returned pages are ranked
globally, and the list is truncated to the page budget, which equals the call
budget.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from e2e_common import package_bytes
from e2e_corpus import CACHE, pages_for

CERTIFICATE_BYTES_PER_CALL = 512  # frozen protocol constant
CENTRAL_STORE = -1


@dataclass
class PageIndex:
    """The shared frozen retriever: CLIP ViT-B/32 visual + TF-IDF page text.

    Both channels are query-independent and cosine-normalised, so the blended
    score lives on a comparable scale.  Every method reads this one index.
    """

    dataset: str
    page_uids: list[str]
    doc_keys: np.ndarray  # int32 document ids
    doc_key_list: list[str]
    visual: np.ndarray  # (pages, dim) float32, L2 normalised
    query_visual: np.ndarray
    text: object  # scipy CSR (pages, vocab), L2 normalised
    query_text: object
    query_rows: dict[str, int]
    blend_weight: float
    uid_to_row: dict[str, int] = field(default_factory=dict)
    dense: np.ndarray | None = None
    query_dense: np.ndarray | None = None
    dense_weight: float = 0.0

    def attach_dense(self, dataset: str, dense_weight: float) -> "PageIndex":
        """Load the dense multilingual text channel and set its share.

        ``dense_weight`` splits the text side of the blend: the text score
        becomes ``dense_weight * dense + (1 - dense_weight) * tfidf``.  A
        weight of zero reproduces the original TF-IDF-only text channel.
        """
        self.dense_weight = float(dense_weight)
        if self.dense_weight <= 0.0 and self.dense is None:
            return self
        if self.dense is None:
            root = CACHE / f"{dataset}_dense_pages"
            page_rows = json.loads((root / "pages.json").read_text(encoding="utf-8"))
            if [r["page_uid"] for r in page_rows] != self.page_uids:
                raise RuntimeError(
                    "the dense index page order does not match the page corpus"
                )
            query_uids = [
                r["uid"]
                for r in json.loads((root / "queries.json").read_text(encoding="utf-8"))
            ]
            if [uid for uid, _ in sorted(self.query_rows.items(), key=lambda kv: kv[1])] != query_uids:
                raise RuntimeError(
                    "the dense index query order does not match the record set"
                )
            self.dense = np.load(root / "page_dense.npy").astype(np.float32)
            self.query_dense = np.load(root / "query_dense.npy").astype(np.float32)
        object.__setattr__(self, "_score_cache", {})
        return self

    @classmethod
    def load(cls, dataset: str, blend_weight: float) -> "PageIndex":
        from scipy import sparse

        root = CACHE / f"{dataset}_clip_pages"
        page_rows = json.loads((root / "pages.json").read_text(encoding="utf-8"))
        visual = np.load(root / "page_visual.npy").astype(np.float32)
        query_visual = np.load(root / "query_visual.npy").astype(np.float32)
        text = sparse.load_npz(root / "page_text.npz").tocsr()
        query_text = sparse.load_npz(root / "query_text.npz").tocsr()
        query_uids = [
            row["uid"]
            for row in json.loads((root / "queries.json").read_text(encoding="utf-8"))
        ]
        if visual.shape[0] != len(page_rows) or text.shape[0] != len(page_rows):
            raise RuntimeError("page feature/page row mismatch")
        if query_visual.shape[0] != len(query_uids) or query_text.shape[0] != len(
            query_uids
        ):
            raise RuntimeError("query feature/query row mismatch")

        corpus_pages = pages_for(dataset)
        if [p.page_uid for p in corpus_pages] != [r["page_uid"] for r in page_rows]:
            raise RuntimeError(
                "the indexed page order no longer matches the page corpus; "
                "rebuild the index before running the experiment"
            )

        doc_key_list = sorted({row["doc_key"] for row in page_rows})
        doc_id = {key: idx for idx, key in enumerate(doc_key_list)}
        doc_keys = np.array(
            [doc_id[row["doc_key"]] for row in page_rows], dtype=np.int32
        )
        page_uids = [row["page_uid"] for row in page_rows]
        return cls(
            dataset=dataset,
            page_uids=page_uids,
            doc_keys=doc_keys,
            doc_key_list=doc_key_list,
            visual=visual,
            query_visual=query_visual,
            text=text,
            query_text=query_text,
            query_rows={uid: idx for idx, uid in enumerate(query_uids)},
            blend_weight=float(blend_weight),
            uid_to_row={uid: idx for idx, uid in enumerate(page_uids)},
        )

    # Scoring a query against the whole corpus once and then indexing rows is
    # far cheaper than slicing a 10-million-nonzero sparse matrix per call, so
    # the full score vector is cached per query under a small LRU.
    _CACHE_LIMIT = 512

    def _full_scores(self, query_uid: str) -> np.ndarray:
        cache = getattr(self, "_score_cache", None)
        if cache is None:
            cache = {}
            object.__setattr__(self, "_score_cache", cache)
        cache_key = (query_uid, self.blend_weight, self.dense_weight)
        hit = cache.get(cache_key)
        if hit is not None:
            return hit
        row = self.query_rows[query_uid]
        visual = self.visual @ self.query_visual[row]
        sparse_text = np.asarray((self.text @ self.query_text[row].T).todense()).ravel()
        if self.dense is not None and self.dense_weight > 0.0:
            dense_text = self.dense @ self.query_dense[row]
            text = (
                self.dense_weight * dense_text
                + (1.0 - self.dense_weight) * sparse_text
            )
        else:
            text = sparse_text
        weight = self.blend_weight
        scores = (weight * visual + (1.0 - weight) * text).astype(np.float32)
        if len(cache) >= self._CACHE_LIMIT:
            cache.pop(next(iter(cache)))
        cache[cache_key] = scores
        return scores

    def scores_for_rows(self, query_uid: str, rows: np.ndarray) -> np.ndarray:
        """Blended page score restricted to the given corpus rows."""
        if rows.size == 0:
            return np.empty(0, dtype=np.float32)
        return self._full_scores(query_uid)[rows]

    def channel_scores(self, query_uid: str, rows: np.ndarray) -> dict:
        row = self.query_rows[query_uid]
        return {
            "visual": (self.visual @ self.query_visual[row])[rows],
            "text": np.asarray(
                (self.text @ self.query_text[row].T).todense()
            ).ravel()[rows],
        }


class ClientPageView:
    """Row indices owned by each client under one federation assignment."""

    def __init__(
        self,
        index: PageIndex,
        doc_to_client: dict[str, int],
        num_clients: int,
        unit: str = "document",
    ):
        self.num_clients = num_clients
        if unit == "page":
            # ownership is keyed by page uid, so a document can be co-held
            self.page_client = np.array(
                [doc_to_client.get(uid, -1) for uid in index.page_uids],
                dtype=np.int32,
            )
        else:
            client_of_doc = np.full(len(index.doc_key_list), -1, dtype=np.int32)
            for position, key in enumerate(index.doc_key_list):
                client_of_doc[position] = doc_to_client.get(key, -1)
            self.page_client = client_of_doc[index.doc_keys]
        self.rows_by_client: dict[int, np.ndarray] = {}
        for client in range(num_clients):
            self.rows_by_client[client] = np.flatnonzero(self.page_client == client)
        self.unassigned_rows = np.flatnonzero(self.page_client == -1)
        self.all_rows = np.arange(len(index.page_uids), dtype=np.int64)


@dataclass
class EvidencePackage:
    ranked_page_uids: list[str]
    ranked_scores: list[float]
    evidence_page_uids: list[str]
    evidence_scores: list[float]
    per_client_page: dict[int, str]
    candidate_pages: int
    client_calls: int


def rank_candidates(
    index: PageIndex,
    view: ClientPageView,
    query_uid: str,
    clients: tuple[int, ...],
    top_n: int = 10,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Ranked candidate rows visible to the called clients."""
    if clients == (CENTRAL_STORE,):
        rows = view.all_rows
    else:
        parts = [view.rows_by_client[c] for c in clients if c in view.rows_by_client]
        rows = (
            np.concatenate(parts)
            if parts
            else np.empty(0, dtype=np.int64)
        )
    if rows.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32), 0
    scores = index.scores_for_rows(query_uid, rows)
    take = min(top_n, rows.size)
    order = np.argpartition(-scores, take - 1)[:take]
    order = order[np.argsort(-scores[order])]
    return rows[order], scores[order], int(rows.size)


def build_package(
    index: PageIndex,
    view: ClientPageView,
    query_uid: str,
    clients: tuple[int, ...],
    budget: int,
    pages_per_client: int = 1,
    rank_depth: int = 10,
) -> EvidencePackage:
    ranked_rows, ranked_scores, candidates = rank_candidates(
        index, view, query_uid, clients, top_n=max(rank_depth, budget)
    )
    ranked_uids = [index.page_uids[int(r)] for r in ranked_rows]

    if clients == (CENTRAL_STORE,):
        chosen_rows = list(ranked_rows[:budget])
        chosen_scores = [float(s) for s in ranked_scores[:budget]]
        per_client = {CENTRAL_STORE: index.page_uids[int(r)] for r in chosen_rows[:1]}
        return EvidencePackage(
            ranked_page_uids=ranked_uids,
            ranked_scores=[float(s) for s in ranked_scores],
            evidence_page_uids=[index.page_uids[int(r)] for r in chosen_rows],
            evidence_scores=chosen_scores,
            per_client_page=per_client,
            candidate_pages=candidates,
            client_calls=1,
        )

    # One best page per called client, then a global ranking truncated to the
    # page budget.  This is the frozen evidence-selection rule.
    picks: list[tuple[float, int, int]] = []
    for client in clients:
        rows = view.rows_by_client.get(client)
        if rows is None or rows.size == 0:
            continue
        scores = index.scores_for_rows(query_uid, rows)
        take = min(pages_per_client, rows.size)
        order = np.argpartition(-scores, take - 1)[:take]
        order = order[np.argsort(-scores[order])]
        for position in order:
            picks.append((float(scores[position]), int(rows[position]), int(client)))

    picks.sort(key=lambda item: -item[0])
    selected = picks[:budget]
    per_client: dict[int, str] = {}
    for score, row, client in picks:
        per_client.setdefault(client, index.page_uids[row])
    return EvidencePackage(
        ranked_page_uids=ranked_uids,
        ranked_scores=[float(s) for s in ranked_scores],
        evidence_page_uids=[index.page_uids[row] for _, row, _ in selected],
        evidence_scores=[score for score, _, _ in selected],
        per_client_page=per_client,
        candidate_pages=candidates,
        client_calls=len(clients),
    )


def oracle_package(
    index: PageIndex,
    query_uid: str,
    gold_page_uids: tuple[str, ...],
    gold_doc_keys: tuple[str, ...],
    budget: int,
) -> EvidencePackage:
    """Official relevant evidence handed to the generator directly."""
    rows: list[int] = []
    if gold_page_uids:
        rows = [index.uid_to_row[u] for u in gold_page_uids if u in index.uid_to_row]
    if not rows and gold_doc_keys:
        doc_ids = {
            index.doc_key_list.index(key)
            for key in gold_doc_keys
            if key in index.doc_key_list
        }
        mask = np.isin(index.doc_keys, list(doc_ids))
        candidate = np.flatnonzero(mask)
        if candidate.size:
            scores = index.scores_for_rows(query_uid, candidate)
            order = np.argsort(-scores)
            rows = [int(candidate[i]) for i in order[:budget]]
    if not rows:
        return EvidencePackage([], [], [], [], {}, 0, 0)
    row_array = np.asarray(rows, dtype=np.int64)
    scores = index.scores_for_rows(query_uid, row_array)
    order = np.argsort(-scores)
    ordered_rows = [int(row_array[i]) for i in order][:budget]
    ordered_scores = [float(scores[i]) for i in order][:budget]
    return EvidencePackage(
        ranked_page_uids=[index.page_uids[r] for r in ordered_rows],
        ranked_scores=ordered_scores,
        evidence_page_uids=[index.page_uids[r] for r in ordered_rows],
        evidence_scores=ordered_scores,
        per_client_page={},
        candidate_pages=len(rows),
        client_calls=0,
    )


def communication_cost(
    query: str,
    budget: int,
    package: EvidencePackage,
    delivered_page_uids: tuple[str, ...],
    page_text: dict[str, str],
    text_char_limit: int,
) -> dict:
    """Actual serialized request/response bytes plus the frozen accounting.

    A system that abstains after calling clients still pays for those calls and
    for the certificates it received; it just does not carry the page payloads
    forward, so ``delivered_page_uids`` is empty for it.
    """
    calls = package.client_calls
    request_bytes = package_bytes({"query": query, "page_budget": budget}) * max(
        calls, 0
    )
    delivered = set(delivered_page_uids)
    response_bytes = 0
    for client, page_uid in package.per_client_page.items():
        payload = {
            "client": client,
            "page_uid": page_uid,
            "certificate": {"score": 0.0, "available": True},
        }
        if page_uid in delivered:
            payload["text"] = page_text.get(page_uid, "")[:text_char_limit]
        response_bytes += package_bytes(payload)
    evidence_text_bytes = sum(
        len(page_text.get(uid, "")[:text_char_limit].encode("utf-8"))
        for uid in delivered_page_uids
    )
    certificate_bytes = CERTIFICATE_BYTES_PER_CALL * max(calls, 0)
    return {
        "client_calls": calls,
        "selected_pages": len(delivered_page_uids),
        "serialized_request_bytes": request_bytes,
        "serialized_response_bytes": response_bytes,
        "evidence_text_bytes": evidence_text_bytes,
        "certificate_bytes": certificate_bytes,
        "protocol_bytes": certificate_bytes + evidence_text_bytes,
    }
