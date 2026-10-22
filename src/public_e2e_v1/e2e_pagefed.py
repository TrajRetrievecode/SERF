#!/usr/bin/env python3
"""A federation in which the pages of a document are spread over several clients.

The document picks a contiguous block of clients by a hash and each page picks one of them. The
split is deterministic, query-independent and derived from page identifiers only. Three co-holders
is the setting used here.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from fedresolve_core import Federation, family_key, stable_u64


def page_client(page_uid: str, doc_key: str, num_clients: int, seed: int, replicas: int) -> int:
    """Deterministic co-holder assignment for one page.

    The document picks a contiguous block of ``replicas`` clients; the page
    picks one of them.  Nothing about the query or the qrels is consulted.
    """
    base = stable_u64(doc_key, seed) % num_clients
    offset = stable_u64(page_uid, seed + 13) % replicas
    return (base + offset) % num_clients


def build_page_federation(
    page_uids: Sequence[str],
    num_clients: int,
    seed: int,
    replicas: int = 3,
) -> Federation:
    """A Federation whose unit of ownership is the page, not the document.

    The returned object keys ``doc_to_client`` by page uid, so every consumer
    that already speaks the Federation interface keeps working; the records
    handed to it must carry page uids in ``doc_keys`` (see
    :func:`as_page_records`).
    """
    assignment: dict[str, int] = {}
    for page_uid in page_uids:
        doc_key = page_uid.rsplit("#p", 1)[0]
        assignment[page_uid] = page_client(
            page_uid, doc_key, num_clients, seed, replicas
        )
    client_documents = {
        client: tuple(
            sorted(uid for uid, owner in assignment.items() if owner == client)
        )
        for client in range(num_clients)
    }
    loads = {client: len(pages) for client, pages in client_documents.items()}
    qualified = {
        f"{family_key(uid.rsplit('#p', 1)[0])}\0{uid}": owner
        for uid, owner in assignment.items()
    }
    return Federation(
        num_clients=num_clients,
        doc_to_client=assignment,
        family_to_client=qualified,
        client_documents=client_documents,
        family_loads=loads,
    )


def as_page_records(records):
    """Restate each record so its ownership unit is its gold pages."""
    return [replace(r, doc_keys=tuple(dict.fromkeys(r.gold_pages))) for r in records]


def complementary_splits(records, seed: int, dev_fraction: float = 0.15) -> dict:
    """Put every multi-gold-page query in test; split the rest into train/dev.

    The rule uses the number of official gold pages, which is a label, so it
    defines the evaluation subgroup and nothing else: no threshold, weight or
    policy is ever chosen on these records.  Routers therefore train only on
    single-page queries and are evaluated on queries whose evidence can span
    clients.
    """
    train, dev, test = [], [], []
    for record in records:
        if len(set(record.gold_pages)) >= 2:
            test.append(record)
            continue
        bucket = stable_u64(record.uid, seed) % 100
        if bucket < dev_fraction * 100:
            dev.append(record)
        else:
            train.append(record)
    return {"train": train, "dev": dev, "test": test}


def cross_client_flags(records, federation: Federation) -> dict[str, bool]:
    """Which test records genuinely need more than one client."""
    return {r.uid: len(federation.gold_clients(r)) > 1 for r in records}
