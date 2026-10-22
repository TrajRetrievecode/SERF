#!/usr/bin/env python3
"""Shared-protocol routing adapters for the public page index.

The adapters use the routing algorithms of scripts/run_formal_gold_v1.py,
with client scores obtained from cached public page scores. Constants come
from the evaluation protocol. Each adapter returns called clients, selected
pages, an abstention decision and a routing trace.
"""

from __future__ import annotations

import numpy as np


def softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    finite = np.isfinite(values)
    output = np.zeros_like(values, dtype=np.float64)
    if not finite.any():
        return output
    scaled = np.where(finite, values, -np.inf) / temperature
    scaled = scaled - np.max(scaled[finite])
    exponentiated = np.where(finite, np.exp(scaled), 0.0)
    total = exponentiated.sum()
    if total <= 0:
        return output
    return exponentiated / total


class QueryViews:
    """Client-level and page-level scores for one query."""

    def __init__(
        self,
        full_page: np.ndarray,
        slot_page: np.ndarray,
        rows_by_client: dict[int, np.ndarray],
        num_clients: int,
        claim_query_weight: float,
        full_query_weight: float,
        temperature: float,
    ):
        self.full_page = full_page
        self.slot_page = slot_page  # (J, pages)
        self.rows_by_client = rows_by_client
        self.num_clients = num_clients

        num_slots = slot_page.shape[0]
        self.full_client = np.full(num_clients, -np.inf)
        self.slot_client = np.full((num_slots, num_clients), -np.inf)
        for client in range(num_clients):
            rows = rows_by_client.get(client)
            if rows is None or rows.size == 0:
                continue
            self.full_client[client] = float(full_page[rows].max())
            self.slot_client[:, client] = slot_page[:, rows].max(axis=1)

        self.combined = (
            claim_query_weight * self.slot_client
            + full_query_weight * self.full_client[None, :]
        )
        self.probabilities = np.vstack(
            [softmax(row, temperature) for row in self.combined]
        )
        self.evidence_score = np.maximum(full_page, slot_page.max(axis=0))
        self.temperature = temperature

    def best_page_row(self, client: int, score: np.ndarray) -> int | None:
        rows = self.rows_by_client.get(client)
        if rows is None or rows.size == 0:
            return None
        return int(rows[int(np.argmax(score[rows]))])

    def evidence_rows(
        self, clients, budget: int, score: np.ndarray | None = None
    ) -> list[int]:
        """One best page per called client, ranked globally, cut to budget."""
        score = self.evidence_score if score is None else score
        picks: list[tuple[float, int]] = []
        for client in clients:
            row = self.best_page_row(int(client), score)
            if row is None:
                continue
            picks.append((float(score[row]), row))
        picks.sort(key=lambda item: (-item[0], item[1]))
        return [row for _, row in picks[:budget]]


# --------------------------------------------------------------------------
# adapters
# --------------------------------------------------------------------------
def local_adapter(views, online, budget, protocol, origin_client, **_):
    selected = [origin_client] if online[origin_client] else []
    rows = views.evidence_rows(selected, budget)
    return {"clients": selected, "rows": rows, "abstain": not rows,
            "trace": {"fixed_prelabel_client": origin_client}}


def all_adapter(views, online, budget, protocol, **_):
    selected = [c for c in range(views.num_clients) if online[c]]
    rows = views.evidence_rows(selected, budget)
    return {"clients": selected, "rows": rows, "abstain": not rows,
            "trace": {"fusion": "one-shot global relevance"}}


def ragroute_adapter(views, online, budget, protocol, **_):
    order = np.argsort(-views.full_client)
    selected = [
        int(c) for c in order if online[c] and np.isfinite(views.full_client[c])
    ][:budget]
    rows = views.evidence_rows(selected, budget, views.full_page)
    return {"clients": selected, "rows": rows, "abstain": not rows,
            "trace": {"routing": "one-shot full-query source selection"}}


def universalrag_adapter(views, online, budget, protocol, **_):
    client_score = np.max(views.slot_client, axis=0)
    order = np.argsort(-client_score)
    selected = [
        int(c) for c in order if online[c] and np.isfinite(client_score[c])
    ][:budget]
    rows = views.evidence_rows(selected, budget)
    return {"clients": selected, "rows": rows, "abstain": not rows,
            "trace": {"routing": "modality-granularity claim-slot maximum"}}


def core_adapter(views, online, budget, protocol, page_client, visual, pool=64, **_):
    """One-shot multimodal MMR over a bounded candidate pool.

    The evaluation set scores every candidate page of the item.  The public
    corpus has tens of thousands of pages per query, so the pool is the top
    ``pool`` pages by evidence score among online clients; the truncation is
    recorded rather than silent.
    """
    lam = protocol["core_mmr"]["lambda_relevance"]
    eligible_mask = np.array(
        [online[c] if c >= 0 else False for c in page_client], dtype=bool
    )
    candidates = np.flatnonzero(eligible_mask)
    if candidates.size == 0:
        return {"clients": [], "rows": [], "abstain": True,
                "trace": {"coordination": "one-shot multimodal MMR", "pool": 0}}
    take = min(pool, candidates.size)
    top = candidates[
        np.argpartition(-views.evidence_score[candidates], take - 1)[:take]
    ]
    top = top[np.argsort(-views.evidence_score[top])]
    selected_rows: list[int] = []
    remaining = list(top)
    while remaining and len(selected_rows) < budget:
        best = None
        for row in remaining:
            if selected_rows:
                redundancy = float(
                    np.max(visual[selected_rows] @ visual[row])
                )
            else:
                redundancy = 0.0
            utility = lam * float(views.evidence_score[row]) - (1 - lam) * redundancy
            candidate = (utility, -int(row))
            if best is None or candidate > best[0]:
                best = (candidate, int(row))
        selected_rows.append(best[1])
        remaining.remove(best[1])
    # This adapter is charged every online client, since it reconciles evidence
    # across all sources before selecting.
    queried = [c for c in range(views.num_clients) if online[c]]
    return {"clients": queried, "rows": selected_rows, "abstain": not selected_rows,
            "trace": {"coordination": "one-shot multimodal MMR", "pool": int(take)}}


def s2g_adapter(views, online, budget, protocol, **_):
    selected: list[int] = []
    claim_best = np.full(views.slot_client.shape[0], -np.inf)
    steps = []
    for _ in range(budget):
        slot = int(np.argmin(claim_best))
        order = np.argsort(-views.slot_client[slot])
        client = next(
            (
                int(v)
                for v in order
                if online[v] and v not in selected
                and np.isfinite(views.slot_client[slot, v])
            ),
            None,
        )
        if client is None:
            break
        selected.append(client)
        claim_best = np.maximum(claim_best, views.slot_client[:, client])
        steps.append({"slot": slot, "client": client})
    rows = views.evidence_rows(selected, budget)
    return {"clients": selected, "rows": rows, "abstain": not rows,
            "trace": {"gap_steps": steps}}


def _critical_missing(views, online, protocol):
    """The availability test, factored out so a baseline can also be given it."""
    probabilities = views.probabilities
    critical = []
    for slot in range(probabilities.shape[0]):
        order = np.argsort(-views.combined[slot])
        global_client = next(
            (int(c) for c in order if np.isfinite(views.combined[slot, c])), None
        )
        online_client = next(
            (
                int(c)
                for c in order
                if online[c] and np.isfinite(views.combined[slot, c])
            ),
            None,
        )
        if (
            global_client is not None
            and online_client is not None
            and not online[global_client]
        ):
            raw_margin = float(
                views.combined[slot, global_client]
                - views.combined[slot, online_client]
            )
            ratio = float(
                probabilities[slot, global_client]
                / max(probabilities[slot, online_client], 1e-9)
            )
            if (
                raw_margin >= protocol["fedresolve"]["missingness_raw_margin"]
                and ratio >= protocol["fedresolve"]["missingness_probability_ratio"]
            ):
                critical.append(
                    {
                        "slot": slot,
                        "offline_client": global_client,
                        "best_online_client": online_client,
                        "raw_margin": raw_margin,
                        "probability_ratio": ratio,
                    }
                )
    return critical


def s2g_avail_adapter(views, online, budget, protocol, **kwargs):
    """Run the S2G-RAG adaptation with the shared availability test."""
    result = s2g_adapter(views, online, budget, protocol, **kwargs)
    critical = _critical_missing(views, online, protocol)
    result = dict(result)
    result["abstain"] = bool(critical) or result["abstain"]
    result["trace"] = {**result["trace"], "critical_missing_slots": critical}
    return result


def fedresolve_adapter(views, online, budget, protocol, **_):
    probabilities = views.probabilities
    selected: list[int] = []
    residual = np.ones(probabilities.shape[0], dtype=np.float64)
    steps = []
    for _ in range(budget):
        best = None
        for client in range(views.num_clients):
            if (
                not online[client]
                or client in selected
                or not np.isfinite(views.full_client[client])
            ):
                continue
            marginal = float(np.sum(residual * probabilities[:, client]))
            candidate = (marginal, -float(client), client)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
        if best is None or best[0] < protocol["fedresolve"]["minimum_marginal_gain"]:
            break
        client = best[2]
        selected.append(client)
        residual *= 1.0 - probabilities[:, client]
        steps.append(
            {
                "client": client,
                "marginal_expected_claim_mass": best[0],
                "remaining_claim_mass": float(residual.sum()),
            }
        )

    critical = _critical_missing(views, online, protocol)
    rows = views.evidence_rows(selected, budget)
    return {
        "clients": selected,
        "rows": rows,
        "abstain": bool(critical) or not rows,
        "trace": {
            "steps": steps,
            "critical_missing_slots": critical,
            "certificate_interface": "query-slot x client probability",
        },
    }


def centralized_adapter(views, online, budget, protocol, page_client, **_):
    mask = np.array(
        [online[c] if c >= 0 else False for c in page_client], dtype=bool
    )
    candidates = np.flatnonzero(mask)
    if candidates.size == 0:
        return {"clients": [], "rows": [], "abstain": True,
                "trace": {"access": "centralized global index"}}
    take = min(budget, candidates.size)
    top = candidates[
        np.argpartition(-views.evidence_score[candidates], take - 1)[:take]
    ]
    rows = list(top[np.argsort(-views.evidence_score[top])])
    queried = sorted({int(page_client[r]) for r in rows})
    return {"clients": queried, "rows": [int(r) for r in rows], "abstain": not rows,
            "trace": {"access": "centralized global index", "calls": 1}}


def oracle_client_adapter(views, online, budget, protocol, gold_clients, **_):
    feasible = [c for c in gold_clients if online[c]]
    selected = feasible[:budget]
    rows = views.evidence_rows(selected, budget)
    return {"clients": selected, "rows": rows,
            "abstain": not feasible,
            "trace": {"oracle_input": "gold client set only"}}


def oracle_evidence_adapter(views, online, budget, protocol, gold_rows, page_client, **_):
    rows = [r for r in gold_rows if page_client[r] < 0 or online[page_client[r]]]
    rows = sorted(rows, key=lambda r: -float(views.evidence_score[r]))[:budget]
    return {"clients": [], "rows": rows, "abstain": not rows,
            "trace": {"oracle_input": "official relevant pages"}}


# --------------------------------------------------------------------------
# ablations of the repository method
#
# Each variant removes exactly one module and leaves everything else at the
# frozen settings, so the call reduction and the abstention gain can be
# attributed rather than assumed.  ``without slot decomposition`` is applied
# by the driver, which passes a single-slot view; the other three are here.
# --------------------------------------------------------------------------
def fedresolve_no_residual(views, online, budget, protocol, **_):
    """Greedy on the raw posterior: no multiplicative coverage update."""
    probabilities = views.probabilities
    score = probabilities.sum(axis=0)
    selected: list[int] = []
    for _ in range(budget):
        best = None
        for client in range(views.num_clients):
            if (
                not online[client]
                or client in selected
                or not np.isfinite(views.full_client[client])
            ):
                continue
            candidate = (float(score[client]), -float(client), client)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
        if best is None or best[0] < protocol["fedresolve"]["minimum_marginal_gain"]:
            break
        selected.append(best[2])
    critical = _critical_missing(views, online, protocol)
    rows = views.evidence_rows(selected, budget)
    return {"clients": selected, "rows": rows,
            "abstain": bool(critical) or not rows,
            "trace": {"ablation": "without residual coverage update"}}


def fedresolve_no_early_stop(views, online, budget, protocol, **_):
    """Residual coverage kept, but the budget is always spent in full."""
    probabilities = views.probabilities
    selected: list[int] = []
    residual = np.ones(probabilities.shape[0], dtype=np.float64)
    for _ in range(budget):
        best = None
        for client in range(views.num_clients):
            if (
                not online[client]
                or client in selected
                or not np.isfinite(views.full_client[client])
            ):
                continue
            marginal = float(np.sum(residual * probabilities[:, client]))
            candidate = (marginal, -float(client), client)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
        if best is None:
            break
        selected.append(best[2])
        residual *= 1.0 - probabilities[:, best[2]]
    critical = _critical_missing(views, online, protocol)
    rows = views.evidence_rows(selected, budget)
    return {"clients": selected, "rows": rows,
            "abstain": bool(critical) or not rows,
            "trace": {"ablation": "without marginal-gain early stop"}}


def fedresolve_no_availability(views, online, budget, protocol, **kwargs):
    """The full router, with the availability test switched off."""
    result = dict(fedresolve_adapter(views, online, budget, protocol, **kwargs))
    result["abstain"] = not result["rows"]
    result["trace"] = {"ablation": "without availability test"}
    return result


ABLATIONS = {
    "FedResolve-noResidual": fedresolve_no_residual,
    "FedResolve-noEarlyStop": fedresolve_no_early_stop,
    "FedResolve-noAvailability": fedresolve_no_availability,
    "FedResolve-noSlots": fedresolve_adapter,  # driver supplies a single-slot view
}


ADAPTERS = {
    "Local": local_adapter,
    "All": all_adapter,
    "Centralized": centralized_adapter,
    "RAGRoute": ragroute_adapter,
    "UniversalRAG": universalrag_adapter,
    "CoRe-MMRAG": core_adapter,
    "S2G-RAG": s2g_adapter,
    "S2G-RAG+avail": s2g_avail_adapter,
    "FedResolve": fedresolve_adapter,
    "Oracle-client": oracle_client_adapter,
    "Oracle-evidence": oracle_evidence_adapter,
}
ADAPTERS.update(ABLATIONS)
