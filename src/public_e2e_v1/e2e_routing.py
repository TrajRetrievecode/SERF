#!/usr/bin/env python3
"""Client-selection layer for the public end-to-end track.

The routers, certifier and negotiation policy are the frozen components in
``src/fedresolve_core.py``; this module only turns them into an explicit,
per-record client selection so the downstream retrieval and generation stages
can consume them.  Method semantics are those of ``src/fedresolve_core.py``.

Budget handling: the call budget ``B`` is a protocol-level cap applied
identically to every routed method.  ``Local`` (one call by definition),
``All`` (broadcast by definition), ``Centralized`` (no federation) and
``Oracle-client`` (one call to the qrel client) keep their definitional call
counts, exactly as on the Gold evaluation set.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

from fedresolve_core import (
    CapabilitySketchRouter,
    Federation,
    MissingnessAwareNegotiator,
    NegotiationPolicy,
    QueryRecord,
    ResidualCertifier,
    SupervisedClientRouter,
    build_federation,
    ranking_metrics,
    split_records,
    stable_u64,
    tune_policy,
)

# Method inventory, in the order the paper reports them.
FRAMEWORK_DIAGNOSTIC = ("Local", "All", "Centralized")
PROTOCOL_ADAPTERS = (
    "RAGRoute",
    "UniversalRAG",
    "CoRe-MMRAG",
    "S2G-RAG",
    "S2G-RAG+avail",
)
REPOSITORY_METHOD = ("FedResolve",)
ORACLES = ("Oracle-client", "Oracle-evidence")
ALL_METHODS = FRAMEWORK_DIAGNOSTIC + PROTOCOL_ADAPTERS + REPOSITORY_METHOD + ORACLES

METHOD_LEVEL = {
    **{name: "framework_diagnostic" for name in FRAMEWORK_DIAGNOSTIC},
    **{name: "protocol_level_adapter" for name in PROTOCOL_ADAPTERS},
    "FedResolve": "repository_method",
    "Oracle-client": "framework_oracle",
    "Oracle-evidence": "framework_oracle",
}

METHOD_NOTES = {
    "Local": "Calls one client fixed by a hash of the query identifier.",
    "All": "Calls every online client.",
    "Centralized": (
        "Ranks the pages of every online client in one index on the server "
        "and takes no part in the call protocol."
    ),
    "RAGRoute": (
        "Ranks the online clients by the best similarity of their pages to "
        "the whole query and calls the top budget."
    ),
    "UniversalRAG": (
        "Ranks the online clients by the best similarity of their pages to "
        "any sub-question and calls the top budget."
    ),
    "CoRe-MMRAG": (
        "Calls every online client and selects pages by maximal marginal "
        "relevance over the evidence score and the image similarity to the "
        "pages already selected, among the 64 pages with the highest "
        "evidence score."
    ),
    "S2G-RAG": (
        "Calls in each round the best-scoring uncalled online client for the "
        "least covered sub-question, without the availability test."
    ),
    "S2G-RAG+avail": (
        "The same adaptation with the availability test of the protocol's "
        "fixed margin and ratio."
    ),
    "FedResolve": (
        "The one-page variant: sub-question routing with residual coverage, "
        "early stop and the availability test, one page per called client."
    ),
    "Oracle-client": (
        "Calls the clients that hold the relevant documents of the official "
        "labels under the simulated federation."
    ),
    "Oracle-evidence": (
        "Hands the official relevant pages or documents to the generator; an "
        "evidence upper bound, not a retrieval method."
    ),
}


@dataclass(frozen=True)
class Selection:
    method: str
    uid: str
    clients: tuple[int, ...]
    abstain: bool
    reason: str
    offline_probability_peak: float
    offline_probability_mass: float


def fixed_policy(max_calls: int, accept: float = 0.0) -> NegotiationPolicy:
    return NegotiationPolicy(
        max_calls=max_calls,
        accept_threshold=accept,
        stop_mass=2.0,  # force max_calls unless fewer clients are online
        offline_mass_threshold=2.0,
    )


DISABLED_AVAILABILITY_TEST = 2.0  # a probability peak can never reach it


def with_max_calls(
    policy: NegotiationPolicy, max_calls: int, availability_test: bool = True
) -> NegotiationPolicy:
    return NegotiationPolicy(
        max_calls=max_calls,
        accept_threshold=policy.accept_threshold,
        stop_mass=policy.stop_mass,
        offline_mass_threshold=(
            policy.offline_mass_threshold
            if availability_test
            else DISABLED_AVAILABILITY_TEST
        ),
    )


def blend(a: np.ndarray, b: np.ndarray, weight: float) -> np.ndarray:
    output = weight * a + (1.0 - weight) * b
    output /= output.sum(axis=1, keepdims=True)
    return output


def tune_blend(
    supervised: np.ndarray,
    sketch: np.ndarray,
    records: Sequence[QueryRecord],
    federation: Federation,
) -> float:
    best = (-1.0, 0.5)
    for weight in np.linspace(0.0, 1.0, 21):
        probs = blend(supervised, sketch, float(weight))
        score = ranking_metrics(probs, records, federation)["mrr"]
        if score > best[0]:
            best = (score, float(weight))
    return best[1]


def origin_client(uid: str, num_clients: int) -> int:
    """Stable, label-free origin client for the Local diagnostic."""
    return (
        sum((idx + 1) * byte for idx, byte in enumerate(uid.encode("utf-8")))
        % num_clients
    )


class RoutingLayer:
    """Fit every router once per (dataset, clients, track) configuration."""

    def __init__(
        self,
        records: Sequence[QueryRecord],
        inventory_documents: Sequence[str],
        num_clients: int,
        track: str,
        seed: int,
        federation=None,
        splits=None,
    ):
        self.num_clients = num_clients
        self.track = track
        self.seed = seed
        # A caller may supply a federation and a split when the track is not
        # the standard document-atomic, hash-split one.
        self.federation = (
            federation
            if federation is not None
            else build_federation(
                inventory_documents, num_clients=num_clients, seed=seed
            )
        )
        self.splits = (
            splits
            if splits is not None
            else split_records(records, track=track, seed=seed + 101)
        )
        if min(map(len, self.splits.values())) == 0:
            raise RuntimeError(
                f"empty split for {track}/{num_clients}: "
                + str({k: len(v) for k, v in self.splits.items()})
            )
        self.test = self.splits["test"]

        supervised_router = SupervisedClientRouter(num_clients, seed)
        supervised_router.fit(self.splits["train"], self.federation)
        sup_dev = supervised_router.predict_proba(self.splits["dev"])
        sup_test = supervised_router.predict_proba(self.test)

        sketch_router = CapabilitySketchRouter(num_clients)
        sketch_router.fit(self.federation, self.splits["train"])
        sketch_dev = sketch_router.score(self.splits["dev"])
        sketch_test = sketch_router.score(self.test)

        self.blend_weight = tune_blend(
            sup_dev, sketch_dev, self.splits["dev"], self.federation
        )
        blended_dev = blend(sup_dev, sketch_dev, self.blend_weight)
        blended_test = blend(sup_test, sketch_test, self.blend_weight)

        certifier = ResidualCertifier(num_clients, seed + 7)
        certifier.fit(
            self.splits["dev"], self.federation, supervised=blended_dev, sketch=sketch_dev
        )
        global_dev = certifier.predict_global(blended_dev)
        global_test = certifier.predict_global(blended_test)
        self.certifier = certifier

        self.fedresolve_policy = tune_policy(
            global_dev, self.splits["dev"], self.federation
        )
        self.s2g_policy = tune_policy(
            blended_dev, self.splits["dev"], self.federation
        )

        size = len(self.test)
        gold_probs = np.zeros((size, num_clients), dtype=np.float64)
        local_probs = np.zeros_like(gold_probs)
        for row, record in enumerate(self.test):
            gold = self.federation.gold_clients(record)
            gold_probs[row, list(gold)] = 1.0 / len(gold)
            local_probs[row, origin_client(record.uid, num_clients)] = 1.0

        self.probabilities = {
            "Local": local_probs,
            "All": np.full((size, num_clients), 1.0 / num_clients),
            "RAGRoute": sup_test,
            "UniversalRAG": sup_test,
            "CoRe-MMRAG": sketch_test,
            "S2G-RAG": blended_test,
            "S2G-RAG+avail": blended_test,
            "FedResolve": global_test,
            "Oracle-client": gold_probs,
        }

    # ------------------------------------------------------------------
    def policy_for(self, method: str, budget: int) -> NegotiationPolicy:
        if method == "Local":
            return fixed_policy(1)
        if method == "All":
            return fixed_policy(self.num_clients)
        if method == "Oracle-client":
            return fixed_policy(1)
        if method == "S2G-RAG":
            # S2G-RAG adaptation: gap-driven
            # stopping is kept, the availability test is not granted.
            return with_max_calls(self.s2g_policy, budget, availability_test=False)
        if method == "S2G-RAG+avail":
            return with_max_calls(self.s2g_policy, budget, availability_test=True)
        if method == "FedResolve":
            return with_max_calls(self.fedresolve_policy, budget)
        if method in ("RAGRoute", "UniversalRAG", "CoRe-MMRAG"):
            return fixed_policy(budget)
        raise ValueError(f"{method} has no routed policy")

    def online_mask(self, record: QueryRecord, scenario: str, seed: int) -> np.ndarray:
        """Availability of each client under a named participation scenario.

        ``full``            every client is online;
        ``qrel_missing``    every client holding an official relevant document
                            is offline, which is the qrel-induced unanswerable
                            proxy;
        ``partial_<p>``     an independent per-record Bernoulli draw.
        """
        online = np.ones(self.num_clients, dtype=bool)
        if scenario == "full":
            return online
        if scenario == "qrel_missing":
            for client in self.federation.gold_clients(record):
                online[client] = False
            return online
        if scenario.startswith("partial_"):
            participation = float(scenario.split("_", 1)[1])
            rng = np.random.default_rng(stable_u64(record.uid, seed))
            return rng.random(self.num_clients) < participation
        raise ValueError(f"unknown scenario: {scenario}")

    def select(
        self, method: str, budget: int, scenario: str, seed: int
    ) -> list[Selection]:
        if method == "Centralized":
            return [
                Selection(method, r.uid, (-1,), False, "centralized_single_store", 0.0, 0.0)
                for r in self.test
            ]
        if method == "Oracle-evidence":
            return [
                Selection(method, r.uid, (), False, "oracle_evidence", 0.0, 0.0)
                for r in self.test
            ]
        probabilities = self.probabilities[method]
        negotiator = MissingnessAwareNegotiator(self.policy_for(method, budget))
        output: list[Selection] = []
        for row, record in enumerate(self.test):
            online = self.online_mask(record, scenario, seed)
            clients, abstain, info = negotiator.select(probabilities[row], online=online)
            output.append(
                Selection(
                    method=method,
                    uid=record.uid,
                    clients=tuple(int(c) for c in clients),
                    abstain=bool(abstain),
                    reason=str(info.get("stop_reason", "")),
                    offline_probability_peak=float(
                        info.get("offline_probability_peak", 0.0)
                    ),
                    offline_probability_mass=float(
                        info.get("offline_probability_mass", 0.0)
                    ),
                )
            )
        return output

    def configuration(self) -> dict:
        return {
            "track": self.track,
            "num_clients": self.num_clients,
            "seed": self.seed,
            "split_counts": {k: len(v) for k, v in self.splits.items()},
            "supervised_blend_weight_selected_on_dev": self.blend_weight,
            "global_temperature": self.certifier.temperature,
            "client_logit_residuals": self.certifier.residuals.tolist(),
            "fedresolve_policy_selected_on_dev": asdict(self.fedresolve_policy),
            "s2g_policy_selected_on_dev": asdict(self.s2g_policy),
            "federation": {
                "status": "QUASI_NATURAL_SIMULATION_NOT_OFFICIAL_FEDERATION",
                "query_independent": True,
                "family_atomic": True,
                "num_documents": len(self.federation.doc_to_client),
                "num_document_families": len(self.federation.family_to_client),
                "client_document_loads": self.federation.family_loads,
            },
        }
