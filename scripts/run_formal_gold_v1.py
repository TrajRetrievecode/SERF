#!/usr/bin/env python3
"""Run the Gold v1 unified-protocol evaluation.

Non-oracle selection functions receive only frozen task inputs, image/query
embeddings, online masks, and the pre-registered protocol.  Gold annotations
are passed exclusively to evaluation and explicitly named oracle adapters.
"""

from __future__ import annotations

import argparse
import os
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np


ROOT_DEFAULT = Path(os.environ.get("SERF_ROOT", str(Path(__file__).resolve().parents[1])))
METHODS = (
    "Local",
    "All",
    "RAGRoute",
    "UniversalRAG",
    "CoRe-MMRAG",
    "S2G-RAG",
    "FedResolve",
    "Centralized",
    "Oracle client set",
    "Oracle claim evidence",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    output = np.zeros_like(values)
    if not finite.any():
        return output
    shifted = values[finite] / temperature
    shifted -= shifted.max()
    output[finite] = np.exp(shifted)
    output /= output.sum()
    return output


def stable_client(canonical_id: str, seed: int, clients: int = 8) -> int:
    digest = hashlib.sha256(f"{canonical_id}|{seed}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % clients


def ece(confidence: list[float], correct: list[float], bins: int = 10) -> float:
    if not confidence:
        return 0.0
    conf = np.asarray(confidence)
    corr = np.asarray(correct)
    result = 0.0
    for lower, upper in zip(
        np.linspace(0, 1, bins + 1)[:-1],
        np.linspace(0, 1, bins + 1)[1:],
    ):
        mask = (conf >= lower) & (conf <= upper if upper == 1 else conf < upper)
        if mask.any():
            result += float(mask.mean()) * abs(
                float(conf[mask].mean()) - float(corr[mask].mean())
            )
    return result


def bootstrap_mean(
    values: list[float], seed: int, samples: int = 2000
) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(samples, len(array)))
    means = array[indices].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }


def best_page_for_client(
    inputs: dict[str, Any], client: int, score: np.ndarray
) -> str | None:
    rows = [
        index
        for index, value in enumerate(inputs["candidate_clients"])
        if value == client
    ]
    if not rows:
        return None
    row = max(rows, key=lambda index: (float(score[index]), -index))
    return inputs["candidate_ids"][row]


def evidence_for_clients(
    inputs: dict[str, Any],
    clients: list[int],
    budget: int,
    score: np.ndarray,
) -> list[str]:
    candidates: list[tuple[float, str]] = []
    for client in clients:
        page = best_page_for_client(inputs, client, score)
        if page is None:
            continue
        index = inputs["candidate_ids"].index(page)
        candidates.append((float(score[index]), page))
    candidates.sort(key=lambda row: (-row[0], row[1]))
    return [page for _, page in candidates[:budget]]


def score_views(
    inputs: dict[str, Any], protocol: dict[str, Any]
) -> dict[str, Any]:
    full = inputs["full_scores"]
    slots = inputs["slot_scores"]
    client_ids = protocol["federation"]["client_ids"]
    full_client = np.full(len(client_ids), -np.inf)
    slot_client = np.full((len(slots), len(client_ids)), -np.inf)
    for row, client in enumerate(inputs["candidate_clients"]):
        full_client[client] = max(full_client[client], full[row])
        for slot in range(len(slots)):
            slot_client[slot, client] = max(
                slot_client[slot, client], slots[slot, row]
            )
    combined = (
        protocol["fedresolve"]["claim_query_weight"] * slot_client
        + protocol["fedresolve"]["full_query_weight"] * full_client[None, :]
    )
    probabilities = np.vstack(
        [
            softmax(row, protocol["fedresolve"]["global_temperature"])
            for row in combined
        ]
    )
    evidence_score = np.maximum(full, slots.max(axis=0))
    return {
        "full_client": full_client,
        "slot_client": slot_client,
        "combined": combined,
        "probabilities": probabilities,
        "evidence_score": evidence_score,
    }


def local_adapter(
    inputs: dict[str, Any],
    views: dict[str, Any],
    online: np.ndarray,
    budget: int,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    client = stable_client(inputs["canonical_id"], protocol["seed"])
    selected = [client] if online[client] else []
    evidence = evidence_for_clients(
        inputs, selected, budget, views["evidence_score"]
    )
    return {
        "queried_clients": selected,
        "evidence_ids": evidence,
        "abstain": not evidence,
        "confidence": float(
            softmax(
                views["full_client"],
                protocol["fedresolve"]["global_temperature"],
            )[client]
        ),
        "client_calls": len(selected),
        "trace": {"fixed_prelabel_client": client},
    }


def all_adapter(
    inputs: dict[str, Any],
    views: dict[str, Any],
    online: np.ndarray,
    budget: int,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    selected = [client for client in range(len(online)) if online[client]]
    evidence = evidence_for_clients(
        inputs, selected, budget, views["evidence_score"]
    )
    return {
        "queried_clients": selected,
        "evidence_ids": evidence,
        "abstain": not evidence,
        "confidence": 1.0 if evidence else 0.0,
        "client_calls": len(selected),
        "trace": {"fusion": "one-shot global relevance"},
    }


def ragroute_adapter(
    inputs: dict[str, Any],
    views: dict[str, Any],
    online: np.ndarray,
    budget: int,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    order = np.argsort(-views["full_client"])
    selected = [
        int(client)
        for client in order
        if online[client] and np.isfinite(views["full_client"][client])
    ][:budget]
    probability = softmax(
        views["full_client"], protocol["fedresolve"]["global_temperature"]
    )
    evidence = evidence_for_clients(
        inputs, selected, budget, inputs["full_scores"]
    )
    return {
        "queried_clients": selected,
        "evidence_ids": evidence,
        "abstain": not evidence,
        "confidence": float(probability[selected].sum()) if selected else 0.0,
        "client_calls": len(selected),
        "trace": {"routing": "one-shot full-query source selection"},
    }


def universalrag_adapter(
    inputs: dict[str, Any],
    views: dict[str, Any],
    online: np.ndarray,
    budget: int,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    client_score = np.max(views["slot_client"], axis=0)
    order = np.argsort(-client_score)
    selected = [
        int(client)
        for client in order
        if online[client] and np.isfinite(client_score[client])
    ][:budget]
    probability = softmax(
        client_score, protocol["fedresolve"]["global_temperature"]
    )
    evidence = evidence_for_clients(
        inputs, selected, budget, views["evidence_score"]
    )
    return {
        "queried_clients": selected,
        "evidence_ids": evidence,
        "abstain": not evidence,
        "confidence": float(probability[selected].sum()) if selected else 0.0,
        "client_calls": len(selected),
        "trace": {"routing": "modality-granularity claim-slot maximum"},
    }


def core_adapter(
    inputs: dict[str, Any],
    views: dict[str, Any],
    online: np.ndarray,
    budget: int,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    eligible = [
        index
        for index, client in enumerate(inputs["candidate_clients"])
        if online[client]
    ]
    selected_rows: list[int] = []
    lam = protocol["core_mmr"]["lambda_relevance"]
    while eligible and len(selected_rows) < budget:
        best: tuple[float, int] | None = None
        for row in eligible:
            redundancy = (
                max(inputs["image_similarity"][row, other] for other in selected_rows)
                if selected_rows
                else 0.0
            )
            utility = lam * views["evidence_score"][row] - (1 - lam) * redundancy
            candidate = (float(utility), -row)
            if best is None or candidate > (best[0], -best[1]):
                best = (float(utility), row)
        assert best is not None
        selected_rows.append(best[1])
        eligible.remove(best[1])
    evidence = [inputs["candidate_ids"][row] for row in selected_rows]
    queried = [client for client in range(len(online)) if online[client]]
    return {
        "queried_clients": queried,
        "evidence_ids": evidence,
        "abstain": not evidence,
        "confidence": float(
            np.mean([views["evidence_score"][row] for row in selected_rows])
        )
        if selected_rows
        else 0.0,
        "client_calls": len(queried),
        "trace": {"coordination": "one-shot multimodal MMR"},
    }


def s2g_adapter(
    inputs: dict[str, Any],
    views: dict[str, Any],
    online: np.ndarray,
    budget: int,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    selected: list[int] = []
    claim_best = np.full(len(views["slot_client"]), -np.inf)
    steps = []
    for _ in range(budget):
        slot = int(np.argmin(claim_best))
        order = np.argsort(-views["slot_client"][slot])
        client = next(
            (
                int(value)
                for value in order
                if online[value]
                and value not in selected
                and np.isfinite(views["slot_client"][slot, value])
            ),
            None,
        )
        if client is None:
            break
        selected.append(client)
        claim_best = np.maximum(claim_best, views["slot_client"][:, client])
        steps.append({"slot": slot, "client": client})
    evidence = evidence_for_clients(
        inputs, selected, budget, views["evidence_score"]
    )
    confidence = float(
        np.mean(
            [
                softmax(
                    views["slot_client"][slot],
                    protocol["fedresolve"]["global_temperature"],
                )[selected].sum()
                for slot in range(len(views["slot_client"]))
            ]
        )
    ) if selected else 0.0
    return {
        "queried_clients": selected,
        "evidence_ids": evidence,
        "abstain": not evidence,
        "confidence": confidence,
        "client_calls": len(selected),
        "trace": {"gap_steps": steps},
    }


def fedresolve_adapter(
    inputs: dict[str, Any],
    views: dict[str, Any],
    online: np.ndarray,
    budget: int,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    probabilities = views["probabilities"]
    selected: list[int] = []
    residual = np.ones(len(probabilities), dtype=np.float64)
    steps = []
    for _ in range(budget):
        best: tuple[float, float, int] | None = None
        for client in range(len(online)):
            if (
                not online[client]
                or client in selected
                or not np.isfinite(views["full_client"][client])
            ):
                continue
            marginal = float(np.sum(residual * probabilities[:, client]))
            candidate = (marginal, -float(client), client)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
        if (
            best is None
            or best[0] < protocol["fedresolve"]["minimum_marginal_gain"]
        ):
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

    critical_missing_slots = []
    for slot in range(len(probabilities)):
        global_order = np.argsort(-views["combined"][slot])
        global_client = next(
            (
                int(client)
                for client in global_order
                if np.isfinite(views["combined"][slot, client])
            ),
            None,
        )
        online_client = next(
            (
                int(client)
                for client in global_order
                if online[client] and np.isfinite(views["combined"][slot, client])
            ),
            None,
        )
        if (
            global_client is not None
            and online_client is not None
            and not online[global_client]
        ):
            raw_margin = (
                views["combined"][slot, global_client]
                - views["combined"][slot, online_client]
            )
            ratio = probabilities[slot, global_client] / max(
                probabilities[slot, online_client], 1e-9
            )
            if (
                raw_margin >= protocol["fedresolve"]["missingness_raw_margin"]
                and ratio >= protocol["fedresolve"]["missingness_probability_ratio"]
            ):
                critical_missing_slots.append(
                    {
                        "slot": slot,
                        "offline_client": global_client,
                        "best_online_client": online_client,
                        "raw_margin": float(raw_margin),
                        "probability_ratio": float(ratio),
                    }
                )
    evidence = evidence_for_clients(
        inputs, selected, budget, views["evidence_score"]
    )
    coverage = 1.0 - residual
    confidence = float(coverage.min()) if len(coverage) else 0.0
    return {
        "queried_clients": selected,
        "evidence_ids": evidence,
        "abstain": bool(critical_missing_slots) or not evidence,
        "confidence": confidence,
        "client_calls": len(selected),
        "trace": {
            "steps": steps,
            "critical_missing_slots": critical_missing_slots,
            "certificate_interface": "query-slot x client calibrated probability",
        },
    }


def centralized_adapter(
    inputs: dict[str, Any],
    views: dict[str, Any],
    online: np.ndarray,
    budget: int,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    rows = [
        index
        for index, client in enumerate(inputs["candidate_clients"])
        if online[client]
    ]
    rows.sort(
        key=lambda index: (-float(views["evidence_score"][index]), index)
    )
    evidence = [inputs["candidate_ids"][row] for row in rows[:budget]]
    queried = sorted(
        {
            client
            for client in inputs["candidate_clients"]
            if online[client]
        }
    )
    return {
        "queried_clients": queried,
        "evidence_ids": evidence,
        "abstain": not evidence,
        "confidence": 1.0 if evidence else 0.0,
        "client_calls": 1 if evidence else 0,
        "trace": {"access": "centralized global index"},
    }


ADAPTERS: dict[str, Callable[..., dict[str, Any]]] = {
    "Local": local_adapter,
    "All": all_adapter,
    "RAGRoute": ragroute_adapter,
    "UniversalRAG": universalrag_adapter,
    "CoRe-MMRAG": core_adapter,
    "S2G-RAG": s2g_adapter,
    "FedResolve": fedresolve_adapter,
    "Centralized": centralized_adapter,
}


def oracle_client_adapter(
    inputs: dict[str, Any],
    views: dict[str, Any],
    online: np.ndarray,
    budget: int,
    gold: dict[str, Any],
) -> dict[str, Any]:
    groups = gold["minimum_sufficient_client_sets"]
    feasible = [group for group in groups if all(online[client] for client in group)]
    selected = list(feasible[0]) if feasible else []
    evidence = evidence_for_clients(
        inputs, selected, budget, views["evidence_score"]
    )
    return {
        "queried_clients": selected,
        "evidence_ids": evidence,
        "abstain": not feasible,
        "confidence": 1.0,
        "client_calls": len(selected),
        "trace": {"oracle_input": "minimum sufficient client set only"},
    }


def oracle_evidence_adapter(
    inputs: dict[str, Any],
    online: np.ndarray,
    gold: dict[str, Any],
) -> dict[str, Any]:
    client_by_candidate = dict(
        zip(inputs["candidate_ids"], inputs["candidate_clients"])
    )
    support = [
        row["candidate_id"]
        for row in gold["candidate_assessments"]
        if row["stance"] == "support"
        and online[client_by_candidate[row["candidate_id"]]]
    ]
    required = {
        row["candidate_id"]
        for row in gold["candidate_assessments"]
        if row["stance"] == "support"
    }
    queried = sorted({client_by_candidate[value] for value in support})
    return {
        "queried_clients": queried,
        "evidence_ids": support,
        "abstain": set(support) != required,
        "confidence": 1.0,
        "client_calls": len(queried),
        "trace": {"oracle_input": "support page ids and stances"},
    }


def item_metrics(
    output: dict[str, Any],
    gold: dict[str, Any],
    inputs: dict[str, Any],
    certificate_bytes: int,
) -> dict[str, float]:
    selected_clients = set(output["queried_clients"])
    groups = [set(group) for group in gold["minimum_sufficient_client_sets"]]
    client_recall = max(
        (len(selected_clients.intersection(group)) / len(group) for group in groups),
        default=0.0,
    )
    complete = any(group.issubset(selected_clients) for group in groups)
    support = {
        row["candidate_id"]
        for row in gold["candidate_assessments"]
        if row["stance"] == "support"
    }
    evidence = set(output["evidence_ids"])
    evidence_recall = len(evidence.intersection(support)) / max(1, len(support))
    evidence_precision = len(evidence.intersection(support)) / max(1, len(evidence))
    citation_f1 = (
        2 * evidence_recall * evidence_precision
        / (evidence_recall + evidence_precision)
        if evidence_recall + evidence_precision
        else 0.0
    )
    bytes_used = certificate_bytes * int(output["client_calls"]) + sum(
        inputs["candidate_bytes"][inputs["candidate_ids"].index(candidate)]
        for candidate in output["evidence_ids"]
    )
    strict = float(
        complete
        and evidence_recall >= 1.0 - 1e-12
        and not output["abstain"]
    )
    return {
        "client_set_recall": client_recall,
        "complete_client_set": float(complete),
        "evidence_recall": evidence_recall,
        "evidence_precision": evidence_precision,
        "citation_f1": citation_f1,
        "strict_success": strict,
        "client_calls": float(output["client_calls"]),
        "bytes": float(bytes_used),
        "abstain": float(output["abstain"]),
        "confidence": float(np.clip(output["confidence"], 0.0, 1.0)),
    }


def aggregate(rows: list[dict[str, float]], seed: int) -> dict[str, Any]:
    keys = [
        "client_set_recall",
        "complete_client_set",
        "evidence_recall",
        "evidence_precision",
        "citation_f1",
        "strict_success",
        "client_calls",
        "bytes",
        "abstain",
    ]
    result = {
        key: float(np.mean([row[key] for row in rows])) for key in keys
    }
    confidence = [row["confidence"] for row in rows]
    strict = [row["strict_success"] for row in rows]
    result["ece_10bin"] = ece(confidence, strict)
    result["brier"] = float(
        np.mean([(left - right) ** 2 for left, right in zip(confidence, strict)])
    )
    answered = [row for row in rows if not row["abstain"]]
    result["risk_at_coverage"] = {
        "coverage": len(answered) / max(1, len(rows)),
        "risk": 1.0
        - float(np.mean([row["strict_success"] for row in answered]))
        if answered
        else 0.0,
    }
    result["bootstrap_95"] = {
        key: bootstrap_mean(
            [row[key] for row in rows],
            seed + int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big"),
        )
        for key in ("strict_success", "evidence_recall", "client_calls")
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    args = parser.parse_args()
    root = args.root.resolve()
    protocol_path = root / "configs/formal_gold_v1_protocol.json"
    gold_path = root / "data/annotation/gold_v1/fedfinrag_resolve_gold_v1.json"
    tasks_path = root / "data/annotation/gold_v1/tasks_gold_v1.json"
    cache = Path(os.environ.get("SERF_GOLD_CACHE", str(root / "artifacts/gold_v3/cache")))
    output_dir = root / "artifacts/formal_gold_v1"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if sha256(gold_path) != protocol["dataset"]["gold_json_sha256"]:
        raise ValueError("Gold JSON hash does not match pre-registered protocol")
    if sha256(tasks_path) != protocol["dataset"]["tasks_json_sha256"]:
        raise ValueError("Frozen task JSON hash does not match pre-registered protocol")
    gold_document = json.loads(gold_path.read_text(encoding="utf-8"))
    tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
    gold_by_id = {
        row["canonical_id"]: row for row in gold_document["annotations"]
    }
    images = json.loads((cache / "images.json").read_text(encoding="utf-8"))
    queries = json.loads((cache / "queries.json").read_text(encoding="utf-8"))
    image_embeddings = np.load(cache / "image_embeddings.npy").astype(np.float64)
    query_embeddings = np.load(cache / "query_embeddings.npy").astype(np.float64)
    image_embeddings /= np.maximum(
        np.linalg.norm(image_embeddings, axis=1, keepdims=True), 1e-12
    )
    query_embeddings /= np.maximum(
        np.linalg.norm(query_embeddings, axis=1, keepdims=True), 1e-12
    )
    image_index = {row["candidate_id"]: index for index, row in enumerate(images)}
    query_index = {row["query_key"]: index for index, row in enumerate(queries)}
    image_path = {
        row["candidate_id"]: Path(row["path"]) for row in images
    }

    task_inputs: dict[str, dict[str, Any]] = {}
    for task in tasks:
        canonical_id = task["canonical_id"]
        candidate_ids = [
            row["candidate_id"] for row in task["candidate_pages"]
        ]
        clients = [int(row["client_id"]) for row in task["candidate_pages"]]
        image_rows = [image_index[value] for value in candidate_ids]
        full_row = query_index[f"{canonical_id}::full"]
        slot_rows = [
            index
            for key, index in query_index.items()
            if key.startswith(f"{canonical_id}::") and not key.endswith("::full")
        ]
        slot_rows.sort(key=lambda index: queries[index]["slot_id"])
        candidate_matrix = image_embeddings[image_rows]
        task_inputs[canonical_id] = {
            "canonical_id": canonical_id,
            "candidate_ids": candidate_ids,
            "candidate_clients": clients,
            "candidate_bytes": [
                image_path[value].stat().st_size for value in candidate_ids
            ],
            "full_scores": candidate_matrix @ query_embeddings[full_row],
            "slot_scores": np.vstack(
                [candidate_matrix @ query_embeddings[row] for row in slot_rows]
            ),
            "image_similarity": candidate_matrix @ candidate_matrix.T,
        }

    all_online = np.ones(8, dtype=bool)
    predictions: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    summary_by_budget: dict[str, Any] = {}
    per_budget_rows: dict[int, dict[str, list[dict[str, float]]]] = {}
    for budget in protocol["retrieval"]["budget_points"]:
        method_rows: dict[str, list[dict[str, float]]] = defaultdict(list)
        for task in tasks:
            canonical_id = task["canonical_id"]
            inputs = task_inputs[canonical_id]
            views = score_views(inputs, protocol)
            gold = gold_by_id[canonical_id]
            for method in METHODS:
                if method == "Oracle client set":
                    selected = oracle_client_adapter(
                        inputs, views, all_online, budget, gold
                    )
                elif method == "Oracle claim evidence":
                    selected = oracle_evidence_adapter(inputs, all_online, gold)
                else:
                    selected = ADAPTERS[method](
                        inputs, views, all_online, budget, protocol
                    )
                metrics = item_metrics(
                    selected,
                    gold,
                    inputs,
                    protocol["federation"]["certificate_bytes_per_client_call"],
                )
                method_rows[method].append(metrics)
                predictions.append(
                    {
                        "budget": budget,
                        "canonical_id": canonical_id,
                        "method": method,
                        "selection": selected,
                        "metrics": metrics,
                    }
                )
                selections.append(
                    {
                        "budget": budget,
                        "canonical_id": canonical_id,
                        "method": method,
                        "selection": selected,
                    }
                )
        per_budget_rows[budget] = method_rows
        summary_by_budget[str(budget)] = {
            method: aggregate(
                method_rows[method],
                protocol["seed"] + budget * 100 + METHODS.index(method),
            )
            for method in METHODS
        }

    participation: dict[str, Any] = {}
    main_budget = protocol["retrieval"]["main_budget"]
    for method in METHODS:
        answerable_total = answerable_success = 0
        unanswerable_total = unanswerable_abstain = 0
        decisions = []
        for task in tasks:
            canonical_id = task["canonical_id"]
            inputs = task_inputs[canonical_id]
            views = score_views(inputs, protocol)
            gold = gold_by_id[canonical_id]
            for scenario, judgement in zip(
                task["participation_scenarios"],
                gold["participation_assessments"],
            ):
                if scenario["scenario_id"] != judgement["scenario_id"]:
                    raise ValueError("participation scenario order mismatch")
                online = np.zeros(8, dtype=bool)
                online[scenario["online_clients"]] = True
                if method == "Oracle client set":
                    selected = oracle_client_adapter(
                        inputs, views, online, main_budget, gold
                    )
                elif method == "Oracle claim evidence":
                    selected = oracle_evidence_adapter(inputs, online, gold)
                else:
                    selected = ADAPTERS[method](
                        inputs, views, online, main_budget, protocol
                    )
                groups = [
                    set(group) for group in gold["minimum_sufficient_client_sets"]
                ]
                complete = any(
                    group.issubset(set(selected["queried_clients"]))
                    for group in groups
                )
                should_answer = judgement["answerability"] == "answerable"
                if should_answer:
                    answerable_total += 1
                    answerable_success += int(
                        not selected["abstain"] and complete
                    )
                else:
                    unanswerable_total += 1
                    unanswerable_abstain += int(selected["abstain"])
                decisions.append(
                    {
                        "canonical_id": canonical_id,
                        "scenario_id": scenario["scenario_id"],
                        "gold_answerable": should_answer,
                        "abstain": selected["abstain"],
                        "complete_client_set": complete,
                    }
                )
        answer_rate = answerable_success / max(1, answerable_total)
        abstain_rate = unanswerable_abstain / max(1, unanswerable_total)
        participation[method] = {
            "answerable_count": answerable_total,
            "answerable_success": answer_rate,
            "unanswerable_count": unanswerable_total,
            "unanswerable_abstain_accuracy": abstain_rate,
            "balanced_missingness_accuracy": 0.5 * (answer_rate + abstain_rate),
            "decisions": decisions,
        }

    main = summary_by_budget[str(main_budget)]
    oracle_gaps = {
        "local_to_oracle_client_set_complete_client_set": (
            main["Oracle client set"]["complete_client_set"]
            - main["Local"]["complete_client_set"]
        ),
        "all_to_oracle_claim_evidence_strict_success": (
            main["Oracle claim evidence"]["strict_success"]
            - main["All"]["strict_success"]
        ),
        "oracle_client_set_to_oracle_claim_evidence_strict_success": (
            main["Oracle claim evidence"]["strict_success"]
            - main["Oracle client set"]["strict_success"]
        ),
        "fedresolve_vs_s2g_missingness_balanced_accuracy": (
            participation["FedResolve"]["balanced_missingness_accuracy"]
            - participation["S2G-RAG"]["balanced_missingness_accuracy"]
        ),
    }
    go_no_go = {
        "cross_client_necessity": (
            "GO"
            if oracle_gaps[
                "local_to_oracle_client_set_complete_client_set"
            ]
            >= 0.15
            else "NO-GO"
        ),
        "static_fusion_insufficiency": (
            "GO"
            if oracle_gaps[
                "all_to_oracle_claim_evidence_strict_success"
            ]
            >= 0.10
            else "NO-GO"
        ),
        "missingness_mechanism": (
            "GO"
            if oracle_gaps[
                "fedresolve_vs_s2g_missingness_balanced_accuracy"
            ]
            >= 0.05
            else "NO-GO"
        ),
        "three_way_conflict": "NO-GO_NO_VERIFIED_CONTRADICT_LABELS",
    }
    result = {
        "schema_version": "fedresolve-formal-gold-v1-results-v1",
        "dataset_version": gold_document["dataset_version"],
        "protocol_sha256": sha256(protocol_path),
        "gold_sha256": sha256(gold_path),
        "cache_manifest_sha256": sha256(cache / "manifest.json"),
        "selection_data_boundary": {
            "non_oracle_inputs": [
                "frozen question text",
                "candidate page images",
                "candidate-to-client assignment",
                "online mask",
                "pre-registered constants",
            ],
            "non_oracle_gold_labels_read": False,
            "oracle_rows_explicitly_labeled": True,
        },
        "baseline_scope": {
            name: {
                "revision": protocol["baseline_revisions"][name],
                "status": "unified-protocol adaptation; not official native score",
            }
            for name in protocol["baseline_revisions"]
        },
        "summary_by_budget": summary_by_budget,
        "participation_main_budget": participation,
        "oracle_gaps": oracle_gaps,
        "go_no_go": go_no_go,
        "limitations": [
            "Gold v1 contains no verified contradict page; three-way stance and conflict-resolution claims are not eligible.",
            "The four external baselines are minimal unified-protocol adapters because official trained assets or native entrypoints are unavailable.",
            "This evaluation-set experiment measures routing, evidence selection, citation, cost, and missingness; it is not a full-corpus official-native reproduction.",
            "The evaluation uses the frozen task and annotation files.",
        ],
    }
    write_json(output_dir / "formal_results.json", result)
    with (output_dir / "predictions.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output_dir / "selections.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        for row in selections:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(output_dir / "protocol_snapshot.json", protocol)

    report_lines = [
        "# Evaluation-set report",
        "",
        f"- Data: {gold_document['dataset_version']}, 45 questions",
        f"- Main budget: {main_budget} pages and calls",
        "- External methods: adaptations under the shared protocol",
        "",
        "## Main results",
        "",
        "| Method | Strict success | Evidence recall | Citation F1 | Client calls | Abstain |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = main[method]
        report_lines.append(
            f"| {method} | {row['strict_success']:.3f} | "
            f"{row['evidence_recall']:.3f} | {row['citation_f1']:.3f} | "
            f"{row['client_calls']:.2f} | {row['abstain']:.3f} |"
        )
    report_lines.extend(
        [
            "",
            "## Missing/offline",
            "",
            "| Method | Answerable success | Unanswerable abstention | Balanced |",
            "|---|---:|---:|---:|",
        ]
    )
    for method in METHODS:
        row = participation[method]
        report_lines.append(
            f"| {method} | {row['answerable_success']:.3f} | "
            f"{row['unanswerable_abstain_accuracy']:.3f} | "
            f"{row['balanced_missingness_accuracy']:.3f} |"
        )
    report_lines.extend(
        [
            "",
            "## Oracle gap / Go-No-Go",
            "",
        ]
    )
    for key, value in oracle_gaps.items():
        report_lines.append(f"- `{key}`: {value:.3f}")
    for key, value in go_no_go.items():
        report_lines.append(f"- `{key}`: **{value}**")
    report_lines.extend(
        [
            "",
            "## Scope",
            "",
            "- No contradiction labels; three-way stance results are not reported.",
            "- The tasks and annotations are fixed inputs.",
            "",
        ]
    )
    (output_dir / "FORMAL_GOLD_V1_REPORT.md").write_text(
        "\n".join(report_lines), encoding="utf-8", newline="\n"
    )
    print(json.dumps({"output": str(output_dir), "go_no_go": go_no_go}, ensure_ascii=False))


if __name__ == "__main__":
    main()
