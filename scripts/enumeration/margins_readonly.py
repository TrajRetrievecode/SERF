#!/usr/bin/env python3
"""Read-only descriptive analysis of the availability test on the enumeration of single-client-offline
scenarios of the evaluation set (same inputs, frozen code and cache as enumerate_single_offline.py).

For every scenario (a question with one candidate-holding client offline, the other seven online) and
every sub-question, the script records whether the client with the highest blended score z_{j,c} is
the offline one and, if so, its lead over the best online client (the raw margin of the test) and
the probability ratio. The scenario statistic is the largest lead over the sub-questions whose best
client is offline; a scenario in which the offline client is the best client of no sub-question has
no lead. The test fires when the lead is at least the margin (0.02) and the ratio at least 1.15.

Outputs are written under artifacts/enumeration. Outputs: margins_by_scenario.csv (one row per scenario
and sub-question with an offline best client) and margins_summary.json. The SERF decisions
recomputed here must equal those of enumerated_decisions.csv (asserted).
"""
from __future__ import annotations

import csv
import json
import sys
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("SERF_ROOT", str(HERE.parents[1])))
sys.path.insert(0, str(ROOT / "scripts"))
import run_formal_gold_v1 as rf  # noqa: E402

OUT = ROOT / "artifacts" / "enumeration"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> int:
    protocol = json.loads((ROOT / "configs/formal_gold_v1_protocol.json").read_text(encoding="utf-8-sig"))
    gold_path = ROOT / "data/annotation/gold_v1/fedfinrag_resolve_gold_v1.json"
    tasks_path = ROOT / "data/annotation/gold_v1/tasks_gold_v1.json"
    cache = Path(os.environ.get("SERF_GOLD_CACHE", str(ROOT / "artifacts/gold_v3/cache")))
    gold_by_id = {r["canonical_id"]: r for r in json.loads(gold_path.read_text(encoding="utf-8-sig"))["annotations"]}
    tasks = json.loads(tasks_path.read_text(encoding="utf-8-sig"))
    images = json.loads((cache / "images.json").read_text(encoding="utf-8"))
    queries = json.loads((cache / "queries.json").read_text(encoding="utf-8"))
    image_embeddings = np.load(cache / "image_embeddings.npy").astype(np.float64)
    query_embeddings = np.load(cache / "query_embeddings.npy").astype(np.float64)
    image_embeddings /= np.maximum(np.linalg.norm(image_embeddings, axis=1, keepdims=True), 1e-12)
    query_embeddings /= np.maximum(np.linalg.norm(query_embeddings, axis=1, keepdims=True), 1e-12)
    image_index = {row["candidate_id"]: i for i, row in enumerate(images)}
    query_index = {row["query_key"]: i for i, row in enumerate(queries)}
    margin = protocol["fedresolve"]["missingness_raw_margin"]
    ratio_thr = protocol["fedresolve"]["missingness_probability_ratio"]
    budget = protocol["retrieval"]["main_budget"]

    recorded = {}
    with (OUT / "enumerated_decisions.csv").open(encoding="utf-8") as handle:
        for r in csv.DictReader(handle):
            if r["method"] == "FedResolve":
                recorded[(r["canonical_id"], int(r["offline_client"]))] = (r["answerable"] == "True", r["abstain"] == "True")

    rows, scenarios = [], []
    for task in tasks:
        cid = task["canonical_id"]
        candidate_ids = [row["candidate_id"] for row in task["candidate_pages"]]
        clients = [int(row["client_id"]) for row in task["candidate_pages"]]
        m = image_embeddings[[image_index[v] for v in candidate_ids]]
        full_row = query_index[f"{cid}::full"]
        slot_rows = sorted((i for k, i in query_index.items() if k.startswith(f"{cid}::") and not k.endswith("::full")),
                           key=lambda i: queries[i]["slot_id"])
        inputs = {"canonical_id": cid, "candidate_ids": candidate_ids, "candidate_clients": clients,
                  "candidate_bytes": [0] * len(candidate_ids), "full_scores": m @ query_embeddings[full_row],
                  "slot_scores": np.vstack([m @ query_embeddings[r] for r in slot_rows]), "image_similarity": m @ m.T}
        views = rf.score_views(inputs, protocol)
        combined, probabilities = views["combined"], views["probabilities"]
        groups = [set(g) for g in gold_by_id[cid]["minimum_sufficient_client_sets"]]
        for client in sorted(set(clients)):
            online = np.ones(8, dtype=bool)
            online[client] = False
            answerable = any(g.issubset(set(np.flatnonzero(online).tolist())) for g in groups)
            decision = rf.fedresolve_adapter(inputs, views, online, budget, protocol)
            fired = bool(decision["trace"]["critical_missing_slots"])
            assert recorded[(cid, client)] == (answerable, bool(decision["abstain"])), (cid, client)
            leads, signed = [], []
            for slot in range(len(combined)):
                order = np.argsort(-combined[slot])
                finite = [int(c) for c in order if np.isfinite(combined[slot, c])]
                global_client = finite[0] if finite else None
                online_client = next((c for c in finite if online[c]), None)
                if online_client is None or not np.isfinite(combined[slot, client]):
                    continue
                # signed lead of the offline client over the best online client, for every sub-question
                # whose candidate pages include a page of the offline client (negative when another
                # client scores higher)
                lead_signed = float(combined[slot, client] - combined[slot, online_client])
                signed.append(lead_signed)
                offline_best = global_client == client
                ratio = float(probabilities[slot, client] / max(probabilities[slot, online_client], 1e-9))
                flagged = offline_best and lead_signed >= margin and ratio >= ratio_thr
                rows.append({"canonical_id": cid, "offline_client": client, "answerable": answerable, "slot": slot,
                             "offline_client_is_best": offline_best, "lead": round(lead_signed, 6),
                             "probability_ratio": round(ratio, 6), "flagged": flagged})
                if offline_best:
                    leads.append(lead_signed)
            scenarios.append({"canonical_id": cid, "offline_client": client, "answerable": answerable,
                              "slots_with_offline_best": len(leads), "max_lead": (max(leads) if leads else None),
                              "max_signed_lead": (max(signed) if signed else None), "test_fired": fired})

    with (OUT / "margins_by_scenario.csv").open("w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    ans = [s for s in scenarios if s["answerable"]]
    una = [s for s in scenarios if not s["answerable"]]
    summary = {
        "scenarios": len(scenarios), "answerable": len(ans), "unanswerable": len(una),
        "margin": margin, "probability_ratio": ratio_thr,
        "answerable_with_offline_best": sum(1 for s in ans if s["slots_with_offline_best"]),
        "answerable_max_lead_at_least_margin": sum(1 for s in ans if s["max_lead"] is not None and s["max_lead"] >= margin),
        "answerable_test_fired": sum(1 for s in ans if s["test_fired"]),
        "answerable_max_lead_max": max((s["max_lead"] for s in ans if s["max_lead"] is not None), default=None),
        "unanswerable_with_offline_best": sum(1 for s in una if s["slots_with_offline_best"]),
        "unanswerable_max_lead_at_least_margin": sum(1 for s in una if s["max_lead"] is not None and s["max_lead"] >= margin),
        "unanswerable_test_fired": sum(1 for s in una if s["test_fired"]),
        "unanswerable_missed": [{"canonical_id": s["canonical_id"], "offline_client": s["offline_client"],
                                 "max_lead": s["max_lead"]} for s in una if not s["test_fired"]],
        "unanswerable_max_lead_min": min((s["max_lead"] for s in una if s["max_lead"] is not None), default=None),
        "unanswerable_max_lead_median": float(np.median([s["max_lead"] for s in una if s["max_lead"] is not None])),
        "answerable_max_signed_lead_max": max(s["max_signed_lead"] for s in ans),
        "answerable_max_signed_lead_min": min(s["max_signed_lead"] for s in ans),
        "unanswerable_max_signed_lead_min": min(s["max_signed_lead"] for s in una),
        "unanswerable_max_signed_lead_below_zero": sum(1 for s in una if s["max_signed_lead"] < 0),
        "unanswerable_max_signed_lead_between_zero_and_margin": sum(1 for s in una if 0 <= s["max_signed_lead"] < margin),
        "scenarios_list": scenarios,
    }
    (OUT / "margins_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "scenarios_list"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
