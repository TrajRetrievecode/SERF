#!/usr/bin/env python3
"""gold_enumeration_v1: every single-client-offline scenario on the evaluation set.

Imports the frozen runner unchanged, re-runs the 80 recorded scenarios as a
reproduction check, then takes offline, one at a time, every client holding a
candidate page of each question. Protocol: protocol_enumeration_v1.json.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("SERF_ROOT", str(HERE.parents[1])))
sys.path.insert(0, str(ROOT / "scripts"))
import run_formal_gold_v1 as rf  # noqa: E402

OUT = ROOT / "artifacts" / "enumeration"
OUT.mkdir(parents=True, exist_ok=True)

EXTERNAL = ("RAGRoute", "UniversalRAG", "CoRe-MMRAG", "S2G-RAG")
METHODS = ("Local", "All", "RAGRoute", "UniversalRAG", "CoRe-MMRAG", "S2G-RAG",
           "FedResolve", "Centralized", "Oracle client set")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def binom_cdf(k: int, n: int, p: float) -> float:
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    def solve(f, lo=0.0, hi=1.0):
        for _ in range(200):
            mid = (lo + hi) / 2
            if f(mid):
                hi = mid
            else:
                lo = mid
        return (lo + hi) / 2
    low = 0.0 if k == 0 else solve(lambda p: 1 - binom_cdf(k - 1, n, p) >= alpha / 2)
    high = 1.0 if k == n else solve(lambda p: binom_cdf(k, n, p) <= alpha / 2)
    return low, high


def main() -> int:
    protocol_path = ROOT / "configs/formal_gold_v1_protocol.json"
    gold_path = ROOT / "data/annotation/gold_v1/fedfinrag_resolve_gold_v1.json"
    tasks_path = ROOT / "data/annotation/gold_v1/tasks_gold_v1.json"
    cache = Path(os.environ.get("SERF_GOLD_CACHE", str(ROOT / "artifacts/gold_v3/cache")))
    recorded = json.loads((ROOT / "artifacts/formal_gold_v1/formal_results.json").read_text(encoding="utf-8-sig"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8-sig"))
    assert sha(gold_path) == protocol["dataset"]["gold_json_sha256"]
    assert sha(tasks_path) == protocol["dataset"]["tasks_json_sha256"]
    assert sha(protocol_path) == recorded["protocol_sha256"]
    assert sha(cache / "manifest.json") == recorded["cache_manifest_sha256"]

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

    task_inputs = {}
    for task in tasks:
        cid = task["canonical_id"]
        candidate_ids = [row["candidate_id"] for row in task["candidate_pages"]]
        clients = [int(row["client_id"]) for row in task["candidate_pages"]]
        image_rows = [image_index[v] for v in candidate_ids]
        full_row = query_index[f"{cid}::full"]
        slot_rows = [i for k, i in query_index.items() if k.startswith(f"{cid}::") and not k.endswith("::full")]
        slot_rows.sort(key=lambda i: queries[i]["slot_id"])
        m = image_embeddings[image_rows]
        task_inputs[cid] = {
            "canonical_id": cid,
            "candidate_ids": candidate_ids,
            "candidate_clients": clients,
            "candidate_bytes": [0] * len(candidate_ids),
            "full_scores": m @ query_embeddings[full_row],
            "slot_scores": np.vstack([m @ query_embeddings[r] for r in slot_rows]),
            "image_similarity": m @ m.T,
        }

    budget = protocol["retrieval"]["main_budget"]

    def decide(method, inputs, views, online, gold):
        if method == "Oracle client set":
            return rf.oracle_client_adapter(inputs, views, online, budget, gold)
        return rf.ADAPTERS[method](inputs, views, online, budget, protocol)

    # Reproduction check on the 80 recorded scenarios.
    rec = {(m, d["scenario_id"]): d for m, block in recorded["participation_main_budget"].items()
           if m in METHODS for d in block["decisions"]}
    mismatches, checked = [], 0
    for task in tasks:
        cid = task["canonical_id"]
        inputs, gold = task_inputs[cid], gold_by_id[cid]
        views = rf.score_views(inputs, protocol)
        groups = [set(g) for g in gold["minimum_sufficient_client_sets"]]
        for scenario in task["participation_scenarios"]:
            online = np.zeros(8, dtype=bool)
            online[scenario["online_clients"]] = True
            for method in METHODS:
                sel = decide(method, inputs, views, online, gold)
                complete = any(g.issubset(set(sel["queried_clients"])) for g in groups)
                want = rec[(method, scenario["scenario_id"])]
                checked += 1
                if bool(sel["abstain"]) != bool(want["abstain"]) or complete != bool(want["complete_client_set"]):
                    mismatches.append((method, scenario["scenario_id"]))
    print(json.dumps({"reproduction_checked": checked, "mismatches": len(mismatches), "examples": mismatches[:5]}))
    if mismatches:
        return 1

    # Enumeration.
    recorded_offline = {}
    for task in tasks:
        for scenario in task["participation_scenarios"]:
            off = sorted(set(range(8)) - set(scenario["online_clients"]))
            recorded_offline[(task["canonical_id"], off[0])] = scenario["scenario_id"]
    rows = []
    for task in tasks:
        cid = task["canonical_id"]
        inputs, gold = task_inputs[cid], gold_by_id[cid]
        views = rf.score_views(inputs, protocol)
        groups = [set(g) for g in gold["minimum_sufficient_client_sets"]]
        necessary = set.intersection(*groups) if groups else set()
        for client in sorted(set(inputs["candidate_clients"])):
            online = np.ones(8, dtype=bool)
            online[client] = False
            answerable = any(g.issubset(set(np.flatnonzero(online).tolist())) for g in groups)
            decisions = {m: decide(m, inputs, views, online, gold) for m in METHODS}
            gate = bool(decisions["FedResolve"]["trace"]["critical_missing_slots"])
            base = {
                "canonical_id": cid,
                "offline_client": client,
                "answerable": answerable,
                "offline_is_necessary": client in necessary,
                "recorded_scenario": recorded_offline.get((cid, client), ""),
                "serf_gate_fired": gate,
            }
            for m in METHODS:
                sel = decisions[m]
                complete = any(g.issubset(set(sel["queried_clients"])) for g in groups)
                rows.append({**base, "method": m, "abstain": bool(sel["abstain"]), "complete_client_set": complete})
            for m in EXTERNAL:
                sel = decisions[m]
                complete = any(g.issubset(set(sel["queried_clients"])) for g in groups)
                rows.append({**base, "method": m + "+avail", "abstain": bool(sel["abstain"]) or gate, "complete_client_set": complete})

    with (OUT / "enumerated_decisions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    def summarize(subset):
        out = {}
        by_method = defaultdict(list)
        for r in subset:
            by_method[r["method"]].append(r)
        for m, rs in by_method.items():
            ans = [r for r in rs if r["answerable"]]
            una = [r for r in rs if not r["answerable"]]
            a_ok = sum(1 for r in ans if r["complete_client_set"] and not r["abstain"])
            u_ok = sum(1 for r in una if r["abstain"])
            a_rate = a_ok / len(ans) if ans else float("nan")
            u_rate = u_ok / len(una) if una else float("nan")
            out[m] = {
                "answerable": len(ans), "answerable_success": a_ok,
                "answerable_rate": a_rate, "answerable_ci": clopper_pearson(a_ok, len(ans)) if ans else None,
                "unanswerable": len(una), "unanswerable_abstain": u_ok,
                "unanswerable_rate": u_rate, "unanswerable_ci": clopper_pearson(u_ok, len(una)) if una else None,
                "false_abstentions": sum(1 for r in ans if r["abstain"]),
                "balanced": 0.5 * (a_rate + u_rate) if ans and una else float("nan"),
            }
        return out

    summary = {
        "protocol_sha256": sha(ROOT / "configs/enumeration/protocol_enumeration_v1.json"),
        "reproduction": {"checked": checked, "mismatches": 0},
        "scenarios": len({(r["canonical_id"], r["offline_client"]) for r in rows}),
        "all": summarize(rows),
        "recorded_80": summarize([r for r in rows if r["recorded_scenario"]]),
        "new_only": summarize([r for r in rows if not r["recorded_scenario"]]),
        "unanswerable_recorded_offline_client": summarize([r for r in rows if not r["answerable"] and r["recorded_scenario"]]),
        "unanswerable_other_necessary_client": summarize([r for r in rows if not r["answerable"] and not r["recorded_scenario"]]),
    }
    (OUT / "enumeration_summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("reproduction", "scenarios")}))
    for block in ("all", "recorded_80", "new_only", "unanswerable_other_necessary_client"):
        s = summary[block]["FedResolve"]
        print(block, {k: s[k] for k in ("answerable", "answerable_success", "unanswerable", "unanswerable_abstain", "false_abstentions", "balanced")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
