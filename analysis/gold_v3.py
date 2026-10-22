#!/usr/bin/env python3
"""SERF-v3 on the evaluation set, one run (protocol_serf_v3.json with amendment_1).

Imports the frozen v1 runner unchanged. SERF-v3 reuses the v1 adapter for scoring, routing and
the availability test and replaces only evidence selection. Before evaluating v3 the script
reproduces, from the frozen runner: the full recorded summaries (every method, budgets 1-3,
including bootstrap intervals and calibration), the 720 recorded decisions on the 80 paired
scenarios, and the SERF-v1 counts on the 176 single-client-offline scenarios. Any mismatch stops
the run before v3 is evaluated.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("SERF_ROOT", str(HERE.parent)))
OUT = ROOT / "artifacts" / "analysis"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "scripts"))
import run_formal_gold_v1 as rf  # noqa: E402

CACHE = Path(os.environ.get("SERF_GOLD_CACHE", str(ROOT / "artifacts/gold_v3/cache")))
RUNNER_SHA = "7cf78be0360ab12ab4e4f48a5cf8137f64ce3a9adae051f9d3b3ab09ebe4e1fe"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(cond, message):
    if not cond:
        raise SystemExit("STOP: " + message)


def close(a, b, tol=1e-12):
    if isinstance(a, dict):
        return isinstance(b, dict) and set(a) == set(b) and all(close(a[k], b[k], tol) for k in a)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return (math.isnan(a) and math.isnan(b)) or abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))
    return a == b


def v3_select(inputs, called, budget, protocol, evidence_score):
    """Evidence selection of SERF-v3 (single_change; amendment_1 presentation order)."""
    called = set(called)
    rows = [i for i, c in enumerate(inputs["candidate_clients"]) if c in called]
    if not rows or budget <= 0:
        return []
    w = float(protocol["fedresolve"]["claim_query_weight"])
    wf = float(protocol["fedresolve"]["full_query_weight"])
    z = w * np.asarray(inputs["slot_scores"], dtype=np.float64) + wf * np.asarray(inputs["full_scores"], dtype=np.float64)[None, :]
    e = evidence_score
    remaining = list(rows)
    first = max(remaining, key=lambda r: (float(e[r]), -r))
    chosen = [first]
    remaining.remove(first)
    best = z[:, first].copy()
    while remaining and len(chosen) < budget:
        j = int(np.argmin(best))
        nxt = max(remaining, key=lambda r: (float(z[j, r]), float(e[r]), -r))
        chosen.append(nxt)
        remaining.remove(nxt)
        best = np.maximum(best, z[:, nxt])
    chosen.sort(key=lambda r: (-float(e[r]), r))
    return [inputs["candidate_ids"][r] for r in chosen]


def v3_adapter(inputs, views, online, budget, protocol):
    base = rf.ADAPTERS["FedResolve"](inputs, views, online, budget, protocol)
    evidence = v3_select(inputs, base["queried_clients"], budget, protocol, views["evidence_score"])
    out = dict(base)
    out["evidence_ids"] = evidence
    out["abstain"] = bool(base["trace"]["critical_missing_slots"]) or not evidence
    out["trace"] = {**base["trace"], "evidence_selection": "serf-v3 coverage over called clients' pages"}
    return out


def main() -> int:
    protocol_path = ROOT / "configs/formal_gold_v1_protocol.json"
    gold_path = ROOT / "data/annotation/gold_v1/fedfinrag_resolve_gold_v1.json"
    tasks_path = ROOT / "data/annotation/gold_v1/tasks_gold_v1.json"
    runner_path = ROOT / "scripts/run_formal_gold_v1.py"
    recorded = json.loads((ROOT / "artifacts/formal_gold_v1/formal_results.json").read_text(encoding="utf-8-sig"))
    enum_rec = json.loads((ROOT / "artifacts/enumeration/enumeration_summary.json").read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8-sig"))
    need(sha(gold_path) == protocol["dataset"]["gold_json_sha256"], "gold hash")
    need(sha(tasks_path) == protocol["dataset"]["tasks_json_sha256"], "tasks hash")
    need(sha(protocol_path) == recorded["protocol_sha256"], "protocol hash")
    need(sha(CACHE / "manifest.json") == recorded["cache_manifest_sha256"], "cache manifest hash")
    need(sha(runner_path).startswith(RUNNER_SHA), "runner hash")
    manifest = json.loads((CACHE / "manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest["artifacts"].items():
        need(sha(CACHE / name) == digest, f"cache artifact {name}")

    gold_by_id = {r["canonical_id"]: r for r in json.loads(gold_path.read_text(encoding="utf-8-sig"))["annotations"]}
    tasks = json.loads(tasks_path.read_text(encoding="utf-8-sig"))
    images = json.loads((CACHE / "images.json").read_text(encoding="utf-8"))
    queries = json.loads((CACHE / "queries.json").read_text(encoding="utf-8"))
    image_bytes = {row["candidate_id"]: Path(row["path"]).stat().st_size for row in images}
    ie = np.load(CACHE / "image_embeddings.npy").astype(np.float64)
    qe = np.load(CACHE / "query_embeddings.npy").astype(np.float64)
    ie /= np.maximum(np.linalg.norm(ie, axis=1, keepdims=True), 1e-12)
    qe /= np.maximum(np.linalg.norm(qe, axis=1, keepdims=True), 1e-12)
    ii = {row["candidate_id"]: i for i, row in enumerate(images)}
    qi = {row["query_key"]: i for i, row in enumerate(queries)}
    task_inputs = {}
    for task in tasks:
        cid = task["canonical_id"]
        cand = [row["candidate_id"] for row in task["candidate_pages"]]
        m = ie[[ii[v] for v in cand]]
        slot_rows = sorted([i for k, i in qi.items() if k.startswith(f"{cid}::") and not k.endswith("::full")],
                           key=lambda i: queries[i]["slot_id"])
        task_inputs[cid] = {
            "canonical_id": cid, "candidate_ids": cand,
            "candidate_clients": [int(row["client_id"]) for row in task["candidate_pages"]],
            "candidate_bytes": [image_bytes[v] for v in cand],
            "full_scores": m @ qe[qi[f"{cid}::full"]],
            "slot_scores": np.vstack([m @ qe[r] for r in slot_rows]),
            "image_similarity": m @ m.T,
        }
    cert = protocol["federation"]["certificate_bytes_per_client_call"]
    all_online = np.ones(8, dtype=bool)

    def decide(method, inputs, views, online, budget, gold):
        if method == "Oracle client set":
            return rf.oracle_client_adapter(inputs, views, online, budget, gold)
        if method == "Oracle claim evidence":
            return rf.oracle_evidence_adapter(inputs, online, gold)
        if method == "SERF-v3":
            return v3_adapter(inputs, views, online, budget, protocol)
        return rf.ADAPTERS[method](inputs, views, online, budget, protocol)

    def run_online(method, budget):
        per = {}
        for task in tasks:
            cid = task["canonical_id"]
            inputs = task_inputs[cid]
            views = rf.score_views(inputs, protocol)
            out = decide(method, inputs, views, all_online, budget, gold_by_id[cid])
            per[cid] = (out, rf.item_metrics(out, gold_by_id[cid], inputs, cert))
        return per

    # 1. full recorded summaries, every method, budgets 1-3
    for budget in protocol["retrieval"]["budget_points"]:
        for method in rf.METHODS:
            per = run_online(method, budget)
            agg = rf.aggregate([m for _, m in per.values()], protocol["seed"] + budget * 100 + rf.METHODS.index(method))
            need(close(agg, recorded["summary_by_budget"][str(budget)][method]), f"summary {method} B={budget}")
    # 2. paired scenarios: 720 recorded decisions
    paired = []
    for task in tasks:
        gold = gold_by_id[task["canonical_id"]]
        for sc, judg in zip(task["participation_scenarios"], gold["participation_assessments"]):
            need(sc["scenario_id"] == judg["scenario_id"], "scenario order")
            online = np.zeros(8, dtype=bool); online[sc["online_clients"]] = True
            paired.append((task["canonical_id"], sc["scenario_id"], online, judg["answerability"] == "answerable"))
    need(len(paired) == 80, "80 paired scenarios")
    rec = {(m, d["scenario_id"]): d for m, b in recorded["participation_main_budget"].items() for d in b["decisions"]}
    checked = 0
    for cid, sid, online, _ in paired:
        inputs = task_inputs[cid]; views = rf.score_views(inputs, protocol); gold = gold_by_id[cid]
        groups = [set(g) for g in gold["minimum_sufficient_client_sets"]]
        for method in rf.METHODS:
            out = decide(method, inputs, views, online, protocol["retrieval"]["main_budget"], gold)
            complete = any(g.issubset(set(out["queried_clients"])) for g in groups)
            want = rec[(method, sid)]
            need(bool(out["abstain"]) == bool(want["abstain"]) and complete == bool(want["complete_client_set"]), f"decision {method} {sid}")
            checked += 1
    # 3. enumerated scenarios
    enum = []
    for task in tasks:
        cid = task["canonical_id"]
        groups = [set(g) for g in gold_by_id[cid]["minimum_sufficient_client_sets"]]
        for client in sorted(set(task_inputs[cid]["candidate_clients"])):
            online = np.ones(8, dtype=bool); online[client] = False
            on = set(np.flatnonzero(online).tolist())
            enum.append((cid, str(client), online, any(g.issubset(on) for g in groups)))
    need(len(enum) == 176, "176 enumerated scenarios")

    def missing(scenarios, method):
        a_ok = a_n = u_ok = u_n = 0
        decisions = {}
        for cid, sid, online, answerable in scenarios:
            inputs = task_inputs[cid]; views = rf.score_views(inputs, protocol); gold = gold_by_id[cid]
            groups = [set(g) for g in gold["minimum_sufficient_client_sets"]]
            out = decide(method, inputs, views, online, 2, gold)
            complete = any(g.issubset(set(out["queried_clients"])) for g in groups)
            decisions[(cid, sid)] = (bool(out["abstain"]), tuple(out["queried_clients"]))
            if answerable:
                a_n += 1; a_ok += int(complete and not out["abstain"])
            else:
                u_n += 1; u_ok += int(out["abstain"])
        bal = 0.5 * (a_ok / a_n + u_ok / u_n) if a_n and u_n else float("nan")
        return {"answerable": a_n, "answered": a_ok, "unanswerable": u_n, "abstained": u_ok, "balanced": bal}, decisions

    v1_enum, v1_enum_dec = missing(enum, "FedResolve")
    want = enum_rec["all"]["FedResolve"]
    need(v1_enum["answered"] == want["answerable_success"] and v1_enum["abstained"] == want["unanswerable_abstain"], "enumeration counts v1")
    print(json.dumps({"reproduction": "ok", "summaries": len(rf.METHODS) * 3, "paired_decisions": checked,
                      "enumeration_v1": v1_enum}))

    # SERF-v3, evaluated once
    results = {"protocol_sha256": sha(ROOT / "configs/serf_v3/protocol_serf_v3.json"), "code_sha256": sha(Path(__file__)),
               "runner_sha256": sha(runner_path), "image_paths_sha256": sha(CACHE / "images.json"),
               "image_sizes_source": "File sizes of the candidate images referenced by images.json",
               "reproduction": {"summaries": len(rf.METHODS) * 3, "paired_decisions": checked, "enumeration_v1": v1_enum},
               "online": {}, "paired_items": {}}
    items = []
    keys = ("strict_success", "complete_client_set", "evidence_recall", "evidence_precision", "citation_f1", "client_calls", "bytes", "abstain")
    for budget in (1, 2, 3):
        v1 = run_online("FedResolve", budget)
        v3 = run_online("SERF-v3", budget)
        results["online"][str(budget)] = {
            "SERF-v1": rf.aggregate([m for _, m in v1.values()], protocol["seed"] + budget * 100 + rf.METHODS.index("FedResolve")),
            "SERF-v3": rf.aggregate([m for _, m in v3.values()], protocol["seed"] + budget * 100 + rf.METHODS.index("FedResolve")),
        }
        cmp = {}
        for k in ("strict_success", "evidence_recall", "evidence_precision", "citation_f1", "bytes"):
            d = [v3[c][1][k] - v1[c][1][k] for c in v1]
            cmp[k] = {"v3_better": sum(x > 1e-12 for x in d), "v1_better": sum(x < -1e-12 for x in d),
                      "equal": sum(abs(x) <= 1e-12 for x in d), "mean_difference": float(np.mean(d))}
        cmp["calls_identical"] = all(v3[c][0]["queried_clients"] == v1[c][0]["queried_clients"] for c in v1)
        cmp["abstain_identical"] = all(v3[c][0]["abstain"] == v1[c][0]["abstain"] for c in v1)
        need(cmp["calls_identical"] and cmp["abstain_identical"], f"v3 routing differs from v1 at B={budget}")
        if budget == 1:
            need(all(v3[c][0]["evidence_ids"] == v1[c][0]["evidence_ids"] for c in v1), "v3 differs from v1 at B=1")
        results["paired_items"][str(budget)] = cmp
        for c in v1:
            items.append({"budget": budget, "canonical_id": c,
                          **{f"v1_{k}": v1[c][1][k] for k in keys}, **{f"v3_{k}": v3[c][1][k] for k in keys},
                          "v1_evidence": "|".join(v1[c][0]["evidence_ids"]), "v3_evidence": "|".join(v3[c][0]["evidence_ids"])})
    for name, scen in (("paired_80", [(c, s, o, a) for c, s, o, a in paired]), ("enumerated_176", enum)):
        v3m, v3d = missing(scen, "SERF-v3")
        v1m, v1d = missing(scen, "FedResolve")
        need(v3d == v1d, f"v3 missingness decisions differ from v1 ({name})")
        results[f"missingness_{name}"] = {"SERF-v3": v3m, "SERF-v1": v1m, "decisions_identical": True}

    (OUT / "gold_v3_results.json").write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    with (OUT / "gold_v3_items.csv").open("w", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(items[0].keys())); w.writeheader(); w.writerows(items)
    for b in ("1", "2", "3"):
        print(b, {n: {k: round(a[k], 4) for k in ("strict_success", "evidence_recall", "evidence_precision", "citation_f1", "client_calls", "bytes")}
                  for n, a in results["online"][b].items()})
        print("  ", {k: v for k, v in results["paired_items"][b].items()})
    print("paired80", results["missingness_paired_80"]["SERF-v3"])
    print("enum176", results["missingness_enumerated_176"]["SERF-v3"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
