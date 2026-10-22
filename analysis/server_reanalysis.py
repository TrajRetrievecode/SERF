#!/usr/bin/env python3
"""server_reanalysis_v1: read-only reanalysis of records already on the server.

Run locally with the README PYTHONPATH; reads plan/prediction files of the
reported sub-question run (*_slot) and of the component-ablation run (serf_v3_ablation/*_test),
prints one JSON document to stdout, and writes nothing.

A. Component ablation (routing level, no generation in that run): per setting and
   budget, evidence hit and evidence recall of the selected pages, calls, and
   routing abstention, for SERF and its four ablations; reproduction check of the
   ablation run's SERF and S2G-RAG rows against the reported run; paired
   bootstrap of SERF minus each ablation on evidence recall and calls.
B. Reported run, budget 2: results split by the number of sub-questions (1 or 2).
C. Reported run, budget 2: paired bootstrap of the proxy balanced score
   (0.5 * (evidence success, all online) + 0.5 * (system abstention, relevant
   clients offline)) of SERF against every other method.
D. Reported run, budget 2: every method combined with the availability test
   (abstain on the unanswerable side when S2G-RAG+avail abstains at routing level,
   the gate being router-independent).
"""
import json
import sys
from collections import defaultdict

import numpy as np

from e2e_common import ROOT, read_json
from e2e_corpus import CACHE
from e2e_records import load_records
from e2e_stats import paired_bootstrap

ART = ROOT / "artifacts"
REF = "FedResolve-v3"
ABLATIONS = ("FedResolve-noResidual", "FedResolve-noEarlyStop", "FedResolve-noAvailability", "FedResolve-noSlots")


def units(page_uids, unit):
    return list(page_uids) if unit == "page" else [u.rsplit("#p", 1)[0] for u in page_uids]


def stream(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def stream_predictions(ds):
    # Keep baseline methods and add the final SERF treatment from its own run.
    yield from stream(ART / "public_e2e_v1" / f"{ds}_slot" / "predictions.jsonl")
    for row in stream(ART / "serf_v3" / f"{ds}_test" / "predictions.jsonl"):
        if row["method"] == REF:
            yield row


def mean(xs):
    return float(np.mean(xs)) if len(xs) else float("nan")


def main():
    out = {"A_ablation": {}, "B_by_subquestions": {}, "C_proxy_paired": {}, "D_method_plus_test": {}}
    for ds in ("finrag", "visdom"):
        records, _, _ = load_records(ds)
        unit = records[0].gold_unit
        gold = {r.uid: set(r.gold_pages if unit == "page" else r.doc_keys) for r in records}
        spans = read_json(CACHE / f"{ds}_slot_queries" / "manifest.json")["spans"]
        nsub = {r.uid: int(spans[i][1]) - int(spans[i][0]) for i, r in enumerate(records)}

        # A. ablation run, routing level
        slot_plan = {}
        for row in stream(ART / "serf_v3" / f"{ds}_test" / "plan.jsonl"):
            if row["method"] in (REF, "S2G-RAG"):
                slot_plan[row["row_key"]] = (row["selected_clients"], row["abstain_routing"], row["evidence_page_uids"])
        repro = {"checked": 0, "mismatched": 0}
        cells = defaultdict(dict)
        for row in stream(ART / "serf_v3_ablation" / f"{ds}_test" / "plan.jsonl"):
            m = row["method"]
            if row["row_key"] in slot_plan:
                repro["checked"] += 1
                if slot_plan[row["row_key"]] != (row["selected_clients"], row["abstain_routing"], row["evidence_page_uids"]):
                    repro["mismatched"] += 1
            if m not in (REF,) + ABLATIONS + ("S2G-RAG", "S2G-RAG+avail"):
                continue
            g = gold[row["uid"]]
            ev = set(units(row["evidence_page_uids"], unit))
            hit = 1.0 if (ev & g) and not row["abstain_routing"] else 0.0
            recall = len(ev & g) / len(g) if g and not row["abstain_routing"] else 0.0
            key = (row["config"], m, row["budget"], row["scenario"])
            cells[key][row["uid"]] = (hit, recall, float(row["cost"]["client_calls"]), 1.0 if row["abstain_routing"] else 0.0)
        del slot_plan
        summary = []
        for (config, m, b, sc), recs in sorted(cells.items()):
            vals = np.array(list(recs.values()))
            summary.append({"config": config, "method": m, "budget": b, "scenario": sc, "n": len(recs),
                            "evidence_hit": float(vals[:, 0].mean()), "evidence_recall": float(vals[:, 1].mean()),
                            "calls": float(vals[:, 2].mean()), "routing_abstain": float(vals[:, 3].mean())})
        paired = []
        for (config, m, b, sc), recs in sorted(cells.items()):
            if m == REF or sc != "full":
                continue
            ref = cells.get((config, REF, b, sc))
            if not ref:
                continue
            uids = sorted(set(ref) & set(recs))
            for j, name in ((1, "evidence_recall"), (2, "calls")):
                paired.append({"config": config, "comparator": m, "budget": b, "metric": name,
                               **paired_bootstrap([ref[u][j] for u in uids], [recs[u][j] for u in uids])})
        out["A_ablation"][ds] = {"reproduction_vs_reported_run": repro, "cells": summary, "paired_serf_minus": paired}
        del cells

        # B, C, D from the reported run's predictions at budget 2
        idx = defaultdict(lambda: defaultdict(dict))
        for row in stream_predictions(ds):
            if row["budget"] != 2:
                continue
            idx[(row["config"], row["method"])][row["scenario"]][row["uid"]] = (
                1.0 if row["evidence_correct"] else 0.0,
                float(row["evidence_recall"]),
                float(row["client_calls"]),
                1.0 if row["abstain_system"] else 0.0,
                1.0 if row["abstain_routing"] else 0.0,
                1.0 if row["answer_correct"] else 0.0,
            )
        configs = sorted({c for c, _ in idx})
        methods = sorted({m for _, m in idx})

        b_rows = []
        for config in configs:
            for m in methods:
                full = idx[(config, m)].get("full", {})
                miss = idx[(config, m)].get("qrel_missing", {})
                for k in (1, 2):
                    fu = [v for u, v in full.items() if nsub[u] == k]
                    mu = [v for u, v in miss.items() if nsub[u] == k]
                    b_rows.append({"config": config, "method": m, "subquestions": k, "n_full": len(fu),
                                   "evidence_recall": mean([v[1] for v in fu]), "calls": mean([v[2] for v in fu]),
                                   "routing_abstain_missing": mean([v[4] for v in mu])})
        out["B_by_subquestions"][ds] = b_rows

        def per_record(config, m, gate=None):
            full = idx[(config, m)].get("full", {})
            miss = idx[(config, m)].get("qrel_missing", {})
            res = {}
            for u in set(full) & set(miss):
                abstain = miss[u][3]
                if gate is not None:
                    abstain = max(abstain, gate.get(u, 0.0))
                res[u] = (0.5 * (full[u][0] + abstain), full[u][0], abstain)
            return res

        c_rows, d_rows = [], []
        for config in configs:
            ref = per_record(config, REF)
            gate = {u: v[4] for u, v in idx[(config, "S2G-RAG+avail")].get("qrel_missing", {}).items()}
            for m in methods:
                if m in (REF, "Oracle-evidence"):
                    continue
                other = per_record(config, m)
                if not other:
                    continue
                uids = sorted(set(ref) & set(other))
                c_rows.append({"config": config, "comparator": m,
                               **paired_bootstrap([ref[u][0] for u in uids], [other[u][0] for u in uids])})
                plus = per_record(config, m, gate)
                d_rows.append({"config": config, "method": m + "+test", "n": len(plus),
                               "answerable_evidence": mean([v[1] for v in plus.values()]),
                               "unanswerable_abstain": mean([v[2] for v in plus.values()]),
                               "balanced": mean([v[0] for v in plus.values()])})
        out["C_proxy_paired"][ds] = c_rows
        out["D_method_plus_test"][ds] = d_rows
        del idx
    json.dump(out, sys.stdout)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
