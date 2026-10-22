#!/usr/bin/env python3
"""Read-only analysis of the SERF-v3 component ablation (protocol_v3_ablation.json); prints JSON.

Checks: S2G-RAG rows equal the reported run (public_e2e_v1/<ds>_slot); FedResolve-v3 rows equal the
SERF-v3 test run (serf_v3/<ds>_test); every ablation variant has the same selected clients and routing
abstention as the SERF ablation run (serf_v3_ablation/<ds>_test). Then: mean over the four settings of
evidence recall and calls (budget 2, full) and routing abstention (qrel_missing), and the paired
bootstrap per setting of SERF-v3 minus each variant on evidence recall and calls.
"""
import json
import math
import sys
from collections import defaultdict

from e2e_common import ROOT, prf
from e2e_records import load_records
from e2e_stats import paired_bootstrap
from run_e2e import gold_units, to_units

ART = ROOT / "artifacts"
VARIANTS = ("FedResolve-noResidual", "FedResolve-noEarlyStop", "FedResolve-noAvailability", "FedResolve-noSlots")
FIELDS = ("selected_clients", "abstain_routing", "evidence_page_uids")


def rows(path, keep):
    out = {}
    with open(path, encoding="utf-8") as h:
        for line in h:
            r = json.loads(line)
            if keep(r):
                out[(r["config"], r["method"], r["budget"], r["scenario"], r["uid"])] = r
    return out


result = {"checks": {}, "cells": [], "paired": []}
for ds in ("finrag", "visdom"):
    new = rows(ART / "serf_v3_ablation" / f"{ds}_test" / "plan.jsonl", lambda r: True)
    rep = rows(ART / "public_e2e_v1" / f"{ds}_slot" / "plan.jsonl", lambda r: r["method"] == "S2G-RAG" and r["budget"] == 2 and r["scenario"] in ("full", "qrel_missing"))
    v3 = rows(ART / "serf_v3" / f"{ds}_test" / "plan.jsonl", lambda r: r["method"] == "FedResolve-v3" and r["budget"] == 2 and r["scenario"] in ("full", "qrel_missing"))
    abl = rows(ART / "serf_v3_ablation" / f"{ds}_test" / "plan.jsonl", lambda r: r["method"] in VARIANTS and r["budget"] == 2 and r["scenario"] in ("full", "qrel_missing"))
    chk = {"s2g_checked": 0, "s2g_mismatch": 0, "v3_checked": 0, "v3_mismatch": 0, "variant_checked": 0, "variant_routing_mismatch": 0, "missing": 0}
    for k, r in new.items():
        m = k[1]
        if m == "S2G-RAG":
            o = rep.get(k)
            chk["s2g_checked"] += 1
            if o is None:
                chk["missing"] += 1
            elif any(r[f] != o[f] for f in FIELDS):
                chk["s2g_mismatch"] += 1
        elif m == "FedResolve-v3":
            o = v3.get(k)
            chk["v3_checked"] += 1
            if o is None:
                chk["missing"] += 1
            elif any(r[f] != o[f] for f in FIELDS):
                chk["v3_mismatch"] += 1
        else:
            o = abl.get(k)
            chk["variant_checked"] += 1
            if o is None:
                chk["missing"] += 1
            elif r["selected_clients"] != o["selected_clients"] or r["abstain_routing"] != o["abstain_routing"]:
                chk["variant_routing_mismatch"] += 1
    result["checks"][ds] = chk
    records, _, _ = load_records(ds)
    unit = records[0].gold_unit
    gold = {r.uid: gold_units(r, unit) for r in records}
    idx = defaultdict(dict)
    for (cfg, m, b, sc, uid), r in new.items():
        if sc == "full":
            ev = prf(to_units(r["evidence_page_uids"], unit), gold[uid])
            if not math.isnan(ev["recall"]):
                idx[(cfg, m, "full")][uid] = (ev["recall"], float(r["cost"]["client_calls"]))
        else:
            idx[(cfg, m, "qrel_missing")][uid] = (1.0 if r["abstain_routing"] else 0.0,)
    for (cfg, m, sc), d in sorted(idx.items()):
        v = list(d.values())
        cell = {"dataset": ds, "config": cfg, "method": m, "scenario": sc, "n": len(v)}
        if sc == "full":
            cell.update({"evidence_recall": sum(x[0] for x in v) / len(v), "calls": sum(x[1] for x in v) / len(v)})
        else:
            cell["routing_abstention"] = sum(x[0] for x in v) / len(v)
        result["cells"].append(cell)
    for cfg in sorted({k[0] for k in idx}):
        t = idx[(cfg, "FedResolve-v3", "full")]
        for m in VARIANTS:
            c = idx[(cfg, m, "full")]
            uids = sorted(set(t) & set(c))
            for j, name in ((0, "evidence_recall"), (1, "calls")):
                result["paired"].append({"dataset": ds, "config": cfg, "variant": m, "metric": name,
                                         **paired_bootstrap([t[u][j] for u in uids], [c[u][j] for u in uids])})
json.dump(result, sys.stdout)
sys.stdout.write("\n")
