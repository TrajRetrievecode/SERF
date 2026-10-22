#!/usr/bin/env python3
"""Read-only, descriptive analysis of the SERF component ablation by number of sub-questions.

Run with the PYTHONPATH in README.md; reads the recorded
plan of the ablation run (artifacts/serf_v3_ablation/<ds>_test/plan.jsonl, budget 2, every client
online), the query records and the sub-question manifest; prints one JSON document; writes nothing.

For each dataset, each setting and each stratum (queries with one sub-question, queries with two),
it gives the number of queries, the mean evidence recall and the mean calls of SERF (FedResolve-v3)
and of the variant without sub-questions (FedResolve-noSlots), and the paired bootstrap of SERF
minus the variant on both (10,000 resamples, seed 20260805, the settings of the paper). It also
pools the four settings (mean of the setting means) and counts the settings in which the
difference is significant. No new run and no new decision: the recorded pages and calls are
re-read and split by a query property that was fixed before any run (the sub-question
decomposition of the reported run).
"""
import json
import math
import sys
from collections import defaultdict

from e2e_common import ROOT, prf, read_json
from e2e_corpus import CACHE
from e2e_records import load_records
from e2e_stats import paired_bootstrap
from run_e2e import gold_units, to_units

ART = ROOT / "artifacts" / "serf_v3_ablation"
SERF, VARIANT = "FedResolve-v3", "FedResolve-noSlots"


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


out = {"description": __doc__.strip(), "datasets": {}}
for ds in ("finrag", "visdom"):
    records, _, _ = load_records(ds)
    unit = records[0].gold_unit
    gold = {r.uid: gold_units(r, unit) for r in records}
    spans = read_json(CACHE / f"{ds}_slot_queries" / "manifest.json")["spans"]
    nsub = {r.uid: int(spans[i][1]) - int(spans[i][0]) for i, r in enumerate(records)}
    idx = defaultdict(dict)  # (config, method) -> uid -> (recall, calls)
    with open(ART / f"{ds}_test" / "plan.jsonl", encoding="utf-8") as h:
        for line in h:
            r = json.loads(line)
            if r["budget"] != 2 or r["scenario"] != "full" or r["method"] not in (SERF, VARIANT):
                continue
            ev = prf(to_units(r["evidence_page_uids"], unit), gold[r["uid"]])
            if math.isnan(ev["recall"]):
                continue
            idx[(r["config"], r["method"])][r["uid"]] = (ev["recall"], float(r["cost"]["client_calls"]))
    configs = sorted({c for c, _ in idx})
    rows, pooled = [], {}
    for k in (1, 2):
        acc = defaultdict(list)
        for cfg in configs:
            t, c = idx[(cfg, SERF)], idx[(cfg, VARIANT)]
            uids = sorted(u for u in set(t) & set(c) if nsub[u] == k)
            row = {"config": cfg, "subquestions": k, "n": len(uids),
                   "serf_recall": mean(t[u][0] for u in uids), "variant_recall": mean(c[u][0] for u in uids),
                   "serf_calls": mean(t[u][1] for u in uids), "variant_calls": mean(c[u][1] for u in uids),
                   "recall": paired_bootstrap([t[u][0] for u in uids], [c[u][0] for u in uids]),
                   "calls": paired_bootstrap([t[u][1] for u in uids], [c[u][1] for u in uids])}
            rows.append(row)
            for key in ("n", "serf_recall", "variant_recall", "serf_calls", "variant_calls"):
                acc[key].append(row[key])
            for m in ("recall", "calls"):
                acc[m + "_diff"].append(row[m]["difference"])
                acc[m + "_up"].append(row[m]["significant_at_95"] and row[m]["difference"] > 0)
                acc[m + "_down"].append(row[m]["significant_at_95"] and row[m]["difference"] < 0)
        pooled[str(k)] = {"n_total": sum(acc["n"]), "serf_recall": mean(acc["serf_recall"]), "variant_recall": mean(acc["variant_recall"]),
                          "serf_calls": mean(acc["serf_calls"]), "variant_calls": mean(acc["variant_calls"]),
                          "recall_diff": mean(acc["recall_diff"]), "recall_sig_up": sum(acc["recall_up"]), "recall_sig_down": sum(acc["recall_down"]),
                          "calls_diff": mean(acc["calls_diff"]), "calls_sig_up": sum(acc["calls_up"]), "calls_sig_down": sum(acc["calls_down"])}
    out["datasets"][ds] = {"by_setting": rows, "pooled": pooled}
json.dump(out, sys.stdout)
sys.stdout.write("\n")
