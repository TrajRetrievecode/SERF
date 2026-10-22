#!/usr/bin/env python3
"""Step 4 of protocol_serf_v3_devconfirm.json: C1, C2, D1 to D3, D5 (evidence level, no generation).

Before any plan row is read: the protocol, plans, confirmation lists, and gold labels are checked
against the registered hashes and digests. Per query it computes evidence recall (also as an exact
fraction hits/|gold units|) and precision with the scorer's functions for scenario 'full' at budgets
1 to 3. Integrity checks (settings, registered counts per setting and method, duplicates, gold units
of every confirmation query, pinned modules) run before anything is written or printed.

C1/C2/D1 pool all (setting, query) pairs of the confirmation population with equal weight and use a
cluster bootstrap over family components (confirmation queries linked through shared document
families, listed in configs/serf_v3_devconfirm/confirm_uids_<dataset>.json), 10,000 resamples, numpy default_rng(20260805), percentile interval
rounded to 12 decimals. The per-setting sign and the leave-one-component-out estimates are exact
rationals. D2/D3 use e2e_stats.paired_bootstrap per setting. The result is written atomically to
artifacts/serf_v3_devconfirm/evidence_dev.json (temporary file, fsync, hard link without replacement).
"""
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from fractions import Fraction

import numpy as np

import pincheck
from digest_util import gold_digest
from e2e_common import ROOT, prf
from e2e_records import load_records
from e2e_stats import paired_bootstrap
from run_e2e import gold_units, to_units

TREAT = "FedResolve-v3"
CONTROLS = ("FedResolve", "S2G-RAG", "RAGRoute", "Centralized")
METHODS = (TREAT,) + CONTROLS
BUDGETS = (1, 2, 3)
CFGS = ("c16_seen_query", "c16_unseen_family", "c8_seen_query", "c8_unseen_family")
ITER, SEED = 10000, 20260805
C = ROOT / "configs" / "serf_v3_devconfirm"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cluster_bootstrap(pairs):
    """pairs: list of (cluster, difference). Resample clusters; every pair of a drawn cluster enters."""
    by = defaultdict(list)
    for k, d in pairs:
        by[k].append(float(d))
    ks = sorted(by)
    s = np.array([sum(by[k]) for k in ks], dtype=np.float64)
    c = np.array([len(by[k]) for k in ks], dtype=np.float64)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(ks), size=(ITER, len(ks)))
    draws = s[idx].sum(axis=1) / c[idx].sum(axis=1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {"estimate": round(float(s.sum() / c.sum()), 12), "ci_low": round(float(lo), 12), "ci_high": round(float(hi), 12),
            "pairs": int(c.sum()), "clusters": len(ks), "iterations": ITER, "seed": SEED}


def main():
    pincheck.env_check("evidence")
    protocol_path = sys.argv[1]
    protocol = json.load(open(protocol_path, encoding="utf-8"))
    out_dir = ROOT / "artifacts" / "serf_v3_devconfirm"
    out = out_dir / "evidence_dev.json"
    if out.exists():
        raise SystemExit("evidence_dev.json exists; the evidence stage runs once")
    result = {"protocol_name": protocol["name"], "protocol_sha256": sha256(protocol_path),
              "manifest_sha256": sha256(C / "SHA256SUMS"), "environment": pincheck.environment(), "datasets": {}}
    vals, comp, disjoint = {}, {}, {}
    for ds, spec in protocol["data"]["plans"].items():
        plan = ROOT / spec["path"]
        if sha256(plan) != spec["sha256"]:
            raise SystemExit(f"{ds}: plan sha256 differs from the protocol")
        cp = protocol["data"]["confirmation_population"][ds]
        if sha256(ROOT / cp["path"]) != cp["sha256"]:
            raise SystemExit(f"{ds}: confirmation list sha256 differs from the protocol")
        registered = json.load(open(C / f"gold_digest_{ds}.json", encoding="utf-8"))
        if gold_digest(ds) != registered:
            raise SystemExit(f"{ds}: gold labels differ from the registered digest")
        cl = json.load(open(ROOT / cp["path"], encoding="utf-8"))
        keep = {cfg: set(v) for cfg, v in cl["keep"].items()}
        if set(keep) != set(CFGS) or any(len(keep[c]) != cp["per_setting"][c] for c in CFGS):
            raise SystemExit(f"{ds}: confirmation list does not match the registered counts")
        comp[ds] = cl["component"]
        disjoint[ds] = set(cl["family_disjoint_queries"])
        records, _, _ = load_records(ds)
        unit = records[0].gold_unit
        gold = {r.uid: set(gold_units(r, unit)) for r in records}
        seen = defaultdict(set)
        dev = defaultdict(dict)
        with plan.open(encoding="utf-8") as h:
            for line in h:
                r = json.loads(line)
                if r["scenario"] != "full" or r["method"] not in METHODS:
                    continue
                key = (r["config"], r["method"], r["budget"])
                if r["uid"] in seen[key]:
                    raise SystemExit(f"{ds}: duplicate row {key}")
                seen[key].add(r["uid"])
                g = gold[r["uid"]]
                if not g:
                    continue
                units = to_units(r["evidence_page_uids"], unit)
                ev = prf(units, g)
                exact = Fraction(len(set(units) & g), len(g))
                if math.isnan(ev["recall"]) or abs(float(exact) - ev["recall"]) > 1e-12:
                    raise SystemExit(f"{ds}: recall mismatch")
                dev[key][r["uid"]] = (exact, ev["precision"], float(len(r["evidence_page_uids"])), float(len(set(units))))
        nogold = {c: sorted(u for u in seen[(c, TREAT, 2)] if not gold[u]) for c in CFGS}
        for cfg in CFGS:
            for m in METHODS:
                for b in BUDGETS:
                    got = seen.get((cfg, m, b), set())
                    if len(got) != spec["dev_queries_per_setting"][cfg] or got != seen[(cfg, TREAT, b)]:
                        raise SystemExit(f"{ds} {cfg} {m} B={b}: rows differ from the registered development queries")
            if not keep[cfg] <= seen[(cfg, TREAT, 2)]:
                raise SystemExit(f"{ds} {cfg}: confirmation queries missing from the plan")
            if keep[cfg] & set(nogold[cfg]):
                raise SystemExit(f"{ds} {cfg}: a confirmation query has no gold units (run invalid)")
        conf = {k: {u: v for u, v in d.items() if u in keep[k[0]]} for k, d in dev.items()}
        vals[(ds, "confirmation")] = conf
        vals[(ds, "development_unrestricted")] = dict(dev)
        result["datasets"][ds] = {"development_queries_without_gold_units": {c: len(nogold[c]) for c in CFGS},
                                  "confirmation_pairs": sum(len(conf[(c, TREAT, 2)]) for c in CFGS),
                                  "gold_digest": registered}

    for (ds, pop), d in vals.items():
        cells, paired = [], []
        for (cfg, m, b), recs in sorted(d.items()):
            v = list(recs.values())
            cells.append({"config": cfg, "method": m, "budget": b, "n": len(v),
                          "evidence_recall": float(sum(x[0] for x in v)) / len(v), "evidence_precision": sum(x[1] for x in v) / len(v),
                          "pages": sum(x[2] for x in v) / len(v), "distinct_units": sum(x[3] for x in v) / len(v)})
        for cfg in CFGS:
            for b in BUDGETS:
                t = d[(cfg, TREAT, b)]
                uids = sorted(t)
                for c in CONTROLS:
                    ctrl = d[(cfg, c, b)]
                    for j, k in ((0, "evidence_recall"), (1, "evidence_precision")):
                        paired.append({"config": cfg, "budget": b, "control": c, "metric": k,
                                       **paired_bootstrap([float(t[u][j]) for u in uids], [float(ctrl[u][j]) for u in uids])})
        result["datasets"][ds][pop] = {"cells": cells, "paired_per_setting": paired}

    def pooled(ds, control, subset=None):
        d = vals[(ds, "confirmation")]
        pairs, exact, per = [], [], {}
        for cfg in CFGS:
            t, c = d[(cfg, TREAT, 2)], d[(cfg, control, 2)]
            diffs = [(u, t[u][0] - c[u][0]) for u in sorted(t) if subset is None or u in subset]
            if diffs:
                per[cfg] = sum(x for _, x in diffs) / len(diffs)
            exact += diffs
            pairs += [(comp[ds][u], x) for u, x in diffs]
        if not pairs:
            return {"pairs": 0}
        res = {**cluster_bootstrap(pairs), "estimate_exact_sign": (lambda s: (s > 0) - (s < 0))(sum(x for _, x in exact)),
               "per_setting_estimate": {k: float(v) for k, v in per.items()},
               "settings_with_positive_estimate": sum(1 for v in per.values() if v > 0)}
        if subset is None:
            res["query_cluster_interval"] = cluster_bootstrap([(u, x) for u, x in exact])
        sizes = defaultdict(int)
        for k, _ in pairs:
            sizes[k] += 1
        top = sorted(sizes, key=lambda k: (-sizes[k], k))[:2]
        res["leave_out"] = []
        for k in top:
            rest = [x for (kk, x) in pairs if kk != k]
            if not rest:
                res["leave_out"].append({"component": k, "pairs_removed": sizes[k], "estimate": None, "positive": False})
                continue
            m = sum(rest, Fraction(0)) / len(rest)
            res["leave_out"].append({"component": k, "pairs_removed": sizes[k], "estimate": float(m), "positive": m > 0})
        return res

    c1 = pooled("finrag", "FedResolve")
    c1["confirmed"] = bool(c1["ci_low"] > 0 and c1["settings_with_positive_estimate"] >= 3 and all(x["positive"] for x in c1["leave_out"]))
    c2 = pooled("finrag", "S2G-RAG")
    c2["rule_met"] = bool(c2["ci_low"] > 0 and c2["settings_with_positive_estimate"] >= 3 and all(x["positive"] for x in c2["leave_out"]))
    c2["confirmed"] = bool(c2["rule_met"] and c1["confirmed"])
    d1 = pooled("visdom", "FedResolve")
    vis = [p for p in result["datasets"]["visdom"]["confirmation"]["paired_per_setting"]
           if p["budget"] == 2 and p["control"] == "FedResolve" and p["metric"] == "evidence_recall"]
    d1["settings_with_interval_below_0"] = sum(1 for p in vis if p["ci_high"] < 0)
    d5 = {"family_disjoint": {ds: pooled(ds, "FedResolve", subset=disjoint[ds]) for ds in ("finrag", "visdom")},
          "full_population_query_cluster_interval": {"finrag": c1["query_cluster_interval"], "visdom": d1["query_cluster_interval"]}}
    result["C1"], result["C2"], result["D1"], result["D5"] = c1, c2, d1, d5
    result["decision"] = "SERF-v3 may become the main method" if c1["confirmed"] else "SERF-v1 stays the main method"
    pincheck.check("evidence")
    text = json.dumps(result, indent=1, default=str) + "\n"
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / f".evidence_dev.{os.getpid()}.tmp"
    with open(tmp, "x", encoding="utf-8") as h:
        h.write(text)
        h.flush()
        os.fsync(h.fileno())
    os.link(tmp, out)
    tmp.unlink()
    print(json.dumps({"C1": {k: c1[k] for k in ("estimate", "ci_low", "ci_high", "settings_with_positive_estimate", "confirmed")},
                      "C1_leave_out": c1["leave_out"],
                      "C2": {k: c2[k] for k in ("estimate", "ci_low", "ci_high", "confirmed")},
                      "D1": {k: d1[k] for k in ("estimate", "ci_low", "ci_high", "settings_with_interval_below_0")},
                      "decision": result["decision"]}, default=str))


if __name__ == "__main__":
    main()
