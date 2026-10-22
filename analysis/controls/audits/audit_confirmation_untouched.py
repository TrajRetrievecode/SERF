"""Read-only C1, C2 and D1 analysis on confirmation queries untouched by prompt and retriever selection. Uses the evidence metrics and family-component bootstrap of dev_evidence.py. Prints JSON."""
import importlib.util
import json
import math
import sys
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

from e2e_common import ROOT as R
sys.path[:0] = [str(R / p) for p in ("scripts/serf_v3_devconfirm", "scripts/public_e2e_v1", "src/public_e2e_v1", "src", "scripts")]
from e2e_common import prf  # noqa: E402
from e2e_records import load_records  # noqa: E402
from fedresolve_core import split_records  # noqa: E402
from run_e2e import gold_units, to_units  # noqa: E402


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


de = load_module(R / "scripts/serf_v3_devconfirm/dev_evidence.py", "dev_evidence_mod")
sp = load_module(R / "scripts/public_e2e_v1/select_prompt.py", "sel_prompt")
sr = load_module(R / "scripts/public_e2e_v1/select_retriever_delivered.py", "sel_retr")
SEED = 20260728
CFGS = ("c16_seen_query", "c16_unseen_family", "c8_seen_query", "c8_unseen_family")
TREAT = "FedResolve-v3"
CONTROLS = ("FedResolve", "S2G-RAG", "RAGRoute", "Centralized")


def pooled(dev, comp, control, keep, subset):
    pairs, exact, per = [], [], {}
    for cfg in CFGS:
        t, c = dev[(cfg, TREAT, 2)], dev[(cfg, control, 2)]
        diffs = [(u, t[u][0] - c[u][0]) for u in sorted(t) if u in keep[cfg] and u in subset]
        if diffs:
            per[cfg] = float(sum(x for _, x in diffs) / len(diffs))
        exact += diffs
        pairs += [(comp[u], x) for u, x in diffs]
    if not pairs:
        return {"pairs": 0}
    res = {**de.cluster_bootstrap(pairs), "per_setting_estimate": per,
           "settings_with_positive_estimate": sum(1 for v in per.values() if v > 0)}
    sizes = defaultdict(int)
    for k, _ in pairs:
        sizes[k] += 1
    top = sorted(sizes, key=lambda k: (-sizes[k], k))[:2]
    res["leave_out"] = []
    for k in top:
        rest = [x for (kk, x) in pairs if kk != k]
        m = sum(rest, Fraction(0)) / len(rest) if rest else None
        res["leave_out"].append({"component": k, "pairs_removed": sizes[k], "estimate": None if m is None else float(m),
                                 "positive": bool(m is not None and m > 0)})
    res["rule_holds"] = bool(res["ci_low"] > 0 and res["settings_with_positive_estimate"] >= 3 and all(x["positive"] for x in res["leave_out"]))
    return res


out = {}
for ds in ("finrag", "visdom"):
    records, _, _ = load_records(ds)
    unit = records[0].gold_unit
    gold = {r.uid: set(gold_units(r, unit)) for r in records}
    cl = json.load(open(R / "configs/serf_v3_devconfirm" / f"confirm_uids_{ds}.json", encoding="utf-8"))
    keep = {cfg: set(v) for cfg, v in cl["keep"].items()}
    comp = cl["component"]
    disjoint = set(cl["family_disjoint_queries"])
    dev = defaultdict(dict)
    with (R / "artifacts/serf_v3" / f"{ds}_dev" / "plan.jsonl").open(encoding="utf-8") as h:
        for line in h:
            r = json.loads(line)
            if r["scenario"] != "full" or r["budget"] != 2 or r["method"] not in (TREAT,) + CONTROLS:
                continue
            g = gold[r["uid"]]
            if not g:
                continue
            units = to_units(r["evidence_page_uids"], unit)
            ev = prf(units, g)
            exact = Fraction(len(set(units) & g), len(g))
            if math.isnan(ev["recall"]) or abs(float(exact) - ev["recall"]) > 1e-12:
                raise SystemExit(f"{ds}: recall mismatch")
            dev[(r["config"], r["method"], 2)][r["uid"]] = (exact, ev["precision"])
    usable_prompt = list(sp.training_only_uids(records, ds, SEED))
    prompt_sample = set(usable_prompt[:150])
    retr_sample = {x.uid if hasattr(x, "uid") else x for x in sr.stratified_sample(records, list(sr.training_only(records, ds, SEED)), 400, SEED)}
    touched = prompt_sample | retr_sample
    all_conf = set().union(*keep.values())
    untouched = all_conf - touched
    res = {"confirmation_queries": len(all_conf), "touched_by_prompt_selection": len(all_conf & prompt_sample),
           "touched_by_retriever_selection": len(all_conf & retr_sample), "touched_by_either": len(all_conf & touched),
           "untouched_queries": len(untouched),
           "untouched_pairs": sum(len(keep[c] & untouched) for c in CFGS),
           "all_pairs": sum(len(keep[c]) for c in CFGS)}
    for control in CONTROLS:
        res[f"vs {control} all"] = pooled(dev, comp, control, keep, all_conf)
        res[f"vs {control} untouched"] = pooled(dev, comp, control, keep, untouched)
    res["vs FedResolve family_disjoint_untouched"] = pooled(dev, comp, "FedResolve", keep, disjoint & untouched)
    res["vs FedResolve family_disjoint_all"] = pooled(dev, comp, "FedResolve", keep, disjoint)
    out[ds] = res
print(json.dumps(out, indent=1, default=str))
