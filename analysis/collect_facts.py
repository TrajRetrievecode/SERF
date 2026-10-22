#!/usr/bin/env python3
"""Collect the fact blocks for Figures 2, 3, and 4 from package artifacts. Writes artifacts/analysis/collected_facts.json."""
import csv
import json
from collections import defaultdict
from pathlib import Path
import os

ROOT = Path(os.environ.get("SERF_ROOT", str(Path(__file__).resolve().parents[1])))
out = {}


def rel(a, b):
    return round(100 * (a / b - 1), 1)


cells = list(csv.DictReader(open(ROOT / "artifacts/serf_v3/v3_cells.csv", encoding="utf-8")))
miss = list(csv.DictReader(open(ROOT / "artifacts/serf_v3/v3_missingness.csv", encoding="utf-8")))
re_ = json.load(open(ROOT / "artifacts/analysis/reanalysis_results.json", encoding="utf-8"))
SERF = "FedResolve-v3"


def cmean(ds, m, b, k):
    r = [float(c[k]) for c in cells if c["dataset"] == ds and c["method"] == m and c["budget"] == str(b) and c["scenario"] == "full"]
    assert len(r) == 4, (ds, m, b, k, len(r))
    return sum(r) / 4


def mmean(ds, m, kind, k):
    r = [float(x[k]) for x in miss if x["dataset"] == ds and x["method"] == m and x["kind"] == kind]
    assert len(r) == 4, (ds, m, kind, k, len(r))
    return sum(r) / 4


for ds in ("finrag", "visdom"):
    o = {}
    # the availability test added to each method (missingness proxy, evidence criterion, budget 2)
    acc = defaultdict(list)
    for r in re_["D_method_plus_test"][ds]:
        acc[r["method"]].append(r["balanced"])
    met0 = json.load(open(ROOT / f"artifacts/public_e2e_v1/{ds}_slot/metrics.json", encoding="utf-8"))
    for r in met0["missingness_proxy"]:
        if r["budget"] == 2:
            acc[r["method"]].append(r["balanced_missingness_accuracy_evidence"])
    assert all(len(v) == 4 for v in acc.values()), {m: len(v) for m, v in acc.items()}
    o["proxy_balanced_by_method"] = {m: sum(v) / len(v) for m, v in acc.items()}
    pb = o["proxy_balanced_by_method"]
    gains = {}
    for m in list(pb):
        if m.endswith("+test") and m[:-5] in pb:
            gains[m[:-5]] = rel(pb[m], pb[m[:-5]])
    o["gain_from_test_pct"] = gains
    serf_proxy = mmean(ds, SERF, "proxy", "balanced_evidence")
    serf_paired = mmean(ds, SERF, "paired", "balanced_evidence")
    o["serf_balanced"] = {"proxy": serf_proxy, "paired": serf_paired}
    o["serf_vs_s2g_pct"] = {"proxy": rel(serf_proxy, mmean(ds, "S2G-RAG", "proxy", "balanced_evidence")),
                            "paired": rel(serf_paired, mmean(ds, "S2G-RAG", "paired", "balanced_evidence"))}
    o["serf_vs_s2g_avail_pct"] = {"proxy": rel(serf_proxy, mmean(ds, "S2G-RAG+avail", "proxy", "balanced_evidence")),
                                  "paired": rel(serf_paired, mmean(ds, "S2G-RAG+avail", "paired", "balanced_evidence"))}
    ext = {m: v for m, v in pb.items() if m in ("RAGRoute", "UniversalRAG", "CoRe-MMRAG", "S2G-RAG")}
    best = max(ext, key=ext.get)
    o["best_external_without_test"] = [best, ext[best], rel(serf_proxy, ext[best])]
    # calls
    o["calls"] = {str(b): {"SERF": cmean(ds, SERF, b, "client_calls"), "fewer_than_budget_pct": round(100 * (1 - cmean(ds, SERF, b, "client_calls") / b), 1)} for b in (1, 2, 3)}
    o["early_stop_removed_calls_more_pct"] = rel(2.0, cmean(ds, SERF, 2, "client_calls"))
    # budget sweep values for the figure
    o["budget_sweep"] = {}
    met = json.load(open(ROOT / f"artifacts/public_e2e_v1/{ds}_slot/metrics.json", encoding="utf-8"))
    by = defaultdict(list)
    for c in met["cells"]:
        if c.get("scenario", "full") == "full":
            by[(c["method"], c["budget"])].append(c)
    for (m, b), v in sorted(by.items()):
        if len(v) == 4:
            o["budget_sweep"].setdefault(m, {})[str(b)] = {"evidence_recall": sum(x["evidence_recall"] for x in v) / 4,
                                                           "client_calls": sum(x["client_calls"] for x in v) / 4}
    o["budget_sweep"]["SERF"] = {str(b): {"evidence_recall": cmean(ds, SERF, b, "evidence_recall"), "client_calls": cmean(ds, SERF, b, "client_calls")} for b in (1, 2, 3)}
    out[ds] = o

# Every single-client-offline scenario of the evaluation set (Figure 2):
# counts and balanced missingness accuracy of SERF, of the external methods without and with the
# availability test, and of the oracle client set, from the registered enumeration
_EN = json.load(open(ROOT / "artifacts/enumeration/enumeration_summary.json",
                     encoding="utf-8"))
_ext = ("S2G-RAG", "RAGRoute", "UniversalRAG", "CoRe-MMRAG")
_rows = ("FedResolve", *_ext, *(m + "+avail" for m in _ext), "Oracle client set")
out["enumeration"] = {
    "scenarios": {"answerable": _EN["all"]["FedResolve"]["answerable"],
                  "unanswerable": _EN["all"]["FedResolve"]["unanswerable"],
                  "unanswerable_outside_paired_set": _EN["new_only"]["FedResolve"]["unanswerable"]},
    "answered": {m: _EN["all"][m]["answerable_success"] for m in _rows},
    "unanswerable_abstain": {m: _EN["all"][m]["unanswerable_abstain"] for m in _rows},
    "unanswerable_abstain_outside_paired_set": {m: _EN["new_only"][m]["unanswerable_abstain"] for m in _rows},
    "balanced": {m: round(_EN["all"][m]["balanced"], 3) for m in _rows},
    "external_with_test_balanced_range_pct": [round(100 * min(_EN["all"][m + "+avail"]["balanced"] for m in _ext), 1),
                                              round(100 * max(_EN["all"][m + "+avail"]["balanced"] for m in _ext), 1)],
}
print("enumeration", json.dumps(out["enumeration"]["external_with_test_balanced_range_pct"]))

# availability test on the same enumeration: signed lead of the offline client over the best online
# client per scenario, and the balanced missingness accuracy as a function of the margin, from the
# read-only analysis scripts/enumeration/margins_readonly.py (Figure 3)
import csv as _csv
_GE = ROOT / "artifacts/enumeration"
_MS = json.load(open(_GE / "margins_summary.json", encoding="utf-8"))
_scen = _MS["scenarios_list"]
_best = {}
for _r in _csv.DictReader(open(_GE / "margins_by_scenario.csv", encoding="utf-8")):
    _k = (_r["canonical_id"], int(_r["offline_client"]))
    _best.setdefault(_k, -1.0)
    if _r["offline_client_is_best"] == "True" and float(_r["probability_ratio"]) >= _MS["probability_ratio"]:
        _best[_k] = max(_best[_k], float(_r["lead"]))
_ans = [(s["canonical_id"], s["offline_client"]) for s in _scen if s["answerable"]]
_una = [(s["canonical_id"], s["offline_client"]) for s in _scen if not s["answerable"]]
_curve = []
for _i in range(0, 61):
    _e = round(0.005 * _i, 3)
    _a = sum(1 for k in _una if _best.get(k, -1.0) >= _e)
    _f = sum(1 for k in _ans if _best.get(k, -1.0) >= _e)
    _curve.append({"margin": _e, "abstain": _a, "false_abstain": _f,
                   "balanced": round(0.5 * ((len(_ans) - _f) / len(_ans) + _a / len(_una)), 4)})
_bal = {c["margin"]: c["balanced"] for c in _curve}
_m = _MS["margin"]
out["availability_margin"] = {
    "scenarios": len(_scen), "answerable": len(_ans), "unanswerable": len(_una),
    "margin": _m, "probability_ratio": _MS["probability_ratio"],
    "answerable_lead_max": round(_MS["answerable_max_signed_lead_max"], 3),
    "answerable_lead_min": round(_MS["answerable_max_signed_lead_min"], 3),
    "unanswerable_lead_at_least_margin": sum(1 for s in _scen if not s["answerable"] and s["max_signed_lead"] >= _m),
    "unanswerable_lead_between_zero_and_margin": _MS["unanswerable_max_signed_lead_between_zero_and_margin"],
    "unanswerable_lead_negative": _MS["unanswerable_max_signed_lead_below_zero"],
    "unanswerable_test_fired": _MS["unanswerable_test_fired"], "answerable_test_fired": _MS["answerable_test_fired"],
    "margins_reported": [0.01, _m, 0.05],
    "balanced_pct_at": {"0.01": round(100 * _bal[0.01], 1), "0.02": round(100 * _bal[0.02], 1), "0.05": round(100 * _bal[0.05], 1)},
    "curve": _curve,
}
assert out["availability_margin"]["unanswerable_lead_at_least_margin"] == _MS["unanswerable_test_fired"]
print("availability margin", json.dumps(out["availability_margin"]["balanced_pct_at"]))

# evidence recall of Oracle client set at budget 2 with every client online, mean over the four
# settings (v3_cells.csv), cited in Section 4.2 to explain the VisDoM comparison
_cells = list(_csv.DictReader(open(ROOT / "artifacts/serf_v3/v3_cells.csv", encoding="utf-8")))
for _ds in ("finrag", "visdom"):
    _v = [float(r["evidence_recall"]) for r in _cells
          if r["dataset"] == _ds and r["method"] == "Oracle-client" and r["budget"] == "2" and r["scenario"] == "full"]
    assert len(_v) == 4, (_ds, len(_v))
    out[_ds]["oracle_client_recall_b2"] = round(sum(_v) / 4, 3)
print("oracle client recall b2", out["finrag"]["oracle_client_recall_b2"], out["visdom"]["oracle_client_recall_b2"])

# public retriever: channel weights and the page-text limit of the registered protocol (Appendix B)
_PUB = json.load(open(ROOT / "configs/public_e2e_v1/protocol_v3.json", encoding="utf-8"))
out["public_retriever"] = {
    "image_weight": _PUB["retriever"]["blend_weight"],
    "dense_weight": _PUB["retriever"]["dense_weight"],
    "page_text_char_limit": _PUB["evidence"]["page_text_char_limit"],
    "score": _PUB["retriever"]["score"],
}
print("public retriever", json.dumps(out["public_retriever"]))
# Owner contacts of Centralized from the control analysis "reference": the mean number of clients
# that hold its cited pages, printed in the Calls cells of the tables by format_tables.py.
own_path = ROOT / "artifacts/controls/reference/results/owner_contacts.csv"
if own_path.exists():
    own = list(csv.DictReader(open(own_path, encoding="utf-8-sig")))
    fair = {"owner_contacts": {}, "owner_contacts_by_setting_b2": {}}
    for r in own:
        if r["config"] in ("mean_over_settings", "c8_evaluation"):
            fair["owner_contacts"][f"{r['dataset']} B{r['budget']}"] = round(float(r["mean_owner_contacts"]), 3)
        elif r["budget"] == "2":
            fair["owner_contacts_by_setting_b2"][f"{r['dataset']} {r['config']}"] = round(float(r["mean_owner_contacts"]), 3)
    out["fairness"] = fair
    print("owner contacts", json.dumps(fair["owner_contacts"]))
json.dump(out, open(ROOT / "artifacts/analysis/collected_facts.json", "w", encoding="utf-8"), indent=1)
