#!/usr/bin/env python3
"""Aggregate the SERF-v3 public test run (artifacts/serf_v3/<dataset>_test/predictions.jsonl).

evidence_success = a relevant unit among the cited pages and no system abstention (the protocol's
evidence criterion); retrieval_hit = a relevant unit among the cited pages (evidence_recall > 0),
independent of the generator.
"""
import csv
import json
from collections import defaultdict

import numpy as np

from e2e_common import ROOT
from e2e_stats import mcnemar, paired_bootstrap

ART = ROOT / "artifacts" / "serf_v3"
METRICS = ("evidence_recall", "evidence_precision", "evidence_f1", "citation_f1", "answer_token_f1",
           "answer_exact_match", "selected_pages", "client_calls", "protocol_bytes")
BOOT = ("evidence_recall", "evidence_precision", "evidence_f1", "citation_f1", "answer_token_f1",
        "client_calls", "protocol_bytes")
PAIRS = (("FedResolve-v3", "FedResolve"), ("FedResolve-v3", "S2G-RAG"), ("FedResolve-v3", "RAGRoute"),
         ("FedResolve-v3", "Centralized"), ("FedResolve", "S2G-RAG"))
MISS = ("FedResolve-v3", "FedResolve", "S2G-RAG", "S2G-RAG+avail", "RAGRoute", "Centralized", "All", "Oracle-client")


def load(ds):
    d = ART / f"{ds}_test"
    fail = d / "failures.jsonl"
    if fail.exists() and fail.stat().st_size > 0:
        raise SystemExit(f"{ds}: rows without generation in failures.jsonl")
    rows = []
    with (d / "predictions.jsonl").open(encoding="utf-8") as h:
        for line in h:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def f(v):
    return float(v) if v is not None else float("nan")


def flag(r, k):
    return 1.0 if r[k] else 0.0


def hit(r):
    return 1.0 if f(r.get("evidence_recall")) > 0 else 0.0


def main() -> int:
    cells, stats, miss = [], [], []
    for ds in ("finrag", "visdom"):
        idx = defaultdict(dict)
        for r in load(ds):
            idx[(r["config"], r["method"], r["budget"], r["scenario"])][r["uid"]] = r
        for (cfg, m, b, sc), recs in sorted(idx.items()):
            vals = list(recs.values())
            cells.append({"dataset": ds, "config": cfg, "method": m, "budget": b, "scenario": sc, "n": len(vals),
                          **{k: float(np.nanmean([f(v.get(k)) for v in vals])) for k in METRICS},
                          "abstain_system": float(np.mean([flag(v, "abstain_system") for v in vals])),
                          "abstain_routing": float(np.mean([flag(v, "abstain_routing") for v in vals])),
                          "evidence_success": float(np.mean([flag(v, "evidence_correct") for v in vals])),
                          "answer_correct": float(np.mean([flag(v, "answer_correct") for v in vals])),
                          "retrieval_hit": float(np.mean([hit(v) for v in vals]))})
        configs = sorted({k[0] for k in idx})
        for cfg in configs:
            for b in (1, 2, 3):
                for t, c in PAIRS:
                    A, B = idx.get((cfg, t, b, "full"), {}), idx.get((cfg, c, b, "full"), {})
                    uids = sorted(set(A) & set(B))
                    if not uids:
                        continue
                    for k in BOOT:
                        stats.append({"dataset": ds, "config": cfg, "budget": b, "treatment": t, "control": c,
                                      "metric": k, **paired_bootstrap([f(A[u][k]) for u in uids], [f(B[u][k]) for u in uids])})
                    for name, fn in (("retrieval_hit_mcnemar", hit), ("evidence_success_mcnemar", lambda r: flag(r, "evidence_correct"))):
                        stats.append({"dataset": ds, "config": cfg, "budget": b, "treatment": t, "control": c,
                                      "metric": name, **mcnemar([fn(A[u]) for u in uids], [fn(B[u]) for u in uids])})
            for m in MISS:
                full = idx.get((cfg, m, 2, "full"), {})
                una = idx.get((cfg, m, 2, "qrel_missing"), {})
                ans = idx.get((cfg, m, 2, "unneeded_offline"), {})
                for kind, side in (("proxy", full), ("paired", ans)):
                    uids = sorted(set(side) & set(una))
                    if not uids:
                        continue
                    ae = np.array([flag(side[u], "evidence_correct") for u in uids])
                    aa = np.array([flag(side[u], "answer_correct") for u in uids])
                    us = np.array([flag(una[u], "abstain_system") for u in uids])
                    ur = np.array([flag(una[u], "abstain_routing") for u in uids])
                    fs = np.array([flag(side[u], "abstain_system") for u in uids])
                    fr = np.array([flag(side[u], "abstain_routing") for u in uids])
                    wrong = np.array([0.0 if una[u]["abstain_system"] or una[u]["answer_correct"] else 1.0 for u in uids])
                    miss.append({"dataset": ds, "config": cfg, "method": m, "kind": kind, "n": len(uids),
                                 "answerable_success_evidence": float(ae.mean()), "answerable_success_answer": float(aa.mean()),
                                 "unanswerable_abstain_system": float(us.mean()), "unanswerable_abstain_routing": float(ur.mean()),
                                 "false_abstention_system": float(fs.mean()), "false_abstention_routing": float(fr.mean()),
                                 "wrong_answer_when_unanswerable": float(wrong.mean()),
                                 "balanced_evidence": float(0.5 * (ae.mean() + us.mean())),
                                 "balanced_answer": float(0.5 * (aa.mean() + us.mean()))})

    def write(name, rows):
        with (ART / name).open("w", encoding="utf-8", newline="") as h:
            w = csv.DictWriter(h, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    write("v3_cells.csv", cells)
    write("v3_missingness.csv", miss)
    (ART / "v3_paired_statistics.json").write_text(json.dumps(stats, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"cells": len(cells), "stats": len(stats), "missingness_rows": len(miss)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
