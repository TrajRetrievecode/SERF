#!/usr/bin/env python3
"""Component ablation of SERF with the SERF-v3 evidence selection (protocol_v3_ablation.json).

Plans the test portions of both datasets at budget 2, scenarios full and qrel_missing, for SERF-v3 and
for the four routing ablations of the reported ablation run (no residual update, no early stop, no
availability test, no sub-questions), each followed by the SERF-v3 evidence selection. It reuses
run_v3_plan.py unchanged: the ablation names keep their original method labels (so the no-sub-question
variant receives the single-query view as before), but ADAPTERS maps each of them to the routing
function followed by the v3 selection below, a copy of e2e_v3_adapter.fedresolve_v3_adapter whose
abstention is the routing function's own decision or an empty selection (identical to the v3 rule for
SERF, whose routing abstains exactly when the availability test fires or no page is available).
Writes only to artifacts/serf_v3_ablation. CPU only; no generation.
"""
import numpy as np

import run_v3_plan
from e2e_common import ROOT
from e2e_slot_adapters import ABLATIONS


def v3_select(base, views, budget, protocol):
    called = base["clients"]
    pools = [views.rows_by_client[c] for c in called if c in views.rows_by_client and views.rows_by_client[c].size]
    chosen = []
    if pools and budget > 0:
        rows = np.concatenate(pools).astype(np.int64)
        w = float(protocol["fedresolve"]["claim_query_weight"])
        wf = float(protocol["fedresolve"]["full_query_weight"])
        z = w * views.slot_page[:, rows].astype(np.float64) + wf * views.full_page[rows].astype(np.float64)[None, :]
        e = views.evidence_score[rows]
        alive = np.ones(rows.size, dtype=bool)
        k = int(np.lexsort((-rows, e))[-1])
        chosen.append(int(rows[k]))
        alive[k] = False
        best = z[:, k].copy()
        while alive.any() and len(chosen) < budget:
            j = int(np.argmin(best))
            idx = np.flatnonzero(alive)
            k = int(idx[np.lexsort((-rows[idx], e[idx], z[j, idx]))[-1]])
            chosen.append(int(rows[k]))
            alive[k] = False
            best = np.maximum(best, z[:, k])
        chosen.sort(key=lambda r: (-float(views.evidence_score[r]), r))
    return {"clients": called, "rows": chosen, "abstain": bool(base["abstain"]) or not chosen,
            "trace": {**base.get("trace", {}), "evidence_selection": "serf-v3 coverage over called clients' pages"}}


def v3_wrap(base_fn):
    def adapter(views, online, budget, protocol, **kwargs):
        return v3_select(base_fn(views, online, budget, protocol, **kwargs), views, budget, protocol)
    return adapter


run_v3_plan.ADAPTERS = {**run_v3_plan.ADAPTERS, **{name: v3_wrap(fn) for name, fn in ABLATIONS.items()}}
run_v3_plan.ARTIFACTS = ROOT / "artifacts" / "serf_v3_ablation"

if __name__ == "__main__":
    raise SystemExit(run_v3_plan.main())
