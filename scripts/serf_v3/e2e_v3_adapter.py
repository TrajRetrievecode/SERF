"""SERF-v3 adapter for the public pipeline (protocol_serf_v3.json, single_change and amendment_1).

Routing, the availability test and abstention come from the frozen SERF-v1 adapter; only the
evidence selection changes: pages may come several from one called client. The selected set is
presented to the generator ranked by the evidence score, as SERF-v1 presents its pages.
"""
import numpy as np

from e2e_slot_adapters import fedresolve_adapter


def fedresolve_v3_adapter(views, online, budget, protocol, **kwargs):
    base = fedresolve_adapter(views, online, budget, protocol, **kwargs)
    called = base["clients"]
    pools = [views.rows_by_client[c] for c in called
             if c in views.rows_by_client and views.rows_by_client[c].size]
    chosen = []
    if pools and budget > 0:
        rows = np.concatenate(pools).astype(np.int64)
        w = float(protocol["fedresolve"]["claim_query_weight"])
        wf = float(protocol["fedresolve"]["full_query_weight"])
        z = (w * views.slot_page[:, rows].astype(np.float64)
             + wf * views.full_page[rows].astype(np.float64)[None, :])
        e = views.evidence_score[rows]
        alive = np.ones(rows.size, dtype=bool)
        # first page: highest evidence score, ties -> lowest page index (lexsort: last key primary)
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
    critical = base["trace"]["critical_missing_slots"]
    return {
        "clients": called,
        "rows": chosen,
        "abstain": bool(critical) or not chosen,
        "trace": {**base["trace"], "evidence_selection": "serf-v3 coverage over called clients' pages"},
    }
