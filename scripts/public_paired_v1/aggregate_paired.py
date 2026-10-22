#!/usr/bin/env python3
"""public_paired_v1 aggregation.

Pairs each record's unneeded_offline (answerable) and qrel_missing (unanswerable)
outcomes at budget 2; writes per-setting tables, dataset means over the four
settings, and paired statistics of SERF (FedResolve) against every other method.
"""
import csv
import json
from collections import defaultdict

import numpy as np

from e2e_common import ROOT
from e2e_stats import mcnemar, paired_bootstrap

ART = ROOT / "artifacts" / "public_paired_v1"
REF = "FedResolve"
BUDGET = 2


def load(dataset):
    rows = []
    with (ART / dataset / "predictions.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def flag(row, name):
    return 1.0 if row[name] else 0.0


def main() -> int:
    table, stats = [], []
    for dataset in ("finrag", "visdom"):
        index = defaultdict(lambda: defaultdict(dict))
        for row in load(dataset):
            if row["budget"] != BUDGET:
                continue
            index[(row["clients"], row["track"], row["method"])][row["scenario"]][row["uid"]] = row
        per_record = {}
        for key, scenarios in sorted(index.items()):
            ans = scenarios.get("unneeded_offline", {})
            una = scenarios.get("qrel_missing", {})
            full = scenarios.get("full", {})
            shared = sorted(set(ans) & set(una))
            if not shared:
                continue
            a_e = np.array([flag(ans[u], "evidence_correct") for u in shared])
            a_a = np.array([flag(ans[u], "answer_correct") for u in shared])
            fa_s = np.array([flag(ans[u], "abstain_system") for u in shared])
            fa_r = np.array([flag(ans[u], "abstain_routing") for u in shared])
            u_s = np.array([flag(una[u], "abstain_system") for u in shared])
            u_r = np.array([flag(una[u], "abstain_routing") for u in shared])
            wrong = np.array(
                [0.0 if una[u]["abstain_system"] or una[u]["answer_correct"] else 1.0 for u in shared]
            )
            in_full = [u for u in shared if u in full]
            table.append(
                {
                    "dataset": dataset,
                    "clients": key[0],
                    "track": key[1],
                    "method": key[2],
                    "budget": BUDGET,
                    "paired_records": len(shared),
                    "answerable_success_evidence": float(a_e.mean()),
                    "answerable_success_answer": float(a_a.mean()),
                    "false_abstention_system": float(fa_s.mean()),
                    "false_abstention_routing": float(fa_r.mean()),
                    "unanswerable_abstain_system": float(u_s.mean()),
                    "unanswerable_abstain_routing": float(u_r.mean()),
                    "wrong_answer_when_unanswerable": float(wrong.mean()),
                    "balanced_missingness_accuracy_evidence": float(0.5 * (a_e.mean() + u_s.mean())),
                    "balanced_missingness_accuracy_answer": float(0.5 * (a_a.mean() + u_s.mean())),
                    "full_participation_success_evidence": float(
                        np.mean([flag(full[u], "evidence_correct") for u in in_full])
                    ) if in_full else float("nan"),
                    "full_participation_false_abstention_system": float(
                        np.mean([flag(full[u], "abstain_system") for u in in_full])
                    ) if in_full else float("nan"),
                }
            )
            per_record[key] = {
                u: (0.5 * (a_e[i] + u_s[i]), 0.5 * (a_a[i] + u_s[i]), fa_s[i], a_e[i], fa_r[i])
                for i, u in enumerate(shared)
            }
        names = (
            "balanced_missingness_accuracy_evidence",
            "balanced_missingness_accuracy_answer",
            "false_abstention_system",
            "answerable_success_evidence",
            "false_abstention_routing",
        )
        for (clients, track, method), records in sorted(per_record.items()):
            if method == REF:
                continue
            ref = per_record.get((clients, track, REF))
            if not ref:
                continue
            uids = sorted(set(ref) & set(records))
            for j, name in enumerate(names):
                treat = [ref[u][j] for u in uids]
                ctrl = [records[u][j] for u in uids]
                entry = {
                    "dataset": dataset, "clients": clients, "track": track,
                    "reference": REF, "comparator": method, "metric": name,
                    **paired_bootstrap(treat, ctrl),
                }
                if name in ("answerable_success_evidence", "false_abstention_system", "false_abstention_routing"):
                    entry["mcnemar"] = mcnemar(treat, ctrl)
                stats.append(entry)

    columns = list(table[0].keys())
    with (ART / "paired_missingness.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(table)
    means = defaultdict(lambda: defaultdict(list))
    for row in table:
        for col in columns[6:]:
            means[(row["dataset"], row["method"])][col].append(row[col])
    mean_rows = [
        {"dataset": d, "method": m, "settings": len(v[columns[6]]),
         **{c: float(np.nanmean(x)) for c, x in v.items()}}
        for (d, m), v in sorted(means.items())
    ]
    with (ART / "paired_missingness_means.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(mean_rows[0].keys()))
        writer.writeheader()
        writer.writerows(mean_rows)
    (ART / "paired_statistics.json").write_text(json.dumps(stats, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(table), "stats": len(stats), "mean_rows": len(mean_rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
