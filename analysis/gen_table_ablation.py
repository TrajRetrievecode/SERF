#!/usr/bin/env python3
"""Table 4: component ablation, budget 2, mean over four settings. Reads artifacts/analysis/v3_ablation.json; writes artifacts/paper/tables/tab_ablation.tex."""
import json
from collections import defaultdict
from pathlib import Path
import os

ROOT = Path(os.environ.get("SERF_ROOT", str(Path(__file__).resolve().parents[1])))
PAPER = ROOT / "artifacts/paper/tables"
PAPER.mkdir(parents=True, exist_ok=True)
abl = json.loads((ROOT / "artifacts/analysis/v3_ablation.json").read_text(encoding="utf-8"))

ROWS = [
    ("FedResolve-v3", r"\method{}"),
    ("FedResolve-noEarlyStop", r"\quad without early stop"),
    ("FedResolve-noResidual", r"\quad without residual update"),
    ("FedResolve-noSlots", r"\quad without sub-questions"),
    ("FedResolve-noAvailability", r"\quad without availability test"),
    ("S2G-RAG", r"\stog{}$^{\dagger}$"),
]


def cell_means(ds):
    acc = defaultdict(list)
    for c in abl["cells"]:
        if c["dataset"] == ds:
            acc[(c["method"], c["scenario"])].append(c)
    out = {}
    for method, _ in ROWS:
        full = acc[(method, "full")]
        miss = acc[(method, "qrel_missing")]
        assert len(full) == 4 and len(miss) == 4, (ds, method)
        out[method] = (
            sum(c["evidence_recall"] for c in full) / 4,
            sum(c["calls"] for c in full) / 4,
            sum(c["routing_abstention"] for c in miss) / 4,
        )
    return out


fin, vis = cell_means("finrag"), cell_means("visdom")
lines = [
    r"\begin{table}[t]",
    r"\centering\footnotesize",
    r"\setlength{\tabcolsep}{2.8pt}",
    r"\begin{tabular}{@{}lcccccc@{}}",
    r"\toprule",
    r" & \multicolumn{3}{c}{\finrag{}} & \multicolumn{3}{c}{\visdom{}} \\",
    r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
    r"Variant & Ev.\ R & Calls & Abst. & Ev.\ R & Calls & Abst. \\",
    r"\midrule",
]
for method, label in ROWS:
    f, v = fin[method], vis[method]
    if method == "S2G-RAG":
        lines.append(r"\midrule")
    lines.append(f"{label} & {f[0]:.3f} & {f[1]:.3f} & {f[2]:.3f} & {v[0]:.3f} & {v[1]:.3f} & {v[2]:.3f} \\\\")
lines += [
    r"\bottomrule",
    r"\end{tabular}",
    r"\caption{Component ablation of \method{} on the public datasets at $\budget=2$, mean over the four settings, measured on the selected pages without generation. Ev.\ R is evidence recall and Calls the mean number of \clientcalls{} with every client online; Abst.\ is the rate of abstention by the router when every client holding a relevant document is offline. $^{\dagger}$ as in Table~\ref{tab:gold-main}.}",
    r"\label{tab:ablation}",
    r"\end{table}",
]
(PAPER / "tab_ablation.tex").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
