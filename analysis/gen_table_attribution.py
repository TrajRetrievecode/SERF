#!/usr/bin/env python3
"""Table 3: routers and global page ranking with evidence selection, budget 2, mean over four settings. Run before format_tables.py."""
import csv
from pathlib import Path

import os
ROOT = Path(os.environ.get("SERF_ROOT", str(Path(__file__).resolve().parents[1])))
rows = list(csv.DictReader(open(ROOT / "artifacts/controls/attribution/results/mean_four_settings.csv", encoding="utf-8")))


def val(ds, method, variant, key):
    r = [x for x in rows if x["dataset"] == ds and x["method"] == method and x["variant"] == variant
         and str(x["budget"]) == "2" and x["scenario"] == "full"]
    assert len(r) == 1, (ds, method, variant, len(r))
    return float(r[0][key])


DAG = r"$^{\dagger}$"
ROUTERS = [("RAGRoute", r"\ragroute{}" + DAG), ("UniversalRAG", r"\universalrag{}" + DAG), ("S2G-RAG", r"\stog{}" + DAG)]
lines = [r"\begin{table}[t]", r"\centering\small", r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
         r" & \multicolumn{2}{c}{\finrag{}} & \multicolumn{2}{c}{\visdom{}} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
         r"Method & Recall & Calls $\downarrow$ & Recall & Calls $\downarrow$ \\", r"\midrule"]
for method, label in ROUTERS:
    for variant, name in (("base", label), ("+both", r"\quad with selection and stop")):
        cells = []
        for ds in ("finrag", "visdom"):
            cells += [f"{val(ds, method, variant, 'evidence_recall'):.3f}", f"{val(ds, method, variant, 'client_calls'):.3f}"]
        lines.append(name + " & " + " & ".join(cells) + r" \\")
lines.append(r"\midrule")
# Calls count the distinct owners of the cited pages.
D_DIR = ROOT / "artifacts/controls/global_selection/results"
drows = list(csv.DictReader(open(D_DIR / "mean_four_settings.csv", encoding="utf-8")))


def dval(ds, variant, key):
    r = [x for x in drows if x["panel"] == "public" and x["dataset"] == ds and x["method"] == "Centralized" and x["variant"] == variant
         and str(x["budget"]) == "2" and x["scenario"] == "full"]
    assert len(r) == 1, ("D", ds, variant, len(r))
    return float(r[0][key])


for variant, name in (("base", "Centralized"), ("+sel", "Centralized with selection")):
    cells = []
    for ds in ("finrag", "visdom"):
        cells += [f"{dval(ds, variant, 'evidence_recall'):.3f}", f"{dval(ds, variant, 'contacts'):.3f}"]
    lines.append(name + " & " + " & ".join(cells) + r" \\")
lines.append(r"\midrule")
cells = []
for ds in ("finrag", "visdom"):
    cells += [f"{val(ds, 'FedResolve', '+sel', 'evidence_recall'):.3f}", f"{val(ds, 'FedResolve', '+sel', 'client_calls'):.3f}"]
lines.append(r"\method{} & " + " & ".join(cells) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Each external router as adapted and with the evidence selection and the early stop of \method{}, and global page ranking as adapted and with the evidence selection, public datasets at budget 2, mean over the four settings.}",
          r"\label{tab:attribution}", r"\end{table}"]
(ROOT / "artifacts/paper/tables").mkdir(parents=True, exist_ok=True)
(ROOT / "artifacts/paper/tables/tab_attribution.tex").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
print("\n".join(lines))
