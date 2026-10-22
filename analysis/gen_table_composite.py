#!/usr/bin/env python3
"""Table 4: complete evidence on composite questions at budgets 2 and 3, mean over four settings. Run before format_tables.py."""
import csv
from pathlib import Path

import os
ROOT = Path(os.environ.get("SERF_ROOT", str(Path(__file__).resolve().parents[1])))
rows = list(csv.DictReader(open(ROOT / "artifacts/controls/composite/results/four_setting_means.csv", encoding="utf-8")))


def val(ds, method, variant, budget):
    r = [x for x in rows if x["dataset"] == ds and x["method"] == method and x["variant"] == variant
         and x["policy"] == "native" and str(x["budget"]) == str(budget) and x["scenario"] == "full"]
    assert len(r) == 1, (ds, method, variant, budget, len(r))
    return float(r[0]["complete_evidence"])


# Global page ranking uses evidence selection on the same composites.
D_DIR = ROOT / "artifacts/controls/global_selection/results"
drows = list(csv.DictReader(open(D_DIR / "mean_four_settings.csv", encoding="utf-8")))


def dval(ds, variant, budget):
    r = [x for x in drows if x["panel"] == "composite" and x["dataset"] == ds and x["method"] == "Centralized" and x["variant"] == variant
         and str(x["budget"]) == str(budget) and x["scenario"] == "full"]
    assert len(r) == 1, ("D", ds, variant, budget, len(r))
    return float(r[0]["complete_evidence"])


for ds in ("finrag", "visdom"):
    for b in (2, 3):
        assert abs(dval(ds, "base", b) - val(ds, "Centralized", "native", b)) < 5e-4, ("D base differs from C Centralized", ds, b)

DAG = r"$^{\dagger}$"
ROWS = [("All", "native", "All"), ("Centralized", "native", "Centralized"), ("D", "+sel", "Centralized with selection"), None,
        ("RAGRoute", "pooled_share", r"\ragroute{}" + DAG), ("UniversalRAG", "pooled_share", r"\universalrag{}" + DAG),
        ("S2G-RAG", "pooled_share", r"\stog{}" + DAG), None,
        ("FedResolve-v3", "native", r"\method{}"), None,
        ("Oracle-client", "native", "Oracle client set")]
lines = [r"\begin{table}[t]", r"\centering\small", r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
         r" & \multicolumn{2}{c}{\finrag{}} & \multicolumn{2}{c}{\visdom{}} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
         r"Method & $\budget=2$ & $\budget=3$ & $\budget=2$ & $\budget=3$ \\", r"\midrule"]
for row in ROWS:
    if row is None:
        lines.append(r"\midrule")
        continue
    method, variant, label = row
    if method == "D":
        cells = [f"{dval(ds, variant, b):.3f}" for ds in ("finrag", "visdom") for b in (2, 3)]
    else:
        cells = [f"{val(ds, method, variant, b):.3f}" for ds in ("finrag", "visdom") for b in (2, 3)]
    lines.append(label + " & " + " & ".join(cells) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Share of the composite questions, whose two parts lie on different clients, with complete evidence at budgets 2 and 3, mean over the four settings.}",
          r"\label{tab:composite}", r"\end{table}"]
(ROOT / "artifacts/paper/tables").mkdir(parents=True, exist_ok=True)
(ROOT / "artifacts/paper/tables/tab_composite.tex").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
print("\n".join(lines))
