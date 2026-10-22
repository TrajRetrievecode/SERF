#!/usr/bin/env python3
"""Table 6: budget-2 full-participation client calls, one column per setting."""
import csv
import os
from pathlib import Path

ROOT = Path(os.environ.get("SERF_ROOT", str(Path(__file__).resolve().parents[1])))
T = ROOT / "artifacts/paper/tables"
T.mkdir(parents=True, exist_ok=True)
def load(path):
    with path.open(encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))

baseline = load(ROOT / "artifacts/public_e2e_v1/main_table_slot.csv")
cells = load(ROOT / "artifacts/serf_v3/v3_cells.csv")
values = {(r["dataset"], f"c{r['clients']}_{r['track']}", r["method"]): float(r["client_calls"]) for r in baseline}
for row in cells:
    if row["method"] == "FedResolve-v3" and row["budget"] == "2" and row["scenario"] == "full":
        values[(row["dataset"], row["config"], row["method"])] = float(row["client_calls"])
CFG = ("c8_seen_query", "c8_unseen_family", "c16_seen_query", "c16_unseen_family")
DAG = r"$^{\dagger}$"
ROWS = [("Local", "Local", 0), ("All", "All", 0), ("Centralized", "Centralized", 0),
        ("RAGRoute", r"\ragroute{}" + DAG, 1), ("UniversalRAG", r"\universalrag{}" + DAG, 1),
        ("CoRe-MMRAG", r"\coremmrag{}" + DAG, 1), ("S2G-RAG", r"\stog{}" + DAG, 1),
        ("S2G-RAG+avail", r"\stog{}+avail" + DAG, 1), ("FedResolve-v3", r"\method{}", 2),
        ("FedResolve", r"\methodone{}", 2), ("Oracle-client", "Oracle client set", 3), ("Oracle-evidence", "Oracle evidence", 3)]
lines = [r"\begin{table*}[t]", r"\centering\small", r"\setlength{\tabcolsep}{5pt}",
         r"\begin{tabular}{@{}lcccccccc@{}}", r"\toprule",
         r" & \multicolumn{4}{c}{\finrag{}} & \multicolumn{4}{c}{\visdom{}} \\",
         r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
         r"Method & 8 seen & 8 held-out & 16 seen & 16 held-out & 8 seen & 8 held-out & 16 seen & 16 held-out \\", r"\midrule"]
previous = None
for method, label, group in ROWS:
    if previous is not None and group != previous:
        lines.append(r"\midrule")
    row = ["--" if method == "Centralized" else f"{values[(dataset, config, method)]:.3f}"
           for dataset in ("finrag", "visdom") for config in CFG]
    lines.append(label + " & " + " & ".join(row) + r" \\")
    previous = group
lines += [r"\bottomrule", r"\end{tabular}", r"\caption{Client calls on the public datasets by setting at budget 2.}",
          r"\label{tab:public-calls}", r"\end{table*}"]
(T / "tab_public_calls.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
