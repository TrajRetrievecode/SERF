#!/usr/bin/env python3
"""Table 1: evaluation-set summaries at budgets 2 and 3, paired accuracy at budget 2."""
import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("SERF_ROOT", str(Path(__file__).resolve().parents[1])))
T = ROOT / "artifacts/paper/tables"
T.mkdir(parents=True, exist_ok=True)
formal = json.loads((ROOT / "artifacts/formal_gold_v1/formal_results.json").read_text(encoding="utf-8"))
serf = json.loads((ROOT / "artifacts/analysis/gold_v3_results.json").read_text(encoding="utf-8"))
DAG = r"$^{\dagger}$"
ROWS = [("Local", "Local", 0), ("All", "All", 0), ("Centralized", "Centralized", 0),
        ("RAGRoute", r"\ragroute{}" + DAG, 1), ("UniversalRAG", r"\universalrag{}" + DAG, 1),
        ("CoRe-MMRAG", r"\coremmrag{}" + DAG, 1), ("S2G-RAG", r"\stog{}" + DAG, 1),
        ("SERF-v3", r"\method{}", 2), ("FedResolve", r"\methodone{}", 2),
        ("Oracle client set", "Oracle client set", 3), ("Oracle claim evidence", "Oracle evidence", 3)]
lines = [r"\begin{table*}[t]", r"\centering\small", r"\setlength{\tabcolsep}{2.5pt}",
         r"\begin{tabular}{@{}lccccccccc@{}}", r"\toprule",
         r" & \multicolumn{4}{c}{$\budget=2$} & \multicolumn{4}{c}{$\budget=3$} & Scenarios \\",
         r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(l){10-10}",
         r"Method & Strict & Cit.\ F1 & Calls & MB & Strict & Cit.\ F1 & Calls & MB & Balanced \\", r"\midrule"]
previous = None
for method, label, group in ROWS:
    if previous is not None and group != previous:
        lines.append(r"\midrule")
    values = []
    for budget in ("2", "3"):
        result = serf["online"][budget][method] if method == "SERF-v3" else formal["summary_by_budget"][budget][method]
        values += [f"{result['strict_success']:.3f}", f"{result['citation_f1']:.3f}",
                   "--" if method == "Centralized" else f"{result['client_calls']:.3f}", f"{result['bytes'] / 1e6:.2f}"]
    balanced = serf["missingness_paired_80"][method]["balanced"] if method == "SERF-v3" else formal["participation_main_budget"][method]["balanced_missingness_accuracy"]
    values.append(f"{balanced:.3f}")
    lines.append(label + " & " + " & ".join(values) + r" \\")
    previous = group
lines += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Evaluation set with every client online at budgets 2 and 3, and balanced missingness accuracy on the paired availability scenarios at budget 2.}",
          r"\label{tab:gold-main}", r"\end{table*}"]
(T / "tab_gold_main_budget.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
