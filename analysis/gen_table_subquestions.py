#!/usr/bin/env python3
"""tab_by_subquestions.tex: evidence recall and client calls of SERF and of the variant without
sub-questions, by the number of sub-questions of a public test query, at budget 2 with every client
online, mean over the four settings; Pairs counts the setting-query pairs of the four settings
together. One block per dataset, so that the table fits one column.
Source: artifacts/analysis/v3_ablation_by_subquestions.json (read-only
analysis of the recorded ablation run). Run before format_tables.py.
"""
import json
from pathlib import Path
import os

ROOT = Path(os.environ.get("SERF_ROOT", str(Path(__file__).resolve().parents[1])))
d = json.load(open(ROOT / "artifacts/analysis/v3_ablation_by_subquestions.json", encoding="utf-8"))["datasets"]
lines = [r"\begin{table}[t]", r"\centering\footnotesize", r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
         r" & & \multicolumn{2}{c}{Recall} & \multicolumn{2}{c}{Calls $\downarrow$} \\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
         r"Sub-questions & Pairs & \method{} & Without & \method{} & Without \\"]
for ds, name in (("finrag", r"\finrag{}"), ("visdom", r"\visdom{}")):
    lines += [r"\midrule", r"\multicolumn{6}{c}{" + name + r"} \\", r"\midrule"]
    for k, word in (("1", "one"), ("2", "two")):
        p = d[ds]["pooled"][k]
        lines.append(f"{word} & {p['n_total']} & {p['serf_recall']:.3f} & {p['variant_recall']:.3f} & "
                     f"{p['serf_calls']:.3f} & {p['variant_calls']:.3f} \\\\")
lines += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Evidence recall and client calls of \method{} and of the variant without \claimslots{} by number of \claimslots{} at budget 2, mean over the four settings.}",
          r"\label{tab:by-subquestions}", r"\end{table}"]
(ROOT / "artifacts/paper/tables/tab_by_subquestions.tex").write_text("\n".join(lines), encoding="utf-8", newline="\n")
print("\n".join(lines))
