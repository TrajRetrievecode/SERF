#!/usr/bin/env python3
"""Table 3: budget-2 paired differences of SERF minus each method, with the supplied bootstrap significance marks. Reads artifacts/serf_v3/v3_paired_statistics.json."""
import json
from pathlib import Path
import os

ROOT = Path(os.environ.get("SERF_ROOT", str(Path(__file__).resolve().parents[1])))
T = ROOT / "artifacts" / "paper" / "tables"
T.mkdir(parents=True, exist_ok=True)
stats = json.load(open(ROOT / "artifacts/serf_v3/v3_paired_statistics.json", encoding="utf-8"))
S = {(r["dataset"], r["config"], r["budget"], r["control"], r["metric"]): r for r in stats if r["treatment"] == "FedResolve-v3"}
CFG = ("c8_seen_query", "c8_unseen_family", "c16_seen_query", "c16_unseen_family")
NAME = {"S2G-RAG": r"\stog{}$^{\dagger}$", "RAGRoute": r"\ragroute{}$^{\dagger}$", "Centralized": "Centralized",
        "FedResolve": r"\Methodone{}"}
BLOCKS = {
    2: [("Evidence recall", "evidence_recall", 3, ("S2G-RAG", "RAGRoute", "Centralized", "FedResolve")),
        ("Evidence precision", "evidence_precision", 3, ("FedResolve",)),
        ("Client calls", "client_calls", 2, ("S2G-RAG",)),
        ("Answer token F1", "answer_token_f1", 3, ("S2G-RAG", "FedResolve"))],

}


def cell(r, nd):
    d = r["difference"]
    if abs(d) < 0.5 * 10 ** -nd:
        v = f"{0:.{nd}f}"
    else:
        v = ("+" if d > 0 else "$-$") + f"{abs(d):.{nd}f}"
    return v + (r"$^{*}$" if r["significant_at_95"] else "")


def table(budget, name, label):
    lines = [r"\begin{table*}[t]", r"\centering\small", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{@{}lcccccccc@{}}", r"\toprule",
             r" & \multicolumn{4}{c}{\finrag{}} & \multicolumn{4}{c}{\visdom{}} \\",
             r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
             r"\method{} minus & 8 seen & 8 held-out & 16 seen & 16 held-out & 8 seen & 8 held-out & 16 seen & 16 held-out \\"]
    counts = {}
    for title, metric, nd, controls in BLOCKS[budget]:
        lines += [r"\midrule", r"\multicolumn{9}{@{}l}{" + title + r"} \\"]
        for c in controls:
            rows = [S[(ds, cfg, budget, c, metric)] for ds in ("finrag", "visdom") for cfg in CFG]
            lines.append(r"\quad " + NAME[c] + " & " + " & ".join(cell(r, nd) for r in rows) + r" \\")
            for ds, part in (("finrag", rows[:4]), ("visdom", rows[4:])):
                counts[(metric, c, ds)] = (sum(r["significant_at_95"] and r["difference"] > 0 for r in part),
                                           sum(r["significant_at_95"] and r["difference"] < 0 for r in part))
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Paired differences of \method{} minus the named method on the public datasets at budget "
              + str(budget) + r" with every client online. $^{*}$ marks a 95\% confidence interval that excludes zero.}",
              r"\label{" + label + "}", r"\end{table*}"]
    (T / name).write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return counts


c2 = table(2, "tab_public_paired.tex", "tab:public-paired")
for k, v in c2.items():
    print(2, k, "significantly higher in", v[0], "settings, lower in", v[1])
