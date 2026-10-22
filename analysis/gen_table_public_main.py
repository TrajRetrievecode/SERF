#!/usr/bin/env python3
"""Table 2: public datasets at budget 2, mean over the four settings. Reads baseline aggregates, paired aggregates, and SERF aggregates under artifacts/."""
import csv
from collections import defaultdict
from pathlib import Path
import os

ROOT = Path(os.environ.get("SERF_ROOT", str(Path(__file__).resolve().parents[1])))
ld = lambda p: list(csv.DictReader(open(p, encoding="utf-8-sig")))
DAG = r"$^{\dagger}$"
ROWS = [("Local", "Local", 0), ("All", "All", 0), ("Centralized", "Centralized", 0),
        ("RAGRoute", r"\ragroute{}" + DAG, 1), ("UniversalRAG", r"\universalrag{}" + DAG, 1),
        ("CoRe-MMRAG", r"\coremmrag{}" + DAG, 1), ("S2G-RAG", r"\stog{}" + DAG, 1),
        ("S2G-RAG+avail", r"\stog{}+avail" + DAG, 1), ("FedResolve-v3", r"\method{} (ours)", 2),
        ("FedResolve", r"\methodone{}", 2),
        ("Oracle-client", "Oracle client set", 3), ("Oracle-evidence", "Oracle evidence", 3)]
ALL_CALLS = {"All", "CoRe-MMRAG"}
BOLD_POOL = {"RAGRoute", "UniversalRAG", "S2G-RAG", "S2G-RAG+avail", "FedResolve-v3", "FedResolve"}

main = defaultdict(list)
for r in ld(ROOT / "artifacts/public_e2e_v1/main_table_slot.csv"):
    main[(r["dataset"], r["method"])].append(r)
proxy = defaultdict(list)
for r in ld(ROOT / "artifacts/public_e2e_v1/missingness_proxy_slot.csv"):
    if r["budget"] == "2":
        proxy[(r["dataset"], r["method"])].append(float(r["balanced_missingness_accuracy_evidence"]))
paired = {(r["dataset"], r["method"]): float(r["balanced_missingness_accuracy_evidence"])
          for r in ld(ROOT / "artifacts/public_paired_v1/paired_missingness_means.csv")}

# SERF (the final version, artifact FedResolve-v3) from the SERF-v3 public run; SERF-v1 (FedResolve) from the reported run
v3cells = ROOT / "artifacts/serf_v3/v3_cells.csv"
v3miss = ROOT / "artifacts/serf_v3/v3_missingness.csv"
for r in ld(v3cells):
    if r["method"] == "FedResolve-v3" and r["budget"] == "2" and r["scenario"] == "full":
        main[(r["dataset"], "FedResolve-v3")].append(r)
acc = defaultdict(list)
for r in ld(v3miss):
    if r["method"] == "FedResolve-v3":
        acc[(r["dataset"], r["kind"])].append(float(r["balanced_evidence"]))
for ds in ("finrag", "visdom"):
    assert len(acc[(ds, "proxy")]) == 4 and len(acc[(ds, "paired")]) == 4
    proxy[(ds, "FedResolve-v3")] = acc[(ds, "proxy")]
    paired[(ds, "FedResolve-v3")] = sum(acc[(ds, "paired")]) / 4
V3NOTE = r" \methodone{}, the one-page variant of \method{}, cites the single best page of each called client."


def mean(ds, m, k):
    rs = main[(ds, m)]
    assert len(rs) == 4, (ds, m, len(rs))
    return sum(float(r[k]) for r in rs) / 4


best = {}
for ds in ("finrag", "visdom"):
    pool = [m for m, _, _ in ROWS if m in BOLD_POOL]
    best[(ds, "er")] = max(round(mean(ds, m, "evidence_recall"), 3) for m in pool)
    best[(ds, "calls")] = min(round(mean(ds, m, "client_calls"), 3) for m in pool)


def cells(m):
    """Evidence recall, answer token F1, calls, balanced missingness accuracy on the paired scenarios."""
    out = []
    for ds in ("finrag", "visdom"):
        out.append(f"{round(mean(ds, m, 'evidence_recall'), 3):.3f}")
        out.append(f"{round(mean(ds, m, 'answer_token_f1'), 3):.3f}")
        out.append("all" if m in ALL_CALLS else f"{round(mean(ds, m, 'client_calls'), 3):.3f}")
        pa = paired.get((ds, m))
        out.append(f"{pa:.3f}" if pa is not None else "--")
    return out


lines = [r"\begin{table*}[t]", r"\centering\small", r"\setlength{\tabcolsep}{5pt}",
         r"\begin{tabular}{@{}lcccccccc@{}}", r"\toprule",
         r" & \multicolumn{4}{c}{\finrag{}} & \multicolumn{4}{c}{\visdom{}} \\",
         r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
         r"Method & Ev.\ R & Ans.\ F1 & Calls & Paired & Ev.\ R & Ans.\ F1 & Calls & Paired \\", r"\midrule"]
prev = None
for m, disp, grp in ROWS:
    if prev is not None and grp != prev:
        lines.append(r"\midrule")
    lines.append(disp + " & " + " & ".join(cells(m)) + r" \\")
    prev = grp
lines += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Public datasets at $\budget=2$, each value the mean over four settings, 8 and 16 clients on the seen and held-out tracks. Ev.\ R is evidence recall and Calls the mean number of \clientcalls{} with every client online, 8/16 marking a method that calls every online client. Proxy and Paired are \balmiss{} under the evidence criterion, with the unanswerable scenario in which every client holding a relevant document is offline; the answerable side of Proxy is the run with every client online, and that of Paired is a scenario with the same number of clients offline, none of which holds a relevant document. \stog{}+avail is \stog{} with the availability test of \method{}. Oracle evidence ignores availability and has no missingness value. Centralized takes no part in the call protocol. $^{\dagger}$ as in Table~\ref{tab:gold-main}. Bold marks the best Ev.\ R and Calls among \method{}, \methodone{}, and the external methods limited to two calls. Values for each setting are in Appendix~\ref{sec:appendix:public}." + V3NOTE + "}",
          r"\label{tab:public-main}", r"\end{table*}"]
(ROOT / "artifacts/paper/tables/tab_public_main.tex").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
print("\n".join(lines[9:-5]))
