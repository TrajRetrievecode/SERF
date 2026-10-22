#!/usr/bin/env python3
"""Format the paper tables without changing their numbers. Run after generators."""
import csv
import json
import re
import sys
import os
from pathlib import Path
ROOT = Path(os.environ.get("SERF_ROOT", str(Path(__file__).resolve().parents[1])))
T_EN = ROOT / "artifacts/paper/tables"
# Centralized ranks the pages of every online client in one index; its Calls cells print the mean
# number of clients that hold its cited pages (control analysis "reference", collected by
# collect_facts.py as fairness.owner_contacts). Before that analysis has run the cells print a dash.
_FACTS_PATH = ROOT / "artifacts/analysis/collected_facts.json"
_FAIR = json.load(open(_FACTS_PATH, encoding="utf-8")).get("fairness", {}) if _FACTS_PATH.exists() else {}
_OWN, _OWN_SET = _FAIR.get("owner_contacts", {}), _FAIR.get("owner_contacts_by_setting_b2", {})
_SETTINGS = ("c8_seen_query", "c8_unseen_family", "c16_seen_query", "c16_unseen_family")


def centralized_contacts(name, row, label_cols, k):
    """Value for the k-th numeric column of a Centralized row, or a dash when no contacts are collected."""
    if name == "tab_attribution":
        return row[label_cols + k]
    if name == "tab_public_main":
        key = {2: "finrag B2", 6: "visdom B2"}.get(k)
    elif name == "tab_gold_main_budget":
        key = {2: "evaluation_set B2", 6: "evaluation_set B3"}.get(k)
    elif name == "tab_public_calls":
        v = _OWN_SET.get(f"{'finrag' if k < 4 else 'visdom'} {_SETTINGS[k % 4]}")
        return f"{v:.3f}" if v is not None else "--"
    else:
        key = None
    v = _OWN.get(key) if key else None
    return f"{v:.3f}" if v is not None else "--"

V3 = ROOT / "artifacts/serf_v3"
MAX, MIN, CALLS = "max", "min", "calls"
ONE = (r"\methodone{}", r"\Methodone{}")
ORACLE_E = "Oracle evidence"
ORACLE_C = "Oracle client set"
S2G_ROW = r"\stog{}$^{\dagger}$"
CFG = {
    "tab_attribution": dict(drop=[], cols=[MAX, CALLS, MAX, CALLS], tabcolsep="3pt"),
    "tab_composite": dict(drop=[], cols=[MAX] * 4, tabcolsep="3pt"),
    "tab_gold_main_budget": dict(drop=[*ONE, ORACLE_E, "Local"], cols=[MAX, MAX, CALLS, MIN, MAX, MAX, CALLS, MIN, MAX], tabcolsep="2.5pt"),
    "tab_public_main": dict(drop=[*ONE, ORACLE_E, "Local"], cols=[MAX, MAX, CALLS, MAX, MAX, MAX, CALLS, MAX]),
    "tab_ablation": dict(drop=[S2G_ROW], cols=[MAX, MIN, MAX, MAX, MIN, MAX], add_first_version=True, tabcolsep="2pt"),
    "tab_public_calls": dict(drop=[*ONE, ORACLE_E], cols=[CALLS] * 8),
}
DOWN = r" $\downarrow$"

EN_HEADER = [
    (r"Cit\.\\ supp\.", "Citation support"), (r"Claim tok\.", "Claim tokens"), (r"Claim num\.", "Claim numbers"),
    (r"Tok\.\\ F1", "Token F1"), (r"Cit\.\\ F1", "Citation F1"), (r"Ans\.\\ F1", "Answer F1"),
    (r"Ans\.\\ \(86\)", "Answered"), (r"Abst\.\\ \(90\)", "Abstained"), (r"New \(50\)", "New"),
    (r"Ev\.\\ R", "Recall"), (r"Ev\.\\ P", "Precision"), (r"Strict(?! success)", "Strict success"),
    (r"Paired", "Balanced"), (r"Abst\.", "Abstention"), (r"Bal\.", "Balanced"), (r"Succ\.", "Success"),
    (r"& F1", "& Answer F1"), (r"Calls(?! \$)", "Calls" + DOWN), (r"MB(?! \$)", "MB" + DOWN),
    (r"Version", "Method"), (r"\+ test", "with test"),
]

EN_LABEL = [(r"\method{} (ours)", r"\method{}"), (r"\methodone{}", r"\Methodone{}"), (r"\methodtwo{}", r"\Methodtwo{}"),
            (r"Test fires, answerable (\%)", r"False alarms of the test, \%"),
            (r"\quad w/o early stop", r"\quad without early stop"), (r"\quad w/o residual update", r"\quad without residual update"),
            (r"\quad w/o sub-questions", r"\quad without sub-questions"),
            (r"\quad w/o availability test", r"\quad without availability test")]

ONE_PAGE_ROW = r"\quad one page per client"

ONE_PAGE_ROWS = (ONE_PAGE_ROW, r"\quad with one page per client")

AS1 = r" Bold and $^{\dagger}$ as in Table~\ref{tab:gold-main}."

CAP_EN = {
    "tab_attribution": r"Each external router as adapted and with the evidence selection and the early stop of \method{}, and global page ranking as adapted and with the evidence selection, public datasets at budget 2, mean over the four settings." + AS1,
    "tab_composite": r"Share of the composite questions, whose two parts lie on different clients, with complete evidence at budgets 2 and 3, mean over the four settings." + AS1,
    "tab_gold_main_budget": r"Evaluation set with every client online at budgets 2 and 3, and \balmiss{} on the paired availability scenarios at budget 2. Bold marks the best value among the external methods and \method{}, and $^{\dagger}$ marks our adaptation of an external method.",
    "tab_public_main": r"Public datasets at budget 2, mean over the four settings, with \balmiss{} on the paired availability scenarios." + AS1,
    "tab_ablation": r"Component ablation of \method{} on the public datasets at budget 2, mean over the four settings. Bold marks the best value.",
    "tab_public_calls": r"Client calls on the public datasets by setting at budget 2." + AS1,
}
PLAIN = {
    "tab_public_paired": dict(caption=r"Paired differences of \method{} minus the named method on the public datasets at budget 2 with every client online. $^{*}$ marks a 95\% confidence interval that excludes zero."),
    "tab_by_subquestions": dict(caption=r"Evidence recall and client calls of \method{} and of the variant without sub-questions by number of sub-questions at budget 2, mean over the four settings, with the number of pairs of a setting and a query.")
}
def strip_bold(cell):
    return re.sub(r"\\textbf\{([^{}]*)\}", r"\1", cell)


def number(cell):
    c = strip_bold(cell).strip()
    return float(c) if re.fullmatch(r"-?\d+(\.\d+)?", c) else None


def row_class(label):
    if label.startswith("Oracle"):
        return "oracle"
    if label in ("Local", "All", "Centralized") or label.startswith("Centralized"):
        return "reference"
    if "False alarms" in label or "Test fires" in label or label == "Scenarios":
        return "other"
    return "compared"


def first_version_row():
    cells = list(csv.DictReader(open(V3 / "v3_cells.csv", encoding="utf-8")))
    miss = list(csv.DictReader(open(V3 / "v3_missingness.csv", encoding="utf-8")))
    out = []
    for ds in ("finrag", "visdom"):
        r = [c for c in cells if c["dataset"] == ds and c["method"] == "FedResolve" and c["budget"] == "2" and c["scenario"] == "full"]
        m = [x for x in miss if x["dataset"] == ds and x["method"] == "FedResolve" and x["kind"] == "proxy"]
        assert len(r) == 4 and len(m) == 4, (ds, len(r), len(m))
        out += [sum(float(c["evidence_recall"]) for c in r) / 4, sum(float(c["client_calls"]) for c in r) / 4,
                sum(float(x["unanswerable_abstain_routing"]) for x in m) / 4]
    return ONE_PAGE_ROW + " & " + " & ".join(f"{v:.3f}" for v in out) + r" \\"


def split_table(text):
    lines = text.replace("\r\n", "\n").split("\n")
    top = next(i for i, ln in enumerate(lines) if ln.strip() == r"\toprule")
    mid = next(i for i in range(top, len(lines)) if lines[i].strip() == r"\midrule")
    bot = next(i for i, ln in enumerate(lines) if ln.strip() == r"\bottomrule")
    cap = next(i for i, ln in enumerate(lines) if ln.startswith(r"\caption{"))
    assert lines[cap].rstrip().endswith("}"), "caption must be on one line"
    return lines, top, mid, bot, cap


def cells_of(line):
    body = line.rstrip()
    assert body.endswith(r"\\"), line
    return [c.strip() for c in body[:-2].split("&")]


def set_tabcolsep(lines, value):
    return [re.sub(r"\\setlength\{\\tabcolsep\}\{[^}]*\}", lambda m: r"\setlength{\tabcolsep}{" + value + "}", ln) for ln in lines]


def format_table(name, cfg):
    path = T_EN / f"{name}.tex"
    lines, top, mid, bot, cap = split_table(path.read_text(encoding="utf-8"))
    label_cols = cfg.get("label_cols", 1)
    header = []
    for i, ln in enumerate(lines[:mid + 1]):
        if i > top:
            for pat, rep in EN_HEADER:
                ln = re.sub(pat, lambda m, rep=rep: rep, ln)
        header.append(ln)
    if cfg.get("tabcolsep"):
        header = set_tabcolsep(header, cfg["tabcolsep"])
    if cfg.get("font"):
        header = [re.sub(r"^\\centering\\[a-z]+$", lambda m: r"\centering" + cfg["font"], ln) for ln in header]
    tail, after = lines[bot:cap], lines[cap + 1:]
    if cfg.get("wide"):
        header = set_tabcolsep(header, "5pt")
        header = [ln.replace(r"\begin{table}[t]", r"\begin{table*}[t]").replace(r"\centering\footnotesize", r"\centering\small")
                  for ln in header]
        after = [ln.replace(r"\end{table}", r"\end{table*}") for ln in after]
    body = []
    for ln in lines[mid + 1:bot]:
        if ln.strip() == r"\midrule":
            body.append(None)
            continue
        c = [strip_bold(x) for x in cells_of(ln)]
        for a, b in EN_LABEL:
            c = [b if x == a else x for x in c[:label_cols]] + c[label_cols:]
        label = c[label_cols - 1]
        if label in cfg["drop"] or label in ONE_PAGE_ROWS or label == "Scenarios":
            continue
        body.append(c)
    if cfg.get("add_first_version"):
        at = next(i for i, c in enumerate(body) if c and c[0] == r"\quad without availability test")
        body.insert(at + 1, cells_of(first_version_row()))
    if cfg.get("count_row"):
        body = [[x.strip() for x in cfg["count_row"].split("&")], None] + body
    cols = list(cfg["cols"])
    app = cfg.get("append")
    if app:
        # one more column, taken from another formatted table by row label; skipped when the
        # column is already there, so that a second run changes nothing
        if all(len(c) == label_cols + len(cols) for c in body if c is not None):
            s_lines, s_top, s_mid, s_bot, _ = split_table((T_EN / f"{app['table']}.tex").read_text(encoding="utf-8"))
            src = {}
            for ln in s_lines[s_mid + 1:s_bot]:
                if ln.strip() != r"\midrule":
                    c = [strip_bold(x) for x in cells_of(ln)]
                    src[c[0]] = c[app["column"]]
            for c in body:
                if c is not None:
                    c.append(src[c[label_cols - 1]])
            n = label_cols + len(cols) + 1
            grown = []
            for ln in header:
                if ln.startswith(r"\begin{tabular}"):
                    ln = ln.replace("@{}}", "c@{}}")
                elif ln.startswith(r"\cmidrule"):
                    ln = ln + r"\cmidrule(l){" + f"{n}-{n}" + "}"
                elif ln.rstrip().endswith(r"\\") and r"\multicolumn" in ln:
                    ln = ln.rstrip()[:-2].rstrip() + " & " + app["group"] + r" \\"
                elif ln.rstrip().endswith(r"\\"):
                    ln = ln.rstrip()[:-2].rstrip() + " & " + app["header"] + r" \\"
                grown.append(ln)
            header = grown
        cols.append(app["direction"])
    clean = []
    for c in body:
        if c is None and (not clean or clean[-1] is None):
            continue
        clean.append(c)
    while clean and clean[-1] is None:
        clean.pop()
    body = clean
    groups = {}
    for i, c in enumerate(body):
        if c is None:
            continue
        groups.setdefault(c[0] if label_cols == 2 else "", []).append(i)
    for idx in groups.values():
        for k, direction in enumerate(cols):
            col = label_cols + k
            if direction == CALLS:
                for i in idx:
                    if body[i][label_cols - 1] == "Centralized":
                        body[i][col] = centralized_contacts(name, body[i], label_cols, k)
            cand = []
            for i in idx:
                label = body[i][label_cols - 1]
                if row_class(label) != "compared":
                    continue
                if direction == CALLS and r"\coremmrag" in label:
                    continue
                v = number(body[i][col])
                if v is not None:
                    cand.append((v, i))
            if len(cand) < 2:
                continue
            vals = [v for v, _ in cand]
            if max(vals) == min(vals):
                continue
            best = max(vals) if direction == MAX else min(vals)
            for v, i in cand:
                if v == best:
                    body[i][col] = r"\textbf{" + body[i][col] + "}"
    out = header
    for c in body:
        out.append(r"\midrule" if c is None else " & ".join(c) + r" \\")
    out += tail + [r"\caption{" + CAP_EN[name] + "}"] + after
    path.write_text("\n".join(out), encoding="utf-8", newline="\n")
    return out


def stack_datasets(directory, name, font=r"\footnotesize", tabcolsep="4pt"):
    """Write <name>_stacked.tex, the one-column form of a table with two dataset blocks side by side.

    The side-by-side table has a label column and the same k columns for each of two datasets.
    The stacked table has the label column and k columns, with one block of rows per dataset."""
    lines, top, mid, bot, cap = split_table((directory / f"{name}.tex").read_text(encoding="utf-8"))
    lines = [ln for ln in lines if not ln.startswith("% ")]
    lines, top, mid, bot, cap = split_table("\n".join(lines))
    group = cells_of(lines[top + 1])
    titles = [re.fullmatch(r"\\multicolumn\{\d+\}\{c\}\{(.*)\}", g).group(1) for g in group[1:]]
    head = cells_of(lines[mid - 1])
    k = (len(head) - 1) // 2
    assert len(titles) == 2 and len(head) == 1 + 2 * k and head[1:1 + k] == head[1 + k:], (name, head)
    rows = [cells_of(ln) for ln in lines[mid + 1:bot] if ln.strip() != r"\midrule"]
    out = [r"\begin{table}[t]", r"\centering" + font, r"\setlength{\tabcolsep}{" + tabcolsep + "}",
           r"\begin{tabular}{@{}l" + "c" * k + "@{}}", r"\toprule", " & ".join(head[:1 + k]) + r" \\"]
    for b, title in enumerate(titles):
        out += [r"\midrule", r"\multicolumn{" + str(1 + k) + r"}{c}{" + title + r"} \\", r"\midrule"]
        out += [" & ".join([r[0]] + r[1 + b * k:1 + (b + 1) * k]) + r" \\" for r in rows]
    out += [r"\bottomrule", r"\end{tabular}", lines[cap], next(ln for ln in lines if ln.startswith(r"\label{")), r"\end{table}"]
    (directory / f"{name}_stacked.tex").write_text("\n".join(out), encoding="utf-8", newline="\n")


def main():
    for name, cfg in CFG.items():
        format_table(name, cfg)
        print("formatted", name)
    stack_datasets(T_EN, "tab_ablation")
    for name, x in PLAIN.items():
        p = T_EN / f"{name}.tex"
        lines, top, mid, bot, cap = split_table(p.read_text(encoding="utf-8"))
        lines[cap] = r"\caption{" + x["caption"] + "}"
        lines = [re.sub(r"^\\methodone\{\} &", lambda m: r"\Methodone{} &", ln) for ln in lines]
        p.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    bad = 0
    for f in T_EN.glob("tab_*.tex"):
        s = f.read_text(encoding="utf-8")
        if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", s) or s.count(r"\caption{") != 1 or s.count(r"\label{") != 1:
            print("PROBLEM", f)
            bad += 1
    sys.exit(1 if bad else 0)

if __name__ == "__main__":
    main()
