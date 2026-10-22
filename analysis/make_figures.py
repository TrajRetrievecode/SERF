#!/usr/bin/env python3
"""Figures 2 (fig_test), 3 (fig_margin), and 4 (fig_budget), from artifacts/analysis/collected_facts.json and artifacts/enumeration/margins_summary.json. English PDF outputs under artifacts/paper/figures."""
import json
from pathlib import Path
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib import font_manager

ROOT = Path(os.environ.get("SERF_ROOT", str(Path(__file__).resolve().parents[1])))
F = json.load(open(ROOT / "artifacts/analysis/collected_facts.json", encoding="utf-8"))
OUT = {"en": ROOT / "artifacts" / "paper" / "figures"}

BLUE, GREY, BLACK = "#1f5fa8", "#8a8a8a", "#000000"
STYLE = {  # series -> (colour, marker, linestyle)
    "SERF": (BLUE, "o", "-"),
    "S2G-RAG": (GREY, "s", "-"),
    "RAGRoute": (GREY, "^", "--"),
    "UniversalRAG": (GREY, "D", ":"),
    "Centralized": (BLACK, "x", "-."),
}
TXT = {
    "en": dict(font="Times New Roman", budget="Budget $B$", recall="Evidence recall", calls=r"Client calls $\downarrow$",
               a="(a) FinRAGBench-V", b="(b) VisDoM", c="(c) Client calls", b2="(b) Client calls",
               serf_fin="SERF, FinRAGBench-V", serf_vis="SERF, VisDoM", ext="Budgeted external\nmethods",
               names={"SERF": "SERF", "S2G-RAG": "S2G-RAG", "RAGRoute": "RAGRoute", "UniversalRAG": "UniversalRAG",
                      "Centralized": "Centralized", "CoRe-MMRAG": "CoRe-MMRAG", "FinRAGBench-V": "FinRAGBench-V", "VisDoM": "VisDoM"},
               bal="Balanced missingness accuracy", alone="Alone", test="With the test",
               t_a="(a) Evaluation set", t_b="(b) FinRAGBench-V", t_c="(c) VisDoM",
               lead="Lead of the offline client", margin=r"Margin $\varepsilon$",
               unans="Unanswerable", ans="Answerable", fires="Abstains", answers="Answers",
               m_a="(a) Lead by scenario", m_b="(b) Accuracy by margin", bal2="Balanced accuracy"),

}
MARGINS = json.load(open(ROOT / "artifacts" / "enumeration" / "margins_summary.json", encoding="utf-8"))


def setup(lang):
    fonts = {f.name for f in font_manager.fontManager.ttflist}
    fam = TXT[lang]["font"] if TXT[lang]["font"] in fonts else "DejaVu Serif"
    plt.rcParams.update({
        "font.family": ["Times New Roman", fam, "DejaVu Serif"], "font.size": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.linewidth": 0.6,
        "lines.linewidth": 1.1, "lines.markersize": 3.6, "pdf.fonttype": 42, "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
    })


def fig_budget(lang):
    # one column, two panels: evidence recall on FinRAGBench-V, and client calls on both public datasets
    t = TXT[lang]
    fig, axes = plt.subplots(1, 2, figsize=(3.12, 1.9))
    budgets = [1, 2, 3]
    handles = {}
    ax = axes[0]
    sweep = F["finrag"]["budget_sweep"]
    for m in ("SERF", "S2G-RAG", "RAGRoute", "UniversalRAG", "Centralized"):
        col, mk, ls = STYLE[m]
        (h,) = ax.plot(budgets, [sweep[m][str(b)]["evidence_recall"] for b in budgets], color=col, marker=mk, linestyle=ls,
                       zorder=3 if m == "SERF" else 2)
        handles[t["names"][m]] = h
    ax.set_xticks(budgets)
    ax.set_xlabel(t["budget"])
    ax.set_ylabel(t["recall"])
    ax = axes[1]
    (h1,) = ax.plot(budgets, [F["finrag"]["budget_sweep"]["SERF"][str(b)]["client_calls"] for b in budgets], color=BLUE, marker="o", linestyle="-")
    (h2,) = ax.plot(budgets, [F["visdom"]["budget_sweep"]["SERF"][str(b)]["client_calls"] for b in budgets], color=BLUE, marker="o", linestyle="--",
                    markerfacecolor="white")
    (h3,) = ax.plot(budgets, budgets, color=GREY, marker="s", linestyle="-")
    ax.set_xticks(budgets)
    ax.set_yticks([1, 2, 3])
    ax.set_xlabel(t["budget"])
    ax.set_ylabel(t["calls"])
    # the two blue lines are SERF (blue = SERF in the figure legend), named by dataset
    ax.legend([h1, h2, h3], [t["names"]["FinRAGBench-V"], t["names"]["VisDoM"], t["ext"]], frameon=False, loc="upper left",
              handlelength=1.5, borderaxespad=0.1, handletextpad=0.4, labelspacing=0.25, fontsize=7)
    ax.set_ylim(0.85, 4.3)
    for a in axes:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        a.tick_params(width=0.6, length=2.5)
    fig.legend(list(handles.values()), list(handles.keys()), loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.55, 1.04),
               handlelength=2.0, columnspacing=1.0, handletextpad=0.4, labelspacing=0.2)
    fig.subplots_adjust(left=0.15, right=0.985, top=0.76, bottom=0.31, wspace=0.5)
    for a, key in zip(axes, ("a", "b2")):
        box = a.get_position()
        fig.text((box.x0 + box.x1) / 2, 0.03, t[key], ha="center", va="bottom", fontsize=8)
    fig.savefig(OUT[lang] / "fig_budget.pdf")
    plt.close(fig)


def fig_test(lang):
    t = TXT[lang]
    methods = ["RAGRoute", "UniversalRAG", "CoRe-MMRAG", "S2G-RAG"]
    en = F["enumeration"]["balanced"]
    panels = [
        ({m: en[m] for m in methods}, {m: en[m + "+avail"] for m in methods}, en["FedResolve"]),
        *[({m: F[ds]["proxy_balanced_by_method"][m] for m in methods},
           {m: F[ds]["proxy_balanced_by_method"][m + "+test"] for m in methods},
           F[ds]["serf_balanced"]["proxy"]) for ds in ("finrag", "visdom")],
    ]
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 1.5), sharey=True)
    for ax, (alone, with_test, serf) in zip(axes, panels):
        for i, m in enumerate(methods):
            a, b = alone[m], with_test[m]
            ax.plot([a, b], [i, i], color=GREY, linewidth=0.9, zorder=1)
            h_a = ax.scatter([a], [i], s=14, color=GREY, zorder=2)
            h_b = ax.scatter([b], [i], s=14, color=BLUE, zorder=3)
        h_s = ax.axvline(serf, color=BLUE, linestyle="--", linewidth=0.9, zorder=0)
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels([t["names"][m] for m in methods])
        ax.set_ylim(-0.6, len(methods) - 0.4)
        ax.invert_yaxis()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(width=0.6, length=2.5)
        lo = min(min(alone.values()), serf)
        hi = max(max(with_test.values()), serf)
        pad = 0.12 * (hi - lo)
        ax.set_xlim(lo - pad, hi + pad)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
        ax.set_xlabel(t["bal"])
    fig.legend([h_a, h_b, h_s], [t["alone"], t["test"], "SERF"], loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.56, 1.03),
               handlelength=1.6, columnspacing=1.2, handletextpad=0.4)
    fig.subplots_adjust(left=0.135, right=0.99, top=0.84, bottom=0.45, wspace=0.14)
    for a, key in zip(axes, ("t_a", "t_b", "t_c")):
        box = a.get_position()
        fig.text((box.x0 + box.x1) / 2, 0.03, t[key], ha="center", va="bottom", fontsize=8)
    fig.savefig(OUT[lang] / "fig_test.pdf")
    plt.close(fig)


def fig_margin(lang):
    t = TXT[lang]
    am = F["availability_margin"]
    eps = am["margin"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(3.12, 1.62), gridspec_kw={"width_ratios": [1.25, 1.0]})
    rng = __import__("numpy").random.default_rng(20260805)
    handles = {}
    for y, answerable in ((0, False), (1, True)):
        pts = [s for s in MARGINS["scenarios_list"] if s["answerable"] == answerable]
        xs = [s["max_signed_lead"] for s in pts]
        jit = rng.uniform(-0.2, 0.2, len(xs))
        for x, j, s in zip(xs, jit, pts):
            if s["test_fired"]:
                handles["fires"] = ax.scatter([x], [y + j], s=6, color=BLUE, linewidths=0, zorder=3)
            else:
                handles["answers"] = ax.scatter([x], [y + j], s=7, facecolors="white", edgecolors=GREY, linewidths=0.6, zorder=3)
    ax.axvline(eps, color=BLACK, linestyle="--", linewidth=0.7, zorder=2)
    ax.set_yticks([0, 1])
    ax.set_yticklabels([t["unans"], t["ans"]])
    ax.set_ylim(-0.55, 1.55)
    ax.set_xlim(-0.45, 0.32)
    ax.set_xticks([-0.4, -0.2, 0.0, 0.2])
    ax.set_xlabel(t["lead"])
    curve = am["curve"]
    bx.plot([c["margin"] for c in curve], [c["balanced"] for c in curve], color=BLUE, linewidth=1.1, drawstyle="steps-post", zorder=3)
    bx.axvline(eps, color=BLACK, linestyle="--", linewidth=0.7, zorder=2)
    bx.set_xlim(0, 0.3)
    bx.set_ylim(0.48, 1.02)
    bx.set_xticks([0, 0.1, 0.2, 0.3])
    bx.set_yticks([0.5, 0.75, 1.0])
    bx.set_xlabel(t["margin"])
    bx.set_ylabel(t["bal2"], labelpad=2)
    for a in (ax, bx):
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        a.tick_params(width=0.6, length=2.5)
    fig.legend([handles["fires"], handles["answers"]], [t["fires"], t["answers"]], loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.36, 1.04), handlelength=1.2, columnspacing=1.0, handletextpad=0.3)
    fig.subplots_adjust(left=0.235, right=0.985, top=0.85, bottom=0.42, wspace=0.55)
    for a, key in zip((ax, bx), ("m_a", "m_b")):
        box = a.get_position()
        # panel labels are centred below their panel and kept inside the canvas
        x = min(max((box.x0 + box.x1) / 2, 0.20), 0.80)
        fig.text(x, 0.03, t[key], ha="center", va="bottom", fontsize=8)
    fig.savefig(OUT[lang] / "fig_margin.pdf")
    plt.close(fig)


for lang in ("en",):
    OUT[lang].mkdir(parents=True, exist_ok=True)
    setup(lang)
    fig_budget(lang)
    fig_test(lang)
    fig_margin(lang)
    print("figures written for", lang)
