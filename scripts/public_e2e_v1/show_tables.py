#!/usr/bin/env python3
"""Print the main and missingness tables in a compact reviewable form."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

ORDER = [
    "Local",
    "All",
    "Centralized",
    "RAGRoute",
    "UniversalRAG",
    "CoRe-MMRAG",
    "S2G-RAG",
    "S2G-RAG+avail",
    "FedResolve",
    "Oracle-client",
    "Oracle-evidence",
]


def num(value, digits=3):
    try:
        return "{:.{d}f}".format(float(value), d=digits)
    except (TypeError, ValueError):
        return "-"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--which", default="main", choices=["main", "missing"])
    parser.add_argument("--datasets", nargs="+", default=["finrag", "visdom"])
    args = parser.parse_args()
    root = Path(args.dir)

    if args.which == "main":
        rows = list(csv.DictReader((root / "main_table.csv").open(encoding="utf-8")))
        header = "{:16s} {:>6s} {:>6s} {:>7s} {:>6s} {:>6s} {:>6s} {:>6s} {:>8s} {:>6s}".format(
            "method", "EM", "F1", "numAcc", "R@5", "evR", "citF1", "calls", "bytes", "abst"
        )
        for dataset in args.datasets:
            for clients in ("8", "16"):
                for track in ("seen_query", "unseen_family"):
                    sub = [
                        r
                        for r in rows
                        if r["dataset"] == dataset
                        and r["clients"] == clients
                        and r["track"] == track
                    ]
                    if not sub:
                        continue
                    n = sub[0]["records"]
                    print("\n== {} {}c {} (n={}) ==".format(dataset, clients, track, n))
                    print(header)
                    for method in ORDER:
                        r = next((x for x in sub if x["method"] == method), None)
                        if r is None:
                            continue
                        print(
                            "{:16s} {:>6s} {:>6s} {:>7s} {:>6s} {:>6s} {:>6s} {:>6s} {:>8.0f} {:>6s}".format(
                                method,
                                num(r["answer_exact_match"]),
                                num(r["answer_token_f1"]),
                                num(r["numeric_accuracy"]),
                                num(r["retrieval_recall@5"]),
                                num(r["evidence_recall"]),
                                num(r["citation_f1"]),
                                num(r["client_calls"], 2),
                                float(r["protocol_bytes"]),
                                num(r["abstain_system_rate"]),
                            )
                        )
    else:
        rows = list(
            csv.DictReader((root / "missingness_proxy.csv").open(encoding="utf-8"))
        )
        header = "{:16s} {:>5s} {:>7s} {:>7s} {:>7s} {:>7s} {:>7s} {:>7s}".format(
            "method", "n", "ansEv", "ansAns", "abstSys", "abstRt", "falseAb", "balEv"
        )
        for dataset in args.datasets:
            for clients in ("8", "16"):
                for track in ("seen_query", "unseen_family"):
                    sub = [
                        r
                        for r in rows
                        if r["dataset"] == dataset
                        and r["clients"] == clients
                        and r["track"] == track
                    ]
                    if not sub:
                        continue
                    print(
                        "\n== {} {}c {} (paired n={}) ==".format(
                            dataset, clients, track, sub[0]["paired_records"]
                        )
                    )
                    print(header)
                    for method in ORDER:
                        r = next((x for x in sub if x["method"] == method), None)
                        if r is None:
                            continue
                        print(
                            "{:16s} {:>5s} {:>7s} {:>7s} {:>7s} {:>7s} {:>7s} {:>7s}".format(
                                method,
                                r["paired_records"],
                                num(r["answerable_success_evidence"]),
                                num(r["answerable_success_answer"]),
                                num(r["unanswerable_abstain_accuracy_system"]),
                                num(r["unanswerable_abstain_accuracy_routing"]),
                                num(r["false_abstention_rate_when_answerable"]),
                                num(r["balanced_missingness_accuracy_evidence"]),
                            )
                        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
