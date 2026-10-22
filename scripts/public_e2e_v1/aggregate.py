#!/usr/bin/env python3
"""Aggregate per-record predictions into the reporting tables.

Outputs
  combined_results.csv    every (dataset, clients, track, method, budget,
                          scenario) cell with quality and cost side by side
  main_table.csv          the main protocol slice: main budget, full
                          participation
  missingness_proxy.csv   the paired qrel-induced answerable / unanswerable
                          track with balanced accuracy
  <dataset>/metrics.json  the same numbers in nested form
  <dataset>/paired_statistics.json  paired bootstrap and McNemar against
                          every other method
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from e2e_common import ROOT, TRACK, read_json, record_pid, write_json
from e2e_stats import align, mcnemar, paired_bootstrap

ARTIFACTS = ROOT / "artifacts" / TRACK

MEAN_METRICS = [
    "retrieval_recall@1",
    "retrieval_recall@3",
    "retrieval_recall@5",
    "evidence_recall",
    "evidence_precision",
    "evidence_f1",
    "citation_recall",
    "citation_precision",
    "citation_f1",
    "answer_exact_match",
    "answer_token_f1",
    "numeric_recall",
    "numeric_accuracy",
    "client_calls",
    "selected_pages",
    "candidate_pages",
    "serialized_request_bytes",
    "serialized_response_bytes",
    "protocol_bytes",
    "retrieval_ms",
]

RATE_METRICS = [
    "abstain_routing",
    "abstain_generator",
    "abstain_system",
    "evidence_correct",
    "answer_correct",
]

PAIRED_METRICS = [
    "answer_token_f1",
    "answer_exact_match",
    "evidence_recall",
    "retrieval_recall@5",
    "client_calls",
    "protocol_bytes",
]


def nan_mean(values: list) -> float:
    array = np.array(
        [float("nan") if v is None else float(v) for v in values], dtype=np.float64
    )
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else float("nan")


def version_suffix(prompt_version: str) -> str:
    return "" if prompt_version == "v1" else f"_{prompt_version}"


def load_predictions(
    dataset: str, prompt_version: str = "v1", tag: str = ""
) -> list[dict]:
    path = (
        ARTIFACTS
        / (dataset + tag)
        / f"predictions{version_suffix(prompt_version)}.jsonl"
    )
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def cell_key(row: dict) -> tuple:
    return (
        row["dataset"],
        row["clients"],
        row["track"],
        row["method"],
        row["budget"],
        row["scenario"],
    )


def summarise(rows: list[dict]) -> dict:
    output = {"records": len(rows)}
    for metric in MEAN_METRICS:
        output[metric] = nan_mean([r.get(metric) for r in rows])
    for metric in RATE_METRICS:
        values = [1.0 if r.get(metric) else 0.0 for r in rows]
        output[metric + "_rate"] = float(np.mean(values)) if values else float("nan")
    output["implementation_level"] = rows[0]["implementation_level"] if rows else ""
    output["parse_failure_rate"] = (
        float(
            np.mean(
                [
                    1.0
                    if r.get("parse_status")
                    not in ("json", "abstained_before_generation")
                    else 0.0
                    for r in rows
                ]
            )
        )
        if rows
        else float("nan")
    )
    return output


def build_cells(rows: list[dict]) -> dict[tuple, dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[cell_key(row)].append(row)
    return {key: summarise(value) for key, value in grouped.items()}


def missingness_table(rows: list[dict], main_budget: int) -> list[dict]:
    """Pair every record's full-participation and qrel-missing outcome."""
    index: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        if row["budget"] != main_budget:
            continue
        if row["scenario"] not in ("full", "qrel_missing"):
            continue
        key = (row["dataset"], row["clients"], row["track"], row["method"])
        index[key].setdefault(row["scenario"], {})[row["uid"]] = row

    output = []
    for key, scenarios in sorted(index.items()):
        full = scenarios.get("full", {})
        missing = scenarios.get("qrel_missing", {})
        shared = sorted(set(full) & set(missing))
        if not shared:
            continue
        answerable_evidence = float(
            np.mean([1.0 if full[u]["evidence_correct"] else 0.0 for u in shared])
        )
        answerable_answer = float(
            np.mean([1.0 if full[u]["answer_correct"] else 0.0 for u in shared])
        )
        abstain_system = float(
            np.mean([1.0 if missing[u]["abstain_system"] else 0.0 for u in shared])
        )
        abstain_routing = float(
            np.mean([1.0 if missing[u]["abstain_routing"] else 0.0 for u in shared])
        )
        false_abstain = float(
            np.mean([1.0 if full[u]["abstain_system"] else 0.0 for u in shared])
        )
        wrong_answer_when_unanswerable = float(
            np.mean(
                [
                    1.0
                    if (not missing[u]["abstain_system"]) and not missing[u]["answer_correct"]
                    else 0.0
                    for u in shared
                ]
            )
        )
        output.append(
            {
                "dataset": key[0],
                "clients": key[1],
                "track": key[2],
                "method": key[3],
                "budget": main_budget,
                "paired_records": len(shared),
                "answerable_success_evidence": answerable_evidence,
                "answerable_success_answer": answerable_answer,
                "unanswerable_abstain_accuracy_system": abstain_system,
                "unanswerable_abstain_accuracy_routing": abstain_routing,
                "false_abstention_rate_when_answerable": false_abstain,
                "wrong_answer_rate_when_unanswerable": wrong_answer_when_unanswerable,
                "balanced_missingness_accuracy_evidence": 0.5
                * (answerable_evidence + abstain_system),
                "balanced_missingness_accuracy_answer": 0.5
                * (answerable_answer + abstain_system),
            }
        )
    return output


def paired_statistics(rows: list[dict], main_budget: int, reference: str) -> list[dict]:
    by_cell: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        if row["budget"] != main_budget or row["scenario"] != "full":
            continue
        key = (row["dataset"], row["clients"], row["track"])
        by_cell[key].setdefault(row["method"], {})[row["uid"]] = row

    output = []
    for key, methods in sorted(by_cell.items()):
        if reference not in methods:
            continue
        for method, rows_b in sorted(methods.items()):
            if method == reference:
                continue
            entry = {
                "dataset": key[0],
                "clients": key[1],
                "track": key[2],
                "budget": main_budget,
                "treatment": reference,
                "control": method,
                "metrics": {},
            }
            for metric in PAIRED_METRICS:
                left, right = align(methods[reference], rows_b, metric)
                entry["metrics"][metric] = paired_bootstrap(left, right)
            left, right = align(methods[reference], rows_b, "answer_token_f1")
            entry["mcnemar_answer_correct"] = mcnemar(left, right, correct_at=0.5)
            left, right = align(methods[reference], rows_b, "evidence_recall")
            entry["mcnemar_evidence_hit"] = mcnemar(left, right, correct_at=1.0)
            output.append(entry)
    return output


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: (
                        ""
                        if isinstance(row.get(k), float) and math.isnan(row.get(k))
                        else row.get(k)
                    )
                    for k in columns
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["finrag", "visdom"])
    parser.add_argument("--reference-method", default="FedResolve")
    parser.add_argument("--prompt-version", default="v1")
    parser.add_argument("--cross-client-only", action="store_true",
        help="keep only records whose gold evidence spans more than one client")
    parser.add_argument("--tag", default="",
        help="suffix on the dataset directory and on every output table")
    args = parser.parse_args()

    record_pid("aggregate")
    protocol = read_json(ROOT / "configs" / TRACK / "protocol_v3.json")
    main_budget = int(protocol["main_budgets"][0])
    suffix = args.tag + version_suffix(args.prompt_version)

    combined_rows: list[dict] = []
    missing_rows: list[dict] = []
    started = time.time()

    for dataset in args.datasets:
        rows = load_predictions(dataset, args.prompt_version, args.tag)
        if args.cross_client_only:
            rows = [r for r in rows if r.get("cross_client_gold")]
        cells = build_cells(rows)
        for key, value in sorted(cells.items()):
            combined_rows.append(
                {
                    "dataset": key[0],
                    "clients": key[1],
                    "track": key[2],
                    "method": key[3],
                    "budget": key[4],
                    "scenario": key[5],
                    **value,
                }
            )
        dataset_missing = missingness_table(rows, main_budget)
        missing_rows.extend(dataset_missing)
        statistics = paired_statistics(rows, main_budget, args.reference_method)
        write_json(
            ARTIFACTS / (dataset + args.tag) / f"metrics{version_suffix(args.prompt_version)}.json",
            {
                "schema_version": "public-e2e-metrics-v1",
                "dataset": dataset,
                "prompt_version": args.prompt_version,
                "main_budget": main_budget,
                "cells": [
                    {
                        "dataset": key[0],
                        "clients": key[1],
                        "track": key[2],
                        "method": key[3],
                        "budget": key[4],
                        "scenario": key[5],
                        **value,
                    }
                    for key, value in sorted(cells.items())
                ],
                "missingness_proxy": dataset_missing,
            },
        )
        write_json(
            ARTIFACTS / (dataset + args.tag) / f"paired_statistics{version_suffix(args.prompt_version)}.json",
            {
                "schema_version": "public-e2e-paired-statistics-v1",
                "dataset": dataset,
                "reference_method": args.reference_method,
                "scope": (
                    "Paired bootstrap over shared record ids at the main budget "
                    "and full participation, plus exact-binomial McNemar. A "
                    "difference whose interval spans zero is not significant and "
                    "must be reported as such."
                ),
                "comparisons": statistics,
            },
        )

    combined_columns = [
        "dataset",
        "clients",
        "track",
        "method",
        "implementation_level",
        "budget",
        "scenario",
        "records",
    ] + MEAN_METRICS + [m + "_rate" for m in RATE_METRICS] + ["parse_failure_rate"]
    write_csv(ARTIFACTS / f"combined_results{suffix}.csv", combined_rows, combined_columns)

    main_rows = [
        row
        for row in combined_rows
        if row["budget"] == main_budget and row["scenario"] == "full"
    ]
    main_columns = [
        "dataset",
        "clients",
        "track",
        "method",
        "implementation_level",
        "records",
        "answer_exact_match",
        "answer_token_f1",
        "numeric_accuracy",
        "numeric_recall",
        "retrieval_recall@1",
        "retrieval_recall@3",
        "retrieval_recall@5",
        "evidence_recall",
        "evidence_precision",
        "citation_f1",
        "client_calls",
        "selected_pages",
        "protocol_bytes",
        "serialized_request_bytes",
        "serialized_response_bytes",
        "abstain_system_rate",
        "retrieval_ms",
    ]
    write_csv(ARTIFACTS / f"main_table{suffix}.csv", main_rows, main_columns)

    missing_columns = [
        "dataset",
        "clients",
        "track",
        "method",
        "budget",
        "paired_records",
        "answerable_success_evidence",
        "answerable_success_answer",
        "unanswerable_abstain_accuracy_system",
        "unanswerable_abstain_accuracy_routing",
        "false_abstention_rate_when_answerable",
        "wrong_answer_rate_when_unanswerable",
        "balanced_missingness_accuracy_evidence",
        "balanced_missingness_accuracy_answer",
    ]
    write_csv(ARTIFACTS / f"missingness_proxy{suffix}.csv", missing_rows, missing_columns)

    print(
        json.dumps(
            {
                "combined_rows": len(combined_rows),
                "main_rows": len(main_rows),
                "missingness_rows": len(missing_rows),
                "elapsed_seconds": round(time.time() - started, 1),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
