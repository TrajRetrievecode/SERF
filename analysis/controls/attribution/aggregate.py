"""Aggregate attribution rows into per-setting means and paired statistics."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import numpy as np

HERE = Path(__file__).resolve().parent
from e2e_common import ROOT
import e2e_stats
from e2e_stats import paired_bootstrap, mcnemar
import attribution as a
from check_reproduction import check

KEYS = ("dataset", "config", "method", "variant", "budget", "scenario")
PUBLIC_METRICS = ("evidence_recall", "evidence_precision", "evidence_f1", "complete_evidence",
    "abstain", "abstain_routing", "client_calls", "selected_pages", "certificate_bytes",
    "serialized_request_bytes", "serialized_response_bytes", "protocol_bytes")
GOLD_METRICS = ("strict_success", "evidence_recall", "evidence_precision", "citation_f1",
    "client_calls", "bytes", "complete_evidence", "complete_client_set", "abstain", "selected_pages")


def panel():
    return [(method, variant) for method in a.METHODS
            for variant in a.VARIANTS + (("topB",) if method == "FedResolve" else ())]


def grouped_rows(rows):
    groups = defaultdict(dict)
    for row in rows:
        key = tuple(row[k] for k in KEYS)
        if row["uid"] in groups[key]:
            raise ValueError("Duplicate query cell: " + str(key) + " " + row["uid"])
        groups[key][row["uid"]] = row
    return groups


def validate_rows(rows, manifests, limit):
    groups = grouped_rows(rows)
    expected_keys = set()
    for dataset in ("finrag", "visdom", "evaluation"):
        manifest = manifests[dataset]
        if manifest["limit"] != limit:
            raise ValueError("Input manifests differ from requested full/smoke mode")
        settings = (["c8_evaluation"] if dataset == "evaluation" else
                    [f"c{c}_{t}" for c in (8, 16) for t in ("seen_query", "unseen_family")])
        for config in settings:
            expected_n = manifest["queries"] if dataset == "evaluation" else next(
                c["queries"] for c in manifest["configurations"] if c["config"] == config)
            if dataset == "evaluation" and expected_n != (45 if limit is None else min(limit, 45)):
                raise ValueError("Incorrect evaluation question count")
            if dataset != "evaluation":
                entry = next(c for c in manifest["configurations"] if c["config"] == config)
                if expected_n != (entry["split_counts"]["test"] if limit is None
                                  else min(limit, entry["split_counts"]["test"])):
                    raise ValueError("Public test count differs from manifest")
            reference_uids = None
            for budget in ((2, 3) if dataset == "evaluation" else (1, 2, 3)):
                for method, variant in panel():
                    key = (dataset, config, method, variant, budget, "full")
                    expected_keys.add(key)
                    if key not in groups or len(groups[key]) != expected_n or expected_n == 0:
                        raise ValueError("Missing or incomplete panel cell: " + str(key))
                    uids = set(groups[key])
                    if reference_uids is None:
                        reference_uids = uids
                    elif uids != reference_uids:
                        raise ValueError("UID pairing mismatch: " + str(key))
                    for row in groups[key].values():
                        metrics = GOLD_METRICS if dataset == "evaluation" else PUBLIC_METRICS
                        for metric in metrics:
                            if not np.isfinite(float(row[metric])):
                                raise ValueError("Nonfinite metric: " + metric)
                        if row["client_calls"] != len(row["selected_clients"]):
                            raise ValueError("Call count mismatch")
                        if row["selected_pages"] != len(row["selected_page_uids"]):
                            raise ValueError("Page count mismatch")
    if set(groups) != expected_keys:
        raise ValueError("Unexpected panel cells")
    return groups


def summaries(groups):
    cells = []
    for key, rows in sorted(groups.items()):
        metrics = GOLD_METRICS if key[0] == "evaluation" else PUBLIC_METRICS
        cells.append({**dict(zip(KEYS, key)), "n": len(rows),
            **{metric: float(np.mean([r[metric] for r in rows.values()])) for metric in metrics}})
    macro_groups = defaultdict(list)
    for cell in cells:
        if cell["dataset"] != "evaluation":
            key = tuple(cell[k] for k in ("dataset", "method", "variant", "budget", "scenario"))
            macro_groups[key].append(cell)
    macros = []
    for key, values in sorted(macro_groups.items()):
        if len(values) != 4:
            raise ValueError("Four settings required for macro mean")
        macros.append({**dict(zip(("dataset", "method", "variant", "budget", "scenario"), key)),
            "settings": 4, "n_sum": sum(v["n"] for v in values),
            **{metric: float(np.mean([v[metric] for v in values])) for metric in PUBLIC_METRICS}})
    return cells, macros


def statistics(groups, design):
    output = []
    for key, control in sorted(groups.items()):
        dataset, config, method, variant, budget, scenario = key
        if (method, variant) == ("FedResolve", "+sel"):
            continue
        treatment = groups[(dataset, config, "FedResolve", "+sel", budget, scenario)]
        if set(treatment) != set(control):
            raise ValueError("Paired statistics require identical UIDs")
        uids = sorted(treatment)
        seed = design["seeds"]["evaluation_bootstrap" if dataset == "evaluation" else "public_bootstrap"]
        base = {"dataset": dataset, "config": config, "budget": budget, "scenario": scenario,
                "treatment": "FedResolve", "treatment_variant": "+sel",
                "control": method, "control_variant": variant}
        for metric in (("evidence_recall",) if dataset == "evaluation" else ("evidence_recall", "client_calls")):
            output.append({**base, "metric": metric, "statistic": "paired_bootstrap",
                **paired_bootstrap([treatment[u][metric] for u in uids],
                                   [control[u][metric] for u in uids],
                                   iterations=design["statistics"]["resamples"], seed=seed)})
        if dataset == "evaluation":
            output.append({**base, "metric": "strict_success", "statistic": "mcnemar", "n": len(uids),
                **mcnemar([treatment[u]["strict_success"] for u in uids],
                          [control[u]["strict_success"] for u in uids])})
    counts = defaultdict(list)
    for row in output:
        if row["dataset"] != "evaluation":
            key = tuple(row[k] for k in ("dataset", "budget", "control", "control_variant", "metric"))
            counts[key].append(row)
    count_rows = []
    for key, values in sorted(counts.items()):
        count_rows.append({**dict(zip(("dataset", "budget", "control", "control_variant", "metric"), key)),
            "settings": len(values), "intervals_excluding_zero": sum(v["significant_at_95"] for v in values),
            "positive_intervals": sum(v["ci_low"] > 0 for v in values),
            "negative_intervals": sum(v["ci_high"] < 0 for v in values)})
    return output, count_rows


def write_table(out, name, rows):
    a.save(out / (name + ".json"), rows)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (out / (name + ".csv")).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                         for k, v in row.items()} for row in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "artifacts/controls/attribution")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/controls/attribution/results")
    parser.add_argument("--design", type=Path, default=HERE / "protocol_fairness_v1_A_attribution.json")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if not args.out.resolve().is_relative_to((ROOT / "artifacts/controls/attribution").resolve()):
        parser.error("Local results must remain inside the job directory")
    design = a.read(args.design)
    a.pin_modules(design, {"e2e_stats": e2e_stats})
    manifests, rows = {}, []
    for dataset in ("finrag", "visdom", "evaluation"):
        manifest = a.read(args.input / (dataset + "_manifest.json"))
        if manifest["design_sha256"] != a.digest(args.design):
            raise ValueError("Manifest does not match the protocol design")
        manifests[dataset] = manifest
        with (args.input / (dataset + ".jsonl")).open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    source_limits = {m["limit"] for m in manifests.values()}
    if len(source_limits) != 1:
        raise ValueError("Mixed full and smoke inputs")
    source_limit = source_limits.pop()
    validate_rows(rows, manifests, source_limit)
    if args.limit is None and source_limit is not None:
        raise ValueError("Smoke inputs require an explicit --limit")
    if source_limit is not None and args.limit > source_limit:
        raise ValueError("Requested limit exceeds the available smoke subset")
    rows = a.limit_rows(rows, args.limit)
    for dataset, manifest in manifests.items():
        manifest["limit"] = args.limit
        if args.limit is not None:
            if dataset == "evaluation":
                manifest["queries"] = min(args.limit, manifest["queries"])
            else:
                for config in manifest["configurations"]:
                    config["queries"] = min(args.limit, config["queries"])
    groups = validate_rows(rows, manifests, args.limit)
    reproduction = check(rows, design["recorded_serf_cells"], args.limit is not None)
    args.out.mkdir(parents=True, exist_ok=True)
    a.save(args.out / "reproduction.json", reproduction)
    if reproduction["status"] == "fail":
        raise SystemExit("SERF reproduction failed; final aggregation stopped")
    cells, macros = summaries(groups)
    stats, counts = statistics(groups, design)
    for name, values in (("per_setting", cells), ("mean_four_settings", macros),
                         ("paired_statistics", stats), ("interval_setting_counts", counts)):
        write_table(args.out, name, values)
    write_table(args.out, "evaluation_per_question", [r for r in rows if r["dataset"] == "evaluation"])
    a.save(args.out / "aggregation_manifest.json", {"design_sha256": a.digest(args.design),
        "limit": args.limit, "status": "smoke_subset" if args.limit is not None else "full",
        "rows": len(rows), "cells": len(cells), "bootstrap_resamples": design["statistics"]["resamples"],
        "input_sha256": {p.name: a.digest(p) for p in sorted(args.input.glob("*.jsonl"))}})
    print(json.dumps({"rows": len(rows), "cells": len(cells), "reproduction": reproduction["status"]}))


if __name__ == "__main__":
    main()

