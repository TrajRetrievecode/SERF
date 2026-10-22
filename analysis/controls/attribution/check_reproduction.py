"""Verify recorded SERF cells after full runs; smoke runs report subset status."""
import sys
sys.dont_write_bytecode = True
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
import attribution as a


def check(rows, references, partial=False):
    groups = defaultdict(list)
    for row in rows:
        if row["method"] == "FedResolve" and row["variant"] == "+sel":
            groups[(row["dataset"], row["config"], row["budget"])].append(row)
    comparisons = []
    for reference in references:
        key = (reference["dataset"], reference["config"], reference["budget"])
        values = groups.get(key)
        if not values:
            raise ValueError("Missing SERF reproduction cell: " + str(key))
        for metric in ("evidence_recall", "client_calls", "selected_pages", "protocol_bytes", "abstain_routing"):
            actual = float(np.mean([r[metric] for r in values]))
            expected = reference[metric]
            comparisons.append({"dataset": key[0], "config": key[1], "budget": key[2],
                "metric": metric, "actual": actual, "recorded": expected,
                "queries": len(values), "recorded_queries": reference["n"],
                "match": None if partial else len(values) == reference["n"]
                    and bool(np.isclose(actual, expected, rtol=1e-12, atol=1e-12))})
    return {"status": "subset_not_comparable" if partial else
            ("pass" if all(r["match"] for r in comparisons) else "fail"),
            "scope": "Recorded full-scenario SERF cells; no generation metrics",
            "comparisons": comparisons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    from e2e_common import ROOT
    out = a.output_dir(args.out, ROOT)
    rows = []
    for dataset in ("finrag", "visdom"):
        manifest = a.read(out / (dataset + "_manifest.json"))
        if args.limit is None and manifest["limit"] is not None:
            raise ValueError("Smoke inputs require an explicit --limit")
        if manifest["limit"] is not None and args.limit > manifest["limit"]:
            raise ValueError("Requested limit exceeds the available smoke subset")
        with (out / (dataset + ".jsonl")).open(encoding="utf-8") as handle:
            import json
            rows.extend(json.loads(line) for line in handle if line.strip())
    rows = a.limit_rows(rows, args.limit)
    result = check(rows, a.read(args.design)["recorded_serf_cells"], args.limit is not None)
    a.save(out / "reproduction.json", result)
    if result["status"] == "fail":
        raise SystemExit("Recorded SERF cell reproduction failed; see reproduction.json")
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()

