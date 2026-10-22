#!/usr/bin/env python3
"""Step 2 of protocol_serf_v3_devconfirm.json: copy the budget-2 rows of every method and scenario
of each registered development plan to artifacts/serf_v3_devconfirm/<dataset>_dev/plan.jsonl.

The source plans are verified against the protocol's sha256. All checks run before anything is
written: every (method, scenario) cell of the 8 methods x 3 scenarios exists and holds exactly the
registered number of development queries summed over settings. The copy is written to a temporary
file and renamed. If the destination exists, it is kept when byte-identical and rejected otherwise.
"""
import hashlib
import json
import sys
from collections import Counter

import pincheck
from e2e_common import ROOT

pincheck.env_check("genplan")
pincheck.check("genplan")
METHODS = ("All", "Centralized", "FedResolve", "FedResolve-v3", "Oracle-client", "RAGRoute", "S2G-RAG", "S2G-RAG+avail")
SCENARIOS = ("full", "qrel_missing", "unneeded_offline")

protocol = json.load(open(sys.argv[1], encoding="utf-8"))
for ds, spec in protocol["data"]["plans"].items():
    src = ROOT / spec["path"]
    if hashlib.sha256(src.read_bytes()).hexdigest() != spec["sha256"]:
        raise SystemExit(f"{ds}: plan sha256 differs from the protocol")
    expected = sum(spec["dev_queries_per_setting"].values())
    rows, counts = [], Counter()
    with src.open(encoding="utf-8") as fin:
        for line in fin:
            r = json.loads(line)
            if r["budget"] != 2:
                continue
            rows.append(line if line.endswith("\n") else line + "\n")
            counts[(r["method"], r["scenario"])] += 1
    want = {(m, sc) for m in METHODS for sc in SCENARIOS}
    if set(counts) != want:
        raise SystemExit(f"{ds}: cells {sorted(set(counts) ^ want)} differ from the registered methods and scenarios")
    bad = {k: v for k, v in counts.items() if v != expected}
    if bad:
        raise SystemExit(f"{ds}: cells with a count other than {expected}: {bad}")
    data = "".join(rows).encode("utf-8")
    digest = hashlib.sha256(data).hexdigest()
    out_dir = ROOT / "artifacts" / "serf_v3_devconfirm" / f"{ds}_dev"
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / "plan.jsonl"
    if dst.exists():
        if hashlib.sha256(dst.read_bytes()).hexdigest() != digest:
            raise SystemExit(f"{dst} exists and differs from the registered budget-2 rows")
        print(json.dumps({"dataset": ds, "skipped": "identical plan exists", "sha256": digest}))
        continue
    tmp = out_dir / "plan.jsonl.tmp"
    tmp.unlink(missing_ok=True)
    tmp.write_bytes(data)
    tmp.replace(dst)
    print(json.dumps({"dataset": ds, "rows": len(rows), "cells": len(counts), "rows_per_cell": expected, "sha256": digest}))
