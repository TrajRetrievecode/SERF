#!/usr/bin/env python3
"""Step 6 of protocol_serf_v3_devconfirm.json: aggregate the budget-2 development run with the
unchanged definitions of aggregate_v3.py (which aggregated the SERF-v3 test run), once on the
confirmation population and once, separately labelled, on the unrestricted development portions.

evidence_success = a relevant unit among the cited pages and no system abstention (the evidence
criterion); retrieval_hit = a relevant unit among the cited pages, independent of the generator.
The generator decline rate of the protocol is abstain_system minus abstain_routing in the cells.

Before aggregating, predictions.jsonl must match plan.jsonl row for row. Both passes write into a
fresh staging directory; the six outputs are published only after both passes succeed, with the
completion marker devconfirm_unrestricted_paired_statistics.json last.
"""
import hashlib
import json
import os
import shutil
import sys
from collections import Counter

import aggregate_v3 as base
import pincheck
from e2e_common import ROOT

ART = ROOT / "artifacts" / "serf_v3_devconfirm"
STAGE = ART / ".agg_tmp"
STEMS = ("v3_cells.csv", "v3_missingness.csv", "v3_paired_statistics.json")
POPS = ("confirmation", "unrestricted")
KEY = lambda r: (r["config"], r["method"], r["budget"], r["scenario"], r["uid"])


def final_name(pop, stem):
    return f"devconfirm_{pop}_{stem.replace('v3_', '')}"


def read_predictions(ds):
    d = ART / f"{ds}_dev"
    fail = d / "failures.jsonl"
    if fail.exists() and fail.stat().st_size > 0:
        raise SystemExit(f"{ds}: rows without generation in failures.jsonl")
    with (d / "plan.jsonl").open(encoding="utf-8") as h:
        plan = Counter(KEY(json.loads(line)) for line in h if line.strip())
    rows = []
    with (d / "predictions.jsonl").open(encoding="utf-8") as h:
        for line in h:
            if line.strip():
                rows.append(json.loads(line))
    if Counter(KEY(r) for r in rows) != plan or max(plan.values()) != 1:
        raise SystemExit(f"{ds}: predictions do not match the plan row for row")
    if {r["budget"] for r in rows} != {2}:
        raise SystemExit(f"{ds}: expected budget-2 rows only")
    return rows


def main() -> int:
    pincheck.env_check("aggregate")
    protocol = json.load(open(sys.argv[1], encoding="utf-8"))
    marker = ART / final_name("unrestricted", "v3_paired_statistics.json")
    keep = {}
    for ds in protocol["data"]["plans"]:
        cp = protocol["data"]["confirmation_population"][ds]
        p = ROOT / cp["path"]
        if hashlib.sha256(p.read_bytes()).hexdigest() != cp["sha256"]:
            raise SystemExit(f"{ds}: confirmation list sha256 differs from the protocol")
        keep[ds] = {cfg: set(v) for cfg, v in json.load(open(p, encoding="utf-8"))["keep"].items()}
        if any(len(keep[ds][c]) != cp["per_setting"][c] for c in cp["per_setting"]) or set(keep[ds]) != set(cp["per_setting"]):
            raise SystemExit(f"{ds}: confirmation list does not match the registered counts")
    shutil.rmtree(STAGE, ignore_errors=True)
    STAGE.mkdir(parents=True)
    try:
        base.ART = STAGE    # aggregate_v3.main() writes to its module-level ART
        for pop in POPS:
            if pop == "confirmation":
                base.load = lambda ds: [r for r in read_predictions(ds) if r["uid"] in keep[ds][r["config"]]]
            else:
                base.load = read_predictions
            base.main()
            for s in STEMS:
                (STAGE / s).replace(STAGE / final_name(pop, s))
        pincheck.check("aggregate")
        names = [final_name(p, s) for p in POPS for s in STEMS]
        names.remove(marker.name)
        names.append(marker.name)   # the completion marker is published last
        digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        for n in names:             # first pass: any existing output must be byte-identical
            if (ART / n).exists() and digest(ART / n) != digest(STAGE / n):
                raise SystemExit(f"{n} exists and differs from the recomputed output; refusing to publish")
        for n in names:             # second pass: publish the missing outputs
            if not (ART / n).exists():
                os.replace(STAGE / n, ART / n)
    finally:
        shutil.rmtree(STAGE, ignore_errors=True)
    print(json.dumps({"outputs": names}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
