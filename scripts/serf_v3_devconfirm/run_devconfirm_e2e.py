#!/usr/bin/env python3
"""Generate or score the development confirmation (protocol_serf_v3_devconfirm.json, steps 3 and 5):
run_e2e.py unchanged, artifact root artifacts/serf_v3_devconfirm, no pid files.

Before the stage: the environment must equal the registration record, every loaded project module
must be pinned, and the gold-label digest (and, for generation, the page-text digest) must equal the
registered digests. After the stage returns (normally or through SystemExit) the same digests and the
pinned-module check are repeated; any failure exits non-zero.
"""
import argparse
import json

import digest_util
import pincheck
import run_e2e

run_e2e.ARTIFACTS = run_e2e.ROOT / "artifacts" / "serf_v3_devconfirm"
run_e2e.record_pid = lambda *a, **k: None
CONF = run_e2e.ROOT / "configs" / "serf_v3_devconfirm"


def check_digests(ds, stages, when):
    names = [("gold_digest", digest_util.gold_digest)]
    if "generate" in stages:
        names.append(("pagetext_digest", digest_util.pagetext_digest))
    for name, fn in names:
        if fn(ds) != json.load(open(CONF / f"{name}_{ds}.json", encoding="utf-8")):
            raise SystemExit(f"{ds}: {name} differs from the registered digest ({when})")


if __name__ == "__main__":
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--dataset", required=True)
    p.add_argument("--stages", nargs="+", required=True)
    known = p.parse_known_args()[0]
    pincheck.env_check("run_devconfirm_e2e")
    check_digests(known.dataset, known.stages, "before")
    pincheck.check("run_devconfirm_e2e:start")
    try:
        rc = run_e2e.main()
    except SystemExit as e:
        rc = e.code
    check_digests(known.dataset, known.stages, "after")
    pincheck.check("run_devconfirm_e2e:end")
    raise SystemExit(rc)
