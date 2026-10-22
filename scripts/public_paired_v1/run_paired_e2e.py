#!/usr/bin/env python3
"""public_paired_v1 generate and score stages.

Runs run_e2e.py (public_e2e_v1) unchanged except that its artifact root points at
artifacts/public_paired_v1 and its pid bookkeeping is disabled, so nothing is
written into the public_e2e_v1 tree.
"""
import run_e2e

run_e2e.ARTIFACTS = run_e2e.ROOT / "artifacts" / "public_paired_v1"
run_e2e.record_pid = lambda *a, **k: None

if __name__ == "__main__":
    raise SystemExit(run_e2e.main())
