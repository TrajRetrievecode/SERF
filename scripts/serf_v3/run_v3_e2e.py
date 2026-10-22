#!/usr/bin/env python3
"""serf_v3 generate and score: run_e2e.py unchanged, artifact root artifacts/serf_v3, no pid files."""
import run_e2e

run_e2e.ARTIFACTS = run_e2e.ROOT / "artifacts" / "serf_v3"
run_e2e.record_pid = lambda *a, **k: None

if __name__ == "__main__":
    raise SystemExit(run_e2e.main())
