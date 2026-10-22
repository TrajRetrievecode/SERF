#!/usr/bin/env python3
"""Bind the development-confirmation protocol to the plans of this run.

Writes artifacts/serf_v3_devconfirm/runtime_protocol.json, a copy of
configs/serf_v3_devconfirm/protocol_serf_v3_devconfirm.json whose plan hashes are
those of the development plans written by scripts/serf_v3/run_v3_plan.py --split dev,
and records the current environment and file manifest that pincheck.env_check reads.
Every membership, digest, constant and analysis rule of the protocol is unchanged.
"""
import hashlib
import json
import os
from pathlib import Path

import pincheck

ROOT = Path(os.environ.get("SERF_ROOT", str(Path(__file__).resolve().parents[2])))
CONF = ROOT / "configs" / "serf_v3_devconfirm"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    protocol = json.load(open(CONF / "protocol_serf_v3_devconfirm.json", encoding="utf-8"))
    for spec in protocol["data"]["plans"].values():
        spec["sha256"] = sha256(ROOT / spec["path"])
    runtime = ROOT / "artifacts" / "serf_v3_devconfirm" / "runtime_protocol.json"
    runtime.parent.mkdir(parents=True, exist_ok=True)
    runtime.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (CONF / "ENVIRONMENT.json").write_text(json.dumps(pincheck.environment(), indent=2) + "\n", encoding="utf-8")
    files = sorted({q.resolve() for tree in ("src", "scripts", "configs") for q in (ROOT / tree).rglob("*") if q.is_file()})
    files += [runtime.resolve()] + [(ROOT / spec["path"]).resolve() for spec in protocol["data"]["plans"].values()]
    (CONF / "REQUIRED").write_text("".join(f"{q}\n" for q in files), encoding="utf-8")
    (CONF / "SHA256SUMS").write_text(
        "".join(f"{sha256(q)}  {q}\n" for q in files if q.name not in ("REQUIRED", "SHA256SUMS")), encoding="utf-8")
    print(runtime)


if __name__ == "__main__":
    main()
