"""Run cached evaluation scores through the package runner and attribution panel."""
import sys
sys.dont_write_bytecode = True
import argparse
import os
import json
import time
from pathlib import Path
import numpy as np
import attribution as a
import run_formal_gold_v1 as rf
from e2e_common import ROOT


def find_selector(root, explicit, expected_hash):
    if explicit:
        return Path(explicit)
    path = root / "analysis/gold_v3.py"
    if not path.is_file() or a.digest(path) != expected_hash:
        raise FileNotFoundError("Package evaluation selector missing or hash mismatch")
    return path


def load_inputs(tasks, cache):
    images, queries = a.read(cache / "images.json"), a.read(cache / "queries.json")
    ie = np.load(cache / "image_embeddings.npy").astype(np.float64)
    qe = np.load(cache / "query_embeddings.npy").astype(np.float64)
    ie /= np.maximum(np.linalg.norm(ie, axis=1, keepdims=True), 1e-12)
    qe /= np.maximum(np.linalg.norm(qe, axis=1, keepdims=True), 1e-12)
    ii = {row["candidate_id"]: i for i, row in enumerate(images)}
    qi = {row["query_key"]: i for i, row in enumerate(queries)}
    image_paths = {row["candidate_id"]: Path(row["path"]) for row in images}
    outputs = {}
    for task in tasks:
        cid = task["canonical_id"]
        candidates = [row["candidate_id"] for row in task["candidate_pages"]]
        matrix = ie[[ii[v] for v in candidates]]
        slot_rows = sorted([i for key, i in qi.items()
            if key.startswith(cid + "::") and not key.endswith("::full")],
            key=lambda i: queries[i]["slot_id"])
        outputs[cid] = {"canonical_id": cid, "candidate_ids": candidates,
            "candidate_clients": [int(row["client_id"]) for row in task["candidate_pages"]],
            "candidate_bytes": [image_paths[v].stat().st_size for v in candidates],
            "full_scores": matrix @ qe[qi[cid + "::full"]],
            "slot_scores": np.vstack([matrix @ qe[i] for i in slot_rows]),
            "image_similarity": matrix @ matrix.T}
    return outputs


def decisions(inputs_by_id, task_ids, protocol, select):
    online = np.ones(len(protocol["federation"]["client_ids"]), dtype=bool)
    rows = []
    for uid in task_ids:
        inputs = inputs_by_id[uid]
        views = rf.score_views(inputs, protocol)
        for budget in (2, 3):
            for method, variant, selection in a.gold_variants(rf, select, inputs, views, online, budget, protocol):
                rows.append({"dataset": "evaluation", "config": "c8_evaluation", "clients": 8,
                    "track": "evaluation", "method": method, "variant": variant, "budget": budget,
                    "scenario": "full", "uid": uid, "selection": selection})
    return rows


def evaluate(rows, inputs_by_id, annotations, certificate_bytes):
    # This is the sole label-consuming boundary. No annotation field is
    # inspected here; the fixed item_metrics receives the annotation.
    for row in rows:
        selection = row["selection"]
        metrics = rf.item_metrics(selection, annotations[row["uid"]],
                                  inputs_by_id[row["uid"]], certificate_bytes)
        evidence = list(selection["evidence_ids"])
        calls = len(selection["queried_clients"])
        yield {**row, **metrics,
            "selected_clients": selection["queried_clients"],
            "selected_page_uids": evidence, "evidence_page_uids": evidence,
            "selected_pages": len(evidence), "complete_evidence": metrics["evidence_recall"] == 1.0,
            "abstain_routing": bool(selection["abstain"]),
            "cost": {"client_calls": calls, "selected_pages": len(evidence),
                "certificate_bytes": certificate_bytes * calls,
                "protocol_bytes": metrics["bytes"], "bytes": metrics["bytes"],
                "serialized_request_bytes": None, "serialized_response_bytes": None},
            "certificate_bytes": certificate_bytes * calls, "protocol_bytes": metrics["bytes"],
            "serialized_request_bytes": None, "serialized_response_bytes": None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--tasks", type=Path)
    parser.add_argument("--gold", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--selector", type=Path)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    root = args.root
    protocol_path = args.protocol or root / "configs/formal_gold_v1_protocol.json"
    tasks_path = args.tasks or Path(os.environ.get("SERF_GOLD_TASKS", str(root / "data/annotation/gold_v1/tasks_gold_v1.json")))
    gold_path = args.gold or root / "data/annotation/gold_v1/fedfinrag_resolve_gold_v1.json"
    cache = args.cache or Path(os.environ.get("SERF_GOLD_CACHE", str(root / "artifacts/gold_v3/cache")))
    design, protocol = a.read(args.design), a.read(protocol_path)
    a.pin_modules(design, {"run_formal_gold_v1": rf})
    if a.digest(protocol_path) != design["evaluation_protocol_sha256"]:
        raise ValueError("Evaluation evaluation protocol hash mismatch")
    for path, key in ((gold_path, "gold_json_sha256"), (tasks_path, "tasks_json_sha256")):
        if a.digest(path) != protocol["dataset"][key]:
            raise ValueError("Evaluation input hash mismatch: " + key)
    recorded = a.read(root / "artifacts/formal_gold_v1/formal_results.json")
    if a.digest(cache / "manifest.json") != recorded["cache_manifest_sha256"]:
        raise ValueError("Evaluation cache manifest hash mismatch")
    manifest = a.read(cache / "manifest.json")
    for name, expected in manifest["artifacts"].items():
        if a.digest(cache / name) != expected:
            raise ValueError("Evaluation cache artifact hash mismatch: " + name)
    selector_path = find_selector(root, args.selector, design["source_sha256"]["gold_v3"])
    select = a.import_selector(selector_path, design["source_sha256"]["gold_v3"])
    tasks = a.read(tasks_path)
    if len(tasks) != protocol["dataset"]["task_count"]:
        raise ValueError("Expected 45 fixed evaluation tasks")
    out = a.output_dir(args.out, ROOT)
    started = time.time()
    if args.limit is not None:
        tasks = tasks[:args.limit]
    inputs = load_inputs(tasks, cache)
    # Freeze every decision before loading annotation content.
    selected = decisions(inputs, [t["canonical_id"] for t in tasks], protocol, select)
    with (out / "evaluation_decisions.jsonl").open("x", encoding="utf-8") as writer:
        for row in selected:
            writer.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    annotations = {r["canonical_id"]: r for r in a.read(gold_path)["annotations"]}
    with (out / "evaluation.jsonl").open("x", encoding="utf-8") as writer:
        for row in evaluate(selected, inputs, annotations,
                            protocol["federation"]["certificate_bytes_per_client_call"]):
            writer.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    a.save(out / "evaluation_manifest.json", {"limit": args.limit, "queries": len(tasks),
        "rows": len(selected), "design_sha256": a.digest(args.design),
        "runner_sha256": a.digest(rf.__file__), "selector_path": str(selector_path),
        "selector_sha256": a.digest(selector_path), "protocol_sha256": a.digest(protocol_path),
        "cache_manifest_sha256": a.digest(cache / "manifest.json"),
        "elapsed_seconds": time.time() - started})
    print(json.dumps({"queries": len(tasks), "rows": len(selected)}), flush=True)


if __name__ == "__main__":
    main()

