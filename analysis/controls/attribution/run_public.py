"""Score each test query once per setting; apply the fixed attribution panel."""
import sys
sys.dont_write_bytecode = True
import argparse
import json
import time
from pathlib import Path
import numpy as np
import attribution as a
import run_v3_plan as upstream
import e2e_slot_adapters
import e2e_v3_adapter
import e2e_records
import e2e_retrieval
import fedresolve_core
from e2e_common import ROOT, package_bytes, prf
from e2e_generation import request_for
from e2e_corpus import page_text_map
from e2e_records import load_records
from e2e_retrieval import PageIndex, ClientPageView, CERTIFICATE_BYTES_PER_CALL
from e2e_slot_adapters import QueryViews
from run_v3_plan import SlotQueries, online_mask
from fedresolve_core import build_federation, split_records, stable_u64


def metric_fields(record, evidence):
    gold = record.gold_pages if record.gold_unit == "page" else record.doc_keys
    units = evidence if record.gold_unit == "page" else [u.rsplit("#p", 1)[0] for u in evidence]
    metrics = prf(units, gold)
    if not np.isfinite(metrics["recall"]):
        raise ValueError("Query has no protocol gold evidence: " + record.uid)
    return {"evidence_recall": metrics["recall"], "evidence_precision": metrics["precision"],
        "evidence_f1": metrics["f1"], "complete_evidence": metrics["recall"] == 1.0}


def plan_row(dataset, config, clients, track, record, view, index, texts, protocol,
             method, variant, budget, result, ranked, elapsed):
    rows = result["rows"]
    abstain = bool(result["abstain"])
    evidence = [] if abstain else [index.page_uids[r] for r in rows]
    calls = len(result["clients"])
    text_limit = protocol["evidence"]["page_text_char_limit"]
    request = request_for(record.query, tuple(evidence), texts, text_limit)
    evidence_bytes = sum(len(texts.get(u, "")[:text_limit].encode("utf-8")) for u in evidence)
    cost = {"client_calls": calls, "selected_pages": len(evidence),
        "serialized_request_bytes": package_bytes({"query": record.query, "page_budget": budget}) * calls,
        "serialized_response_bytes": sum(package_bytes({"client": int(view.page_client[r]),
            "page_uid": index.page_uids[r], "certificate": {"score": 0.0, "available": True},
            "text": texts.get(index.page_uids[r], "")[:text_limit]}) for r in rows)
            if not abstain else CERTIFICATE_BYTES_PER_CALL * calls,
        "certificate_bytes": CERTIFICATE_BYTES_PER_CALL * calls,
        "protocol_bytes": CERTIFICATE_BYTES_PER_CALL * calls + evidence_bytes}
    return {"row_key": f"{dataset}|{config}|{method}|{variant}|b{budget}|full|{record.uid}",
        "dataset": dataset, "config": config, "clients": clients, "track": track,
        "method": method, "variant": variant, "budget": budget, "scenario": "full", "uid": record.uid,
        "selected_clients": [int(c) for c in result["clients"]],
        "selected_page_uids": evidence, "evidence_page_uids": evidence,
        "selection_before_abstention": [index.page_uids[r] for r in rows],
        "abstain": abstain, "abstain_routing": abstain, "trace": result["trace"],
        "routing_reason": json.dumps(result["trace"], ensure_ascii=False)[:400],
        "offline_probability_peak": 0.0,
        "candidate_pages": int(sum(view.rows_by_client[c].size for c in result["clients"])),
        "ranked_page_uids": [index.page_uids[int(r)] for r in ranked],
        "generation_key": None if abstain else request.key, "evidence_labels": list(request.evidence_labels),
        "cost": cost, **cost, "retrieval_ms": elapsed, **metric_fields(record, evidence)}


def run_setting(dataset, clients, track, records, inventory, index, slots, texts,
                protocol, gold, limit, writer):
    seed = protocol["seed"] + clients + (0 if dataset == "finrag" else 50000)
    federation = build_federation(sorted(set(index.doc_key_list) | set(inventory)),
        num_clients=clients, seed=seed)
    splits = split_records(records, track=track, seed=seed + 101)
    view = ClientPageView(index, federation.doc_to_client, clients)
    positions = {r.uid: i for i, r in enumerate(records)}
    selected = splits["test"] if limit is None else splits["test"][:limit]
    config = f"c{clients}_{track}"
    count = 0
    for record in selected:
        t0 = time.perf_counter()
        start, stop = slots.spans[positions[record.uid]]
        slot_page = slots.page_scores(slice(start, stop), protocol["retriever"]["blend_weight"][dataset])
        full_page = index.scores_for_rows(record.uid, np.arange(len(index.page_uids), dtype=np.int64))
        views = QueryViews(full_page, slot_page, view.rows_by_client, clients,
            gold["fedresolve"]["claim_query_weight"], gold["fedresolve"]["full_query_weight"],
            gold["fedresolve"]["global_temperature"])
        score_ms = (time.perf_counter() - t0) * 1000
        online = online_mask(record, federation, "full", clients, protocol["seed"])
        ranked = np.argsort(-views.evidence_score)[:protocol["evidence"]["ranked_candidate_depth"]]
        for budget in (1, 2, 3):
            for method, variant, result in a.public_variants(views, online, budget, gold):
                row = plan_row(dataset, config, clients, track, record, view, index, texts, protocol,
                    method, variant, budget, result, ranked, score_ms)
                writer.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                count += 1
    return {"config": config, "seed": seed, "split_seed": seed + 101,
        "split_counts": {k: len(v) for k, v in splits.items()}, "queries": len(selected), "rows": count,
        "slot_decomposition": slots.decomposition}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("finrag", "visdom"), required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/public_e2e_v1/protocol_v3.json")
    parser.add_argument("--gold-protocol", type=Path, default=ROOT / "configs/formal_gold_v1_protocol.json")
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    protocol, gold, design = a.read(args.protocol), a.read(args.gold_protocol), a.read(args.design)
    a.validate_design(design, protocol, gold)
    if a.digest(args.protocol) != design["input_protocol_sha256"]["PROTOCOL_v3.json"]:
        raise ValueError("Protocol public protocol hash mismatch")
    if a.digest(args.gold_protocol) != upstream.FROZEN_GOLD_PROTOCOL_SHA256:
        raise ValueError("Evaluation protocol hash mismatch")
    a.pin_modules(design, {"run_v3_plan": upstream, "e2e_slot_adapters": e2e_slot_adapters,
        "e2e_v3_adapter": e2e_v3_adapter, "e2e_records": e2e_records,
        "e2e_retrieval": e2e_retrieval, "fedresolve_core": fedresolve_core})
    out = a.output_dir(args.out, ROOT)
    started = time.time()
    records, _, inventory = load_records(args.dataset)
    blend = protocol["retriever"]["blend_weight"][args.dataset]
    dense = protocol["retriever"]["dense_weight"][args.dataset]
    index = PageIndex.load(args.dataset, blend)
    if dense > 0:
        index.attach_dense(args.dataset, dense)
    slots = SlotQueries(args.dataset, index, dense)
    texts = page_text_map(args.dataset)
    configs = []
    with (out / (args.dataset + ".jsonl")).open("x", encoding="utf-8") as writer:
        for clients in (8, 16):
            for track in ("seen_query", "unseen_family"):
                configs.append(run_setting(args.dataset, clients, track, records, inventory, index,
                    slots, texts, protocol, gold, args.limit, writer))
                print(json.dumps(configs[-1]), flush=True)
    a.save(out / (args.dataset + "_manifest.json"), {"dataset": args.dataset,
        "design_sha256": a.digest(args.design), "protocol_sha256": a.digest(args.protocol),
        "gold_protocol_sha256": a.digest(args.gold_protocol), "limit": args.limit,
        "configurations": configs, "elapsed_seconds": time.time() - started})


if __name__ == "__main__":
    main()

