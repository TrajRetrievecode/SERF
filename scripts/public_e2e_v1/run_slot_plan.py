#!/usr/bin/env python3
"""Plan stage using the evaluation set's adapters instead of the reduced ones.

Everything except the adapters matches the main run: same federation, same
splits, same budgets, same scenarios, same shared retriever, same evidence
packaging rule and the same plan schema, so the generate and score stages of
``run_e2e.py`` consume this plan unchanged and the two runs are comparable
cell by cell.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from e2e_common import JsonlAppender, ROOT, TRACK, read_json, record_pid, write_json
from e2e_corpus import CACHE, page_text_map
from e2e_generation import request_for
from e2e_records import load_records
from e2e_retrieval import CERTIFICATE_BYTES_PER_CALL, ClientPageView, PageIndex
from e2e_slot_adapters import ADAPTERS, QueryViews
from e2e_slots import statistics
from e2e_routing import METHOD_LEVEL, origin_client
from e2e_common import package_bytes

from fedresolve_core import build_federation, split_records, stable_u64

ARTIFACTS = ROOT / "artifacts" / TRACK


class SlotQueries:
    def __init__(self, dataset: str, index: PageIndex, dense_weight: float):
        from scipy import sparse

        root = CACHE / f"{dataset}_slot_queries"
        manifest = read_json(root / "manifest.json")
        self.spans = [tuple(s) for s in manifest["spans"]]
        self.visual = np.load(root / "slot_visual.npy").astype(np.float32)
        self.sparse = sparse.load_npz(root / "slot_text.npz").tocsr()
        self.dense = np.load(root / "slot_dense.npy").astype(np.float32)
        self.index = index
        self.dense_weight = dense_weight
        self.decomposition = manifest["decomposition"]

    def page_scores(self, rows: slice, blend: float) -> np.ndarray:
        """Blended page scores for a contiguous block of slot texts."""
        visual = self.index.visual @ self.visual[rows].T
        sparse_text = np.asarray(
            (self.index.text @ self.sparse[rows].T).todense()
        )
        if self.index.dense is not None and self.dense_weight > 0.0:
            dense_text = self.index.dense @ self.dense[rows].T
            text = (
                self.dense_weight * dense_text
                + (1.0 - self.dense_weight) * sparse_text
            )
        else:
            text = sparse_text
        return (blend * visual + (1.0 - blend) * text).astype(np.float32).T


def online_mask(record, federation, scenario: str, num_clients: int, seed: int):
    online = np.ones(num_clients, dtype=bool)
    if scenario == "full":
        return online
    if scenario == "qrel_missing":
        for client in federation.gold_clients(record):
            online[client] = False
        return online
    raise ValueError(scenario)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["finrag", "visdom"])
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--out-suffix", default="_slot")
    parser.add_argument("--gold-protocol", default=None)
    args = parser.parse_args()

    protocol = read_json(Path(args.protocol))
    gold_protocol = read_json(
        Path(args.gold_protocol)
        if args.gold_protocol
        else ROOT / "configs" / "formal_gold_v1_protocol.json"
    )
    record_pid(f"run_slot_plan_{args.dataset}")

    records, _, inventory = load_records(args.dataset)
    by_uid = {r.uid: r for r in records}
    blend = float(protocol["retriever"]["blend_weight"][args.dataset])
    dense_weight = float(
        (protocol["retriever"].get("dense_weight") or {}).get(args.dataset, 0.0)
    )
    index = PageIndex.load(args.dataset, blend)
    if dense_weight > 0.0:
        index.attach_dense(args.dataset, dense_weight)
    slots = SlotQueries(args.dataset, index, dense_weight)
    texts = page_text_map(args.dataset)
    text_limit = int(protocol["evidence"]["page_text_char_limit"])
    rank_depth = int(protocol["evidence"]["ranked_candidate_depth"])
    seed = int(protocol["seed"])
    federation_inventory = sorted(set(index.doc_key_list) | set(inventory))

    out_dir = ARTIFACTS / (args.dataset + args.out_suffix)
    out_dir.mkdir(parents=True, exist_ok=True)
    plan_writer = JsonlAppender(out_dir / "plan.jsonl")
    configurations = {}
    started = time.time()
    written = 0

    for clients in protocol["client_scales"]:
        for track in protocol["tracks"]:
            key = f"c{clients}_{track}"
            layer_seed = seed + clients + (0 if args.dataset == "finrag" else 50_000)
            federation = build_federation(
                federation_inventory, num_clients=clients, seed=layer_seed
            )
            splits = split_records(records, track=track, seed=layer_seed + 101)
            view = ClientPageView(index, federation.doc_to_client, clients)
            configurations[key] = {
                "clients": clients,
                "track": track,
                "seed": layer_seed,
                "split_counts": {k: len(v) for k, v in splits.items()},
                "adapters": "evaluation-set transcription",
                "slot_decomposition": slots.decomposition,
            }

            uid_to_position = {r.uid: i for i, r in enumerate(records)}

            for record in splits["test"]:
                start, stop = slots.spans[uid_to_position[record.uid]]
                slot_page = slots.page_scores(slice(start, stop), blend)
                full_page = index.scores_for_rows(
                    record.uid, np.arange(len(index.page_uids), dtype=np.int64)
                )
                views = QueryViews(
                    full_page=full_page,
                    slot_page=slot_page,
                    rows_by_client=view.rows_by_client,
                    num_clients=clients,
                    claim_query_weight=gold_protocol["fedresolve"]["claim_query_weight"],
                    full_query_weight=gold_protocol["fedresolve"]["full_query_weight"],
                    temperature=gold_protocol["fedresolve"]["global_temperature"],
                )
                gold_rows = [
                    index.uid_to_row[u]
                    for u in record.gold_pages
                    if u in index.uid_to_row
                ] or [
                    r
                    for r in range(len(index.page_uids))
                    if index.doc_key_list[index.doc_keys[r]] in set(record.doc_keys)
                ][:8]

                # The single-slot view used by the "without slot decomposition"
                # ablation: the full query only, everything else identical.
                views_single = QueryViews(
                    full_page=full_page,
                    slot_page=full_page[None, :],
                    rows_by_client=view.rows_by_client,
                    num_clients=clients,
                    claim_query_weight=gold_protocol["fedresolve"]["claim_query_weight"],
                    full_query_weight=gold_protocol["fedresolve"]["full_query_weight"],
                    temperature=gold_protocol["fedresolve"]["global_temperature"],
                )

                for method in protocol["methods"]:
                    for budget in protocol["budgets"]:
                        for scenario in protocol["scenarios"]:
                            if scenario != "full" and method == "Oracle-evidence":
                                continue
                            if (
                                budget not in protocol["main_budgets"]
                                and scenario != "full"
                            ):
                                continue
                            online = online_mask(
                                record, federation, scenario, clients, seed
                            )
                            t0 = time.perf_counter()
                            result = ADAPTERS[method](
                                views_single if method == "FedResolve-noSlots" else views,
                                online,
                                budget,
                                gold_protocol,
                                origin_client=origin_client(record.uid, clients),
                                page_client=view.page_client,
                                visual=index.visual,
                                gold_clients=list(federation.gold_clients(record)),
                                gold_rows=gold_rows,
                            )
                            retrieval_ms = (time.perf_counter() - t0) * 1000.0

                            rows = result["rows"]
                            abstain = bool(result["abstain"])
                            evidence_uids = (
                                ()
                                if abstain
                                else tuple(index.page_uids[r] for r in rows)
                            )
                            ranked = np.argsort(-views.evidence_score)[:rank_depth]
                            request = request_for(
                                record.query, evidence_uids, texts, text_limit
                            )
                            calls = len(result["clients"])
                            if method == "Centralized":
                                calls = 1 if rows else 0
                            evidence_bytes = sum(
                                len(texts.get(u, "")[:text_limit].encode("utf-8"))
                                for u in evidence_uids
                            )
                            cost = {
                                "client_calls": calls,
                                "selected_pages": len(evidence_uids),
                                "serialized_request_bytes": package_bytes(
                                    {"query": record.query, "page_budget": budget}
                                )
                                * calls,
                                "serialized_response_bytes": sum(
                                    package_bytes(
                                        {
                                            "client": int(view.page_client[r]),
                                            "page_uid": index.page_uids[r],
                                            "certificate": {"score": 0.0, "available": True},
                                            "text": texts.get(index.page_uids[r], "")[
                                                :text_limit
                                            ],
                                        }
                                    )
                                    for r in rows
                                )
                                if not abstain
                                else CERTIFICATE_BYTES_PER_CALL * calls,
                                "certificate_bytes": CERTIFICATE_BYTES_PER_CALL * calls,
                                "protocol_bytes": CERTIFICATE_BYTES_PER_CALL * calls
                                + evidence_bytes,
                            }

                            plan_writer.write(
                                {
                                    "row_key": f"{key}|{method}|b{budget}|{scenario}|{record.uid}",
                                    "dataset": args.dataset,
                                    "config": key,
                                    "clients": clients,
                                    "track": track,
                                    "method": method,
                                    "implementation_level": METHOD_LEVEL.get(
                                        method, "framework_diagnostic"
                                    ),
                                    "budget": budget,
                                    "scenario": scenario,
                                    "uid": record.uid,
                                    "selected_clients": [int(c) for c in result["clients"]],
                                    "abstain_routing": abstain,
                                    "routing_reason": json.dumps(
                                        result["trace"], ensure_ascii=False
                                    )[:400],
                                    "offline_probability_peak": 0.0,
                                    "candidate_pages": int(
                                        sum(
                                            view.rows_by_client[c].size
                                            for c in result["clients"]
                                            if c in view.rows_by_client
                                        )
                                    ),
                                    "ranked_page_uids": [
                                        index.page_uids[int(r)] for r in ranked
                                    ],
                                    "evidence_page_uids": list(evidence_uids),
                                    "generation_key": None if abstain else request.key,
                                    "evidence_labels": list(request.evidence_labels),
                                    "cost": cost,
                                    "retrieval_ms": retrieval_ms,
                                    "cross_client_gold": len(
                                        federation.gold_clients(record)
                                    )
                                    > 1,
                                }
                            )
                            written += 1
                            if written % 5000 == 0:
                                print(
                                    json.dumps(
                                        {
                                            "stage": "slot_plan",
                                            "written": written,
                                            "config": key,
                                            "elapsed_seconds": round(
                                                time.time() - started, 1
                                            ),
                                        }
                                    ),
                                    flush=True,
                                )

    plan_writer.close()
    write_json(out_dir / "routing_configurations.json", configurations)
    summary = {
        "stage": "slot_plan",
        "dataset": args.dataset,
        "rows_written": written,
        "elapsed_seconds": time.time() - started,
    }
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
