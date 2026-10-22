#!/usr/bin/env python3
"""Re-select the shared retriever against the metric the protocol delivers.

The first selection maximised whole-corpus recall@5.  That is the wrong
target here: a called client returns its single best page, so what decides
whether evidence reaches the generator is "within the client that owns the
relevant document, is the top-scoring page a relevant one".  Those two
metrics disagree on VisDoM, where the whole-corpus score improved while the
delivered evidence got worse.

This job measures the delivered quantity directly, still on training records
only and still with a rule fixed before the numbers are seen: maximise the
mean delivered-evidence hit rate across the four (client scale, track)
federations, with the whole-corpus recall@5 as the tie-break.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from e2e_common import ROOT, TRACK, recall_at_k, record_pid, write_json
from e2e_records import load_records
from e2e_retrieval import ClientPageView, PageIndex

from fedresolve_core import build_federation, split_records, stable_u64


def training_only(records, dataset: str, seed: int):
    offset = 0 if dataset == "finrag" else 50_000
    train: set[str] = set()
    test: set[str] = set()
    for track in ("seen_query", "unseen_family"):
        for clients in (8, 16):
            splits = split_records(
                records, track=track, seed=seed + clients + offset + 101
            )
            train.update(r.uid for r in splits["train"])
            test.update(r.uid for r in splits["test"])
    return sorted(train - test)


def stratified_sample(records, usable: list[str], size: int, seed: int):
    """Round-robin over (language, category) so the sample is representative.

    Taking the first N of a sorted uid list is not: VisDoM uids begin with the
    task name, so the alphabetically first task would fill the whole sample.
    """
    by_uid = {r.uid: r for r in records}
    strata: dict[tuple, list[str]] = {}
    for uid in usable:
        record = by_uid[uid]
        strata.setdefault((record.lang, record.category), []).append(uid)
    for uids in strata.values():
        uids.sort(key=lambda uid: stable_u64(uid, seed))
    order = sorted(strata)
    picked: list[str] = []
    depth = 0
    while len(picked) < size and any(len(strata[s]) > depth for s in order):
        for stratum in order:
            if len(picked) >= size:
                break
            if len(strata[stratum]) > depth:
                picked.append(strata[stratum][depth])
        depth += 1
    return [by_uid[uid] for uid in picked]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["finrag", "visdom"])
    parser.add_argument("--weights", type=float, nargs="+", default=[0.0, 0.25, 0.5])
    parser.add_argument("--dense-weights", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--sample", type=int, default=400)
    parser.add_argument("--budget", type=int, default=2)
    parser.add_argument("--output-tag", default="_delivered")
    args = parser.parse_args()

    record_pid(f"select_retriever_delivered_{args.dataset}")
    records, _, inventory = load_records(args.dataset)
    by_uid = {r.uid: r for r in records}
    usable = training_only(records, args.dataset, args.seed)
    sample = stratified_sample(records, usable, args.sample, args.seed)

    index = PageIndex.load(args.dataset, 0.5)
    if any(w > 0 for w in args.dense_weights):
        index.attach_dense(args.dataset, max(args.dense_weights))
    federation_inventory = sorted(set(index.doc_key_list) | set(inventory))
    unit = records[0].gold_unit
    offset = 0 if args.dataset == "finrag" else 50_000

    federations = {}
    for clients in (8, 16):
        for track in ("seen_query", "unseen_family"):
            federation = build_federation(
                federation_inventory,
                num_clients=clients,
                seed=args.seed + clients + offset,
            )
            federations[(clients, track)] = (
                federation,
                ClientPageView(index, federation.doc_to_client, clients),
            )

    all_rows = np.arange(len(index.page_uids), dtype=np.int64)
    results = []
    started = time.time()
    for dense_weight in args.dense_weights:
        for weight in args.weights:
            index.blend_weight = float(weight)
            index.dense_weight = float(dense_weight)

            per_federation = {}
            for (clients, track), (federation, view) in federations.items():
                hits = 0
                total = 0
                for record in sample:
                    gold_clients = federation.gold_clients(record)
                    picks = []
                    for client in gold_clients:
                        rows = view.rows_by_client.get(client)
                        if rows is None or rows.size == 0:
                            continue
                        scores = index.scores_for_rows(record.uid, rows)
                        best = int(rows[int(np.argmax(scores))])
                        picks.append((float(scores.max()), best))
                    picks.sort(key=lambda item: -item[0])
                    delivered = [index.page_uids[row] for _, row in picks[: args.budget]]
                    if unit == "page":
                        gold = set(record.gold_pages)
                        got = set(delivered)
                    else:
                        gold = set(record.doc_keys)
                        got = {uid.rsplit("#p", 1)[0] for uid in delivered}
                    total += 1
                    hits += int(bool(gold & got))
                per_federation[f"{clients}c_{track}"] = hits / max(total, 1)

            # whole-corpus recall@5, kept only as the tie-break
            recalls = []
            for record in sample:
                scores = index.scores_for_rows(record.uid, all_rows)
                take = min(10, scores.size)
                order = np.argpartition(-scores, take - 1)[:take]
                order = order[np.argsort(-scores[order])]
                ranked = [index.page_uids[int(all_rows[i])] for i in order]
                if unit == "document":
                    ranked = list(
                        dict.fromkeys(uid.rsplit("#p", 1)[0] for uid in ranked)
                    )
                    gold = set(record.doc_keys)
                else:
                    gold = set(record.gold_pages)
                recalls.append(recall_at_k(ranked, gold, 5))

            row = {
                "blend_weight": float(weight),
                "dense_weight": float(dense_weight),
                "delivered_hit_by_federation": per_federation,
                "delivered_hit_mean": float(np.mean(list(per_federation.values()))),
                "whole_corpus_recall@5": float(np.mean(recalls)),
                "records": len(sample),
            }
            results.append(row)
            print(
                json.dumps(
                    {
                        "visual_w": weight,
                        "dense_w": dense_weight,
                        "delivered_hit_mean": round(row["delivered_hit_mean"], 4),
                        "recall@5": round(row["whole_corpus_recall@5"], 4),
                    }
                ),
                flush=True,
            )

    best = max(
        results,
        key=lambda r: (r["delivered_hit_mean"], r["whole_corpus_recall@5"]),
    )
    payload = {
        "schema_version": "public-e2e-retriever-selection-delivered-v1",
        "dataset": args.dataset,
        "created_unix": time.time(),
        "elapsed_seconds": time.time() - started,
        "selection_rule": (
            "maximise the mean delivered-evidence hit rate over the four "
            "(client scale, track) federations, where delivery follows the "
            "protocol exactly: the client that owns a relevant document "
            "returns its single best-scoring page and the page budget "
            "truncates the list; whole-corpus recall@5 is the tie-break"
        ),
        "why_this_rule": (
            "the earlier whole-corpus recall@5 rule is not the quantity the "
            "protocol delivers, and the two disagreed on VisDoM"
        ),
        "split_scope": (
            "records in the training portion of at least one setting and in no test portion"
        ),
        "usable_training_records": len(usable),
        "sampled_records": len(sample),
        "sampling": (
            "round-robin over (language, category) strata, each stratum "
            "ordered by a seed-stable hash of the record id"
        ),
        "sampled_categories": sorted({r.category for r in sample}),
        "evidence_budget": args.budget,
        "candidates": results,
        "selected_blend_weight": best["blend_weight"],
        "selected_dense_weight": best["dense_weight"],
        "selected": best,
    }
    out = write_json(
        ROOT
        / "artifacts"
        / TRACK
        / f"retriever_selection_{args.dataset}{args.output_tag}.json",
        payload,
    )
    print(json.dumps({"output": str(out), "selected": best}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
