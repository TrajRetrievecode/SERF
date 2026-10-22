#!/usr/bin/env python3
"""Run the public end-to-end experiment for one dataset.

Three stages, each resumable on its own:

  plan      client selection -> page retrieval -> evidence packaging.
            Writes selections.jsonl and the deduplicated generation queue.
            CPU only.
  generate  one shared frozen generator answers every unique evidence
            package exactly once.  GPU, cooperative, cached by prompt hash.
  score     joins the plan with the generator cache and writes
            predictions.jsonl plus failures.jsonl.

Nothing here reads a gold label before the scoring stage.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

from e2e_common import (
    JsonlAppender,
    ROOT,
    TRACK,
    exact_match,
    gpu_snapshot,
    numeric_accuracy,
    numeric_recall,
    prf,
    read_json,
    record_pid,
    recall_at_k,
    token_f1,
    write_json,
)
from e2e_corpus import page_text_map
from e2e_generation import (
    GenerationCache,
    VllmGenerator,
    parse_output,
    request_for,
)
from e2e_records import load_records
from e2e_retrieval import (
    CENTRAL_STORE,
    ClientPageView,
    PageIndex,
    build_package,
    communication_cost,
    oracle_package,
)
from e2e_pagefed import (
    as_page_records,
    build_page_federation,
    complementary_splits,
    cross_client_flags,
)
from e2e_routing import ALL_METHODS, METHOD_LEVEL, METHOD_NOTES, RoutingLayer

ARTIFACTS = ROOT / "artifacts" / TRACK


def config_key(clients: int, track: str) -> str:
    return f"c{clients}_{track}"


def version_suffix(prompt_version: str) -> str:
    """v1 keeps the original filenames so the first run stays addressable."""
    return "" if prompt_version == "v1" else f"_{prompt_version}"


def cache_name(prompt_version: str) -> str:
    return f"generation_cache{version_suffix(prompt_version)}.jsonl"


def predictions_name(prompt_version: str) -> str:
    return f"predictions{version_suffix(prompt_version)}.jsonl"


def failures_name(prompt_version: str) -> str:
    return f"failures{version_suffix(prompt_version)}.jsonl"


# --------------------------------------------------------------------------
# stage: plan
# --------------------------------------------------------------------------
def stage_plan(dataset: str, protocol: dict, out_dir: Path, limit_test: int = 0) -> dict:
    records, _, inventory = load_records(dataset)
    by_uid = {r.uid: r for r in records}
    index = PageIndex.load(
        dataset, protocol["retriever"]["blend_weight"][dataset]
    )
    dense_weight = float(
        (protocol["retriever"].get("dense_weight") or {}).get(dataset, 0.0)
    )
    if dense_weight > 0.0:
        index.attach_dense(dataset, dense_weight)
    texts = page_text_map(dataset)
    text_limit = int(protocol["evidence"]["page_text_char_limit"])
    rank_depth = int(protocol["evidence"]["ranked_candidate_depth"])
    seed = int(protocol["seed"])

    # The federation owns every document that exists in the page corpus, so a
    # called client can also hold distractor documents.  For VisDoM the CSV
    # candidate pools are unioned in as well, which is a no-op when every
    # referenced PDF is present.
    federation_inventory = sorted(set(index.doc_key_list) | set(inventory))

    # The complementary track co-holds each document's pages on a few
    # clients, so a query whose qrels name several pages can genuinely need
    # several clients.  Records are restated in page units for it.
    assignment_unit = protocol["federation"].get("assignment_unit", "family")
    page_atomic = assignment_unit == "page"
    replicas = int(protocol["federation"].get("page_replicas", 3))
    if page_atomic:
        records = as_page_records(records)
        by_uid = {r.uid: r for r in records}
        page_universe = index.page_uids

    plan_path = out_dir / "plan.jsonl"
    selections_path = out_dir / "selections.jsonl"
    configurations: dict[str, dict] = {}

    done_keys: set[str] = set()
    if plan_path.exists():
        with plan_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done_keys.add(row["row_key"])

    plan_writer = JsonlAppender(plan_path)
    selection_writer = JsonlAppender(selections_path)
    started = time.time()
    written = 0

    for clients in protocol["client_scales"]:
        for track in protocol["tracks"]:
            key = config_key(clients, track)
            layer_seed = seed + clients + (0 if dataset == "finrag" else 50_000)
            if page_atomic:
                federation = build_page_federation(
                    page_universe, clients, layer_seed, replicas
                )
                layer = RoutingLayer(
                    records=records,
                    inventory_documents=federation_inventory,
                    num_clients=clients,
                    track=track,
                    seed=layer_seed,
                    federation=federation,
                    splits=complementary_splits(records, layer_seed + 101),
                )
            else:
                layer = RoutingLayer(
                    records=records,
                    inventory_documents=federation_inventory,
                    num_clients=clients,
                    track=track,
                    seed=layer_seed,
                )
            if limit_test:
                # Deterministic pilot slice, stratified round-robin over
                # (language, category) so the pilot actually exercises both
                # languages and every question type instead of whichever
                # stratum happens to sort first.
                strata: dict[tuple, list[int]] = {}
                for position, record in enumerate(layer.test):
                    stratum = (by_uid[record.uid].lang, by_uid[record.uid].category)
                    strata.setdefault(stratum, []).append(position)
                order = sorted(strata)
                picked: list[int] = []
                depth = 0
                while len(picked) < limit_test and any(
                    len(strata[s]) > depth for s in order
                ):
                    for stratum in order:
                        if len(picked) >= limit_test:
                            break
                        if len(strata[stratum]) > depth:
                            picked.append(strata[stratum][depth])
                    depth += 1
                picked.sort()
                layer.test = [layer.test[i] for i in picked]
                index_array = np.asarray(picked, dtype=np.int64)
                for method, matrix in layer.probabilities.items():
                    layer.probabilities[method] = matrix[index_array]
            configurations[key] = layer.configuration()
            view = ClientPageView(
                index,
                layer.federation.doc_to_client,
                clients,
                unit="page" if page_atomic else "document",
            )
            cross_client = (
                cross_client_flags(layer.test, layer.federation)
                if page_atomic
                else {}
            )
            configurations[key]["cross_client_test_records"] = sum(
                1 for v in cross_client.values() if v
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
                            continue  # missingness proxy runs at the main budget
                        selections = layer.select(method, budget, scenario, seed)
                        for selection in selections:
                            record = by_uid[selection.uid]
                            row_key = (
                                f"{key}|{method}|b{budget}|{scenario}|{selection.uid}"
                            )
                            if row_key in done_keys:
                                continue
                            clients_tuple = selection.clients
                            abstain_routing = selection.abstain
                            reason = selection.reason

                            if method == "Oracle-client":
                                gold_clients = layer.federation.gold_clients(record)
                                online = layer.online_mask(record, scenario, seed)
                                if not any(online[g] for g in gold_clients):
                                    clients_tuple = ()
                                    abstain_routing = True
                                    reason = "oracle_gold_client_offline"
                                else:
                                    clients_tuple = tuple(
                                        g for g in gold_clients if online[g]
                                    )
                                    abstain_routing = False
                                    reason = "oracle_gold_client"

                            t0 = time.perf_counter()
                            if method == "Oracle-evidence":
                                package = oracle_package(
                                    index,
                                    selection.uid,
                                    record.gold_pages,
                                    record.doc_keys,
                                    budget,
                                )
                            else:
                                # The package is always built from the clients
                                # that were actually called, so an abstaining
                                # system still pays for the calls it made.
                                package = build_package(
                                    index,
                                    view,
                                    selection.uid,
                                    clients_tuple,
                                    budget,
                                    protocol["evidence"]["pages_per_client"],
                                    rank_depth,
                                )
                            retrieval_ms = (time.perf_counter() - t0) * 1000.0

                            evidence_uids = (
                                () if abstain_routing else tuple(package.evidence_page_uids)
                            )
                            request = request_for(
                                record.query, evidence_uids, texts, text_limit
                            )
                            cost = communication_cost(
                                record.query,
                                budget,
                                package,
                                evidence_uids,
                                texts,
                                text_limit,
                            )

                            plan_writer.write(
                                {
                                    "row_key": row_key,
                                    "dataset": dataset,
                                    "config": key,
                                    "clients": clients,
                                    "track": track,
                                    "method": method,
                                    "implementation_level": METHOD_LEVEL[method],
                                    "budget": budget,
                                    "scenario": scenario,
                                    "uid": selection.uid,
                                    "selected_clients": list(clients_tuple),
                                    "abstain_routing": abstain_routing,
                                    "routing_reason": reason,
                                    "offline_probability_peak": selection.offline_probability_peak,
                                    "candidate_pages": package.candidate_pages,
                                    "ranked_page_uids": package.ranked_page_uids[:rank_depth],
                                    "evidence_page_uids": list(evidence_uids),
                                    "generation_key": (
                                        None if abstain_routing else request.key
                                    ),
                                    "evidence_labels": list(request.evidence_labels),
                                    "cost": cost,
                                    "retrieval_ms": retrieval_ms,
                                    "cross_client_gold": bool(
                                        cross_client.get(selection.uid, False)
                                    ),
                                }
                            )
                            selection_writer.write(
                                {
                                    "row_key": row_key,
                                    "uid": selection.uid,
                                    "method": method,
                                    "budget": budget,
                                    "scenario": scenario,
                                    "config": key,
                                    "selected_clients": list(clients_tuple),
                                    "abstain_routing": abstain_routing,
                                    "routing_reason": reason,
                                    "evidence_page_uids": list(evidence_uids),
                                }
                            )
                            written += 1
                            if written % 5000 == 0:
                                print(
                                    json.dumps(
                                        {
                                            "stage": "plan",
                                            "written": written,
                                            "config": key,
                                            "method": method,
                                            "elapsed_seconds": round(
                                                time.time() - started, 1
                                            ),
                                        }
                                    ),
                                    flush=True,
                                )
    plan_writer.close()
    selection_writer.close()
    write_json(out_dir / "routing_configurations.json", configurations)
    summary = {
        "stage": "plan",
        "dataset": dataset,
        "rows_written_now": written,
        "elapsed_seconds": time.time() - started,
        "plan_path": str(plan_path),
    }
    print(json.dumps(summary), flush=True)
    return summary


# --------------------------------------------------------------------------
# stage: generate
# --------------------------------------------------------------------------
def stage_generate(dataset: str, protocol: dict, out_dir: Path, args) -> dict:
    records, _, _ = load_records(dataset)
    by_uid = {r.uid: r for r in records}
    texts = page_text_map(dataset)
    text_limit = int(protocol["evidence"]["page_text_char_limit"])

    version = args.prompt_version
    queue: "OrderedDict[str, str]" = OrderedDict()
    with (out_dir / "plan.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("generation_key") is None:
                continue  # the system abstained before any generation
            record = by_uid[row["uid"]]
            # The key is recomputed rather than read from the plan so a prompt
            # version change reuses the plan and only re-generates.
            request = request_for(
                record.query,
                tuple(row["evidence_page_uids"]),
                texts,
                text_limit,
                version,
            )
            if request.key not in queue:
                queue[request.key] = request.prompt

    cache = GenerationCache(out_dir / cache_name(version))
    pending = [(k, p) for k, p in queue.items() if k not in cache]
    print(
        json.dumps(
            {
                "stage": "generate",
                "unique_requests": len(queue),
                "already_cached": len(queue) - len(pending),
                "pending": len(pending),
            }
        ),
        flush=True,
    )
    if not pending:
        cache.close()
        return {"stage": "generate", "pending": 0, "generated": 0}

    generator = VllmGenerator(
        model_path=os.environ.get("SERF_GENERATOR", protocol["generator"]["model_path"]),
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=int(protocol["generator"]["max_model_len"]),
        seed=int(protocol["generator"]["seed"]),
        max_tokens=int(protocol["generator"]["max_tokens"]),
        enforce_eager=not args.no_enforce_eager,
    )
    write_json(
        out_dir / f"generator_configuration{version_suffix(version)}.json",
        {
            **generator.configuration(version),
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "gpu_snapshot_at_start": gpu_snapshot(),
            "shared_gpu_policy": (
                "vLLM is started inside the memory left free by the incumbent "
                "workloads. No other process was killed, suspended, reniced or "
                "connected to."
            ),
        },
    )

    started = time.time()
    generated = 0
    chunk = args.generation_chunk
    for offset in range(0, len(pending), chunk):
        batch = pending[offset : offset + chunk]
        outputs = generator.generate([p for _, p in batch], version)
        for (key, _), output in zip(batch, outputs):
            cache.put(key, output)
        generated += len(batch)
        rate = generated / max(time.time() - started, 1e-6)
        print(
            json.dumps(
                {
                    "stage": "generate",
                    "done": generated,
                    "pending": len(pending),
                    "per_second": round(rate, 2),
                    "eta_seconds": round((len(pending) - generated) / max(rate, 1e-6)),
                    "elapsed_seconds": round(time.time() - started, 1),
                }
            ),
            flush=True,
        )
    cache.close()
    return {
        "stage": "generate",
        "pending": len(pending),
        "generated": generated,
        "elapsed_seconds": time.time() - started,
        "seconds_per_generation": (time.time() - started) / max(generated, 1),
    }


# --------------------------------------------------------------------------
# stage: score
# --------------------------------------------------------------------------
def gold_units(record, unit: str) -> tuple[str, ...]:
    return record.gold_pages if unit == "page" else record.doc_keys


def to_units(page_uids, unit: str) -> list[str]:
    if unit == "page":
        return list(page_uids)
    return [uid.rsplit("#p", 1)[0] for uid in page_uids]


def stage_score(
    dataset: str, protocol: dict, out_dir: Path, prompt_version: str = "v1"
) -> dict:
    records, _, _ = load_records(dataset)
    by_uid = {r.uid: r for r in records}
    unit = records[0].gold_unit
    texts = page_text_map(dataset)
    text_limit = int(protocol["evidence"]["page_text_char_limit"])
    cache = GenerationCache(out_dir / cache_name(prompt_version))
    predictions = out_dir / predictions_name(prompt_version)
    failures = out_dir / failures_name(prompt_version)
    predictions.unlink(missing_ok=True)
    failures.unlink(missing_ok=True)

    answer_threshold = float(protocol["scoring"]["answer_correct_token_f1"])
    writer = JsonlAppender(predictions)
    failure_writer = JsonlAppender(failures)
    started = time.time()
    total = 0
    missing_generation = 0

    with (out_dir / "plan.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            record = by_uid[row["uid"]]
            gold = gold_units(record, unit)
            labels = tuple(row["evidence_labels"])
            abstain_routing = bool(row["abstain_routing"])

            parsed = {
                "answer": "",
                "citations": [],
                "citations_dropped": [],
                "parse_status": "abstained_before_generation",
                "generator_abstained": False,
            }
            if not abstain_routing:
                key = (
                    request_for(
                        record.query,
                        tuple(row["evidence_page_uids"]),
                        texts,
                        text_limit,
                        prompt_version,
                    ).key
                    if row.get("generation_key") is not None
                    else None
                )
                output = cache.get(key) if key else None
                if output is None:
                    missing_generation += 1
                    failure_writer.write(
                        {
                            "row_key": row["row_key"],
                            "reason": "generation_missing_from_cache",
                            "generation_key": key,
                        }
                    )
                    continue
                parsed = parse_output(output, labels)

            abstain_generator = bool(parsed["generator_abstained"])
            abstain_system = abstain_routing or abstain_generator
            answer = "" if abstain_system else parsed["answer"]

            evidence_units = to_units(row["evidence_page_uids"], unit)
            ranked_units = to_units(row["ranked_page_uids"], unit)
            ranked_dedup = list(dict.fromkeys(ranked_units))
            cited_uids = [
                row["evidence_page_uids"][labels.index(c)]
                for c in parsed["citations"]
                if c in labels and labels.index(c) < len(row["evidence_page_uids"])
            ]
            cited_units = to_units(cited_uids, unit)

            evidence = prf(evidence_units, gold)
            citation = prf(cited_units, gold)
            em = 0.0 if abstain_system else exact_match(answer, record.answer)
            f1 = 0.0 if abstain_system else token_f1(answer, record.answer)
            num_recall = None if abstain_system else numeric_recall(answer, record.answer)
            num_acc = None if abstain_system else numeric_accuracy(answer, record.answer)

            evidence_hit = bool(set(evidence_units) & set(gold))
            evidence_correct = evidence_hit and not abstain_system
            answer_correct = (f1 >= answer_threshold) and not abstain_system

            writer.write(
                {
                    "row_key": row["row_key"],
                    "dataset": dataset,
                    "config": row["config"],
                    "clients": row["clients"],
                    "track": row["track"],
                    "method": row["method"],
                    "implementation_level": row["implementation_level"],
                    "budget": row["budget"],
                    "scenario": row["scenario"],
                    "uid": row["uid"],
                    "lang": record.lang,
                    "category": record.category,
                    "gold_unit": unit,
                    "cross_client_gold": bool(row.get("cross_client_gold", False)),
                    "gold_count": len(gold),
                    "abstain_routing": abstain_routing,
                    "abstain_generator": abstain_generator,
                    "abstain_system": abstain_system,
                    "routing_reason": row["routing_reason"],
                    "parse_status": parsed["parse_status"],
                    "client_calls": row["cost"]["client_calls"],
                    "selected_pages": row["cost"]["selected_pages"],
                    "candidate_pages": row["candidate_pages"],
                    "retrieval_recall@1": recall_at_k(ranked_dedup, gold, 1),
                    "retrieval_recall@3": recall_at_k(ranked_dedup, gold, 3),
                    "retrieval_recall@5": recall_at_k(ranked_dedup, gold, 5),
                    "evidence_recall": evidence["recall"],
                    "evidence_precision": evidence["precision"],
                    "evidence_f1": evidence["f1"],
                    "citation_recall": citation["recall"],
                    "citation_precision": citation["precision"],
                    "citation_f1": citation["f1"],
                    "answer_exact_match": em,
                    "answer_token_f1": f1,
                    "numeric_recall": num_recall,
                    "numeric_accuracy": num_acc,
                    "evidence_correct": evidence_correct,
                    "answer_correct": answer_correct,
                    "serialized_request_bytes": row["cost"]["serialized_request_bytes"],
                    "serialized_response_bytes": row["cost"]["serialized_response_bytes"],
                    "protocol_bytes": row["cost"]["protocol_bytes"],
                    "retrieval_ms": row["retrieval_ms"],
                    "prediction": answer[:2000],
                }
            )
            total += 1

    writer.close()
    failure_writer.close()
    summary = {
        "stage": "score",
        "dataset": dataset,
        "rows_scored": total,
        "rows_missing_generation": missing_generation,
        "elapsed_seconds": time.time() - started,
    }
    print(json.dumps(summary), flush=True)
    return summary


# --------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["finrag", "visdom"])
    parser.add_argument(
        "--stages", nargs="+", default=["plan", "generate", "score"]
    )
    parser.add_argument("--protocol", type=Path, default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.11)
    parser.add_argument("--generation-chunk", type=int, default=512)
    parser.add_argument("--no-enforce-eager", action="store_true")
    parser.add_argument("--limit-test", type=int, default=0)
    parser.add_argument("--out-suffix", default="")
    parser.add_argument(
        "--prompt-version",
        default="v1",
        help=(
            "shared generator prompt version; the plan is reused across "
            "versions and only generation and scoring re-run"
        ),
    )
    args = parser.parse_args()

    protocol_path = args.protocol or (
        ROOT / "configs" / TRACK / "protocol_v3.json"
    )
    protocol = read_json(protocol_path)
    out_dir = ARTIFACTS / (args.dataset + args.out_suffix)
    out_dir.mkdir(parents=True, exist_ok=True)
    record_pid(f"run_e2e_{args.dataset}{args.out_suffix}", {"stages": args.stages})

    summaries = []
    for stage in args.stages:
        if stage == "plan":
            summaries.append(
                stage_plan(args.dataset, protocol, out_dir, args.limit_test)
            )
        elif stage == "generate":
            summaries.append(stage_generate(args.dataset, protocol, out_dir, args))
        elif stage == "score":
            summaries.append(
                stage_score(args.dataset, protocol, out_dir, args.prompt_version)
            )
        else:
            raise ValueError(f"unknown stage: {stage}")
    write_json(
        out_dir
        / (
            f"run_summary_{'_'.join(args.stages)}"
            f"{version_suffix(args.prompt_version)}.json"
        ),
        {
            "dataset": args.dataset,
            "prompt_version": args.prompt_version,
            "summaries": summaries,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
