#!/usr/bin/env python3
"""Choose the shared generator prompt on training-split records only.

The prompt is shared by every method and chosen without reading a test record.

Three evidence conditions are built for the same sampled questions:

  gold      the official relevant pages, so the correct behaviour is to answer
  unrelated pages drawn deterministically from other documents, so the correct
            behaviour is to refuse
  empty     no page at all, so the correct behaviour is to refuse

Pre-registered criterion: maximise
    0.5 * (1 - abstain_rate_on_gold) + 0.5 * mean(abstain_rate_on_unrelated,
                                                  abstain_rate_on_empty)
with token F1 on the gold condition as the tie-break.  Both terms matter: a
prompt that never refuses would score 0.5 and lose to one that discriminates.
"""

from __future__ import annotations

import argparse
import os
import json
import time
from pathlib import Path

import numpy as np

from e2e_common import ROOT, TRACK, record_pid, token_f1, write_json
from e2e_corpus import page_text_map, pages_for
from e2e_generation import PROMPT_VERSIONS, VllmGenerator, parse_output, request_for
from e2e_records import load_records
from e2e_retrieval import PageIndex

from fedresolve_core import split_records, stable_u64


def training_only_uids(records, dataset: str, seed: int) -> list[str]:
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


def build_conditions(dataset: str, records, budget: int, seed: int) -> list[dict]:
    """gold / unrelated / empty evidence for each sampled record."""
    index = PageIndex.load(dataset, 0.5)
    by_doc: dict[str, list[str]] = {}
    for page in pages_for(dataset):
        by_doc.setdefault(page.doc_key, []).append(page.page_uid)
    all_docs = sorted(by_doc)

    conditions = []
    for record in records:
        gold = list(record.gold_pages)
        if not gold:
            for key in record.doc_keys:
                gold.extend(by_doc.get(key, [])[:budget])
        gold = gold[:budget]

        # Deterministic unrelated pages: walk the document list from a
        # record-derived offset and skip anything the record actually needs.
        forbidden = set(record.doc_keys)
        start = stable_u64(record.uid, seed) % max(len(all_docs), 1)
        unrelated: list[str] = []
        step = 0
        while len(unrelated) < budget and step < len(all_docs):
            doc = all_docs[(start + step) % len(all_docs)]
            step += 1
            if doc in forbidden:
                continue
            pages = by_doc.get(doc) or []
            if pages:
                unrelated.append(pages[len(pages) // 2])

        conditions.append(
            {
                "uid": record.uid,
                "query": record.query,
                "answer": record.answer,
                "gold": tuple(gold),
                "unrelated": tuple(unrelated),
                "empty": (),
            }
        )
    return conditions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["finrag", "visdom"])
    parser.add_argument("--versions", nargs="+", default=["v1", "v2", "v3"])
    parser.add_argument("--sample", type=int, default=150)
    parser.add_argument("--budget", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--text-char-limit", type=int, default=1800)
    parser.add_argument("--model", default=os.environ.get("SERF_GENERATOR", str(Path(os.environ.get("SERF_MODELS", "models")) / "Qwen/Qwen2.5-7B-Instruct")))
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.11)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--generator-seed", type=int, default=20260805)
    args = parser.parse_args()

    record_pid(f"select_prompt_{args.dataset}")
    records, _, _ = load_records(args.dataset)
    by_uid = {r.uid: r for r in records}
    usable = training_only_uids(records, args.dataset, args.seed)
    sample = [by_uid[u] for u in usable][: args.sample]
    texts = page_text_map(args.dataset)
    conditions = build_conditions(args.dataset, sample, args.budget, args.seed)

    generator = VllmGenerator(
        model_path=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        seed=args.generator_seed,
        max_tokens=args.max_tokens,
    )

    started = time.time()
    results = []
    for version in args.versions:
        row: dict = {"prompt_version": version, "conditions": {}}
        for condition in ("gold", "unrelated", "empty"):
            requests = [
                request_for(
                    item["query"],
                    item[condition],
                    texts,
                    args.text_char_limit,
                    version,
                )
                for item in conditions
            ]
            outputs = generator.generate([r.prompt for r in requests], version)
            parsed = [
                parse_output(output, request.evidence_labels)
                for request, output in zip(requests, outputs)
            ]
            abstain = float(
                np.mean([1.0 if p["generator_abstained"] else 0.0 for p in parsed])
            )
            answered = [
                (p, item)
                for p, item in zip(parsed, conditions)
                if not p["generator_abstained"]
            ]
            f1_all = float(
                np.mean(
                    [
                        0.0
                        if p["generator_abstained"]
                        else token_f1(p["answer"], item["answer"])
                        for p, item in zip(parsed, conditions)
                    ]
                )
            )
            f1_answered = (
                float(np.mean([token_f1(p["answer"], i["answer"]) for p, i in answered]))
                if answered
                else 0.0
            )
            row["conditions"][condition] = {
                "abstain_rate": abstain,
                "token_f1_all": f1_all,
                "token_f1_when_answered": f1_answered,
                "answered": len(answered),
                "records": len(conditions),
                "parse_failures": sum(
                    1 for p in parsed if p["parse_status"] not in ("json",)
                ),
            }
        gold = row["conditions"]["gold"]
        refuse = 0.5 * (
            row["conditions"]["unrelated"]["abstain_rate"]
            + row["conditions"]["empty"]["abstain_rate"]
        )
        row["answer_when_supported"] = 1.0 - gold["abstain_rate"]
        row["refuse_when_unsupported"] = refuse
        row["balanced_discrimination"] = 0.5 * (
            row["answer_when_supported"] + refuse
        )
        row["tie_break_token_f1_gold"] = gold["token_f1_all"]
        results.append(row)
        print(
            json.dumps(
                {
                    "version": version,
                    "answer_when_supported": round(row["answer_when_supported"], 4),
                    "refuse_when_unsupported": round(refuse, 4),
                    "balanced": round(row["balanced_discrimination"], 4),
                    "gold_token_f1": round(gold["token_f1_all"], 4),
                }
            ),
            flush=True,
        )

    best = max(
        results,
        key=lambda r: (r["balanced_discrimination"], r["tie_break_token_f1_gold"]),
    )
    payload = {
        "schema_version": "public-e2e-prompt-selection-v1",
        "dataset": args.dataset,
        "created_unix": time.time(),
        "elapsed_seconds": time.time() - started,
        "selection_rule": (
            "maximise 0.5 * P(answer | gold evidence) + 0.5 * mean(P(refuse | "
            "unrelated evidence), P(refuse | no evidence)); tie-break on token "
            "F1 under gold evidence"
        ),
        "split_scope": (
            "records in the training portion of at least one setting and in no test portion"
        ),
        "usable_training_records": len(usable),
        "sampled_records": len(sample),
        "evidence_budget": args.budget,
        "candidates": results,
        "selected_prompt_version": best["prompt_version"],
        "selected": best,
        "prompt_texts": {v: PROMPT_VERSIONS[v] for v in args.versions},
        "output_contract_unchanged": True,
        "parser_unchanged": True,
        "applies_identically_to_every_method": True,
    }
    out = write_json(
        ROOT / "artifacts" / TRACK / f"prompt_selection_{args.dataset}.json", payload
    )
    print(
        json.dumps(
            {"output": str(out), "selected": best["prompt_version"]}, ensure_ascii=False
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
