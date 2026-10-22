"""Shared attribution operations; adapters and selectors remain upstream code."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
from types import FunctionType
import hashlib
import importlib.util
import json
import numpy as np
from e2e_slot_adapters import ADAPTERS, softmax
import e2e_v3_adapter

METHODS = ("RAGRoute", "UniversalRAG", "S2G-RAG", "FedResolve")
VARIANTS = ("base", "+sel", "+stop", "+both")
JOB = "attribution"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def output_dir(out, root):
    out = Path(out).resolve()
    allowed = (Path(root) / "artifacts" / "controls" / "attribution").resolve()
    if out not in [allowed, allowed / "smoke"]:
        raise ValueError("Output must be the job or job smoke artifact directory")
    out.mkdir(parents=True, exist_ok=True)
    return out


def stopped_clients(method, full, slots, online, base_clients, protocol, trace):
    """Keep a prefix of the imported router order; never rerank it."""
    if method == "FedResolve":
        return list(base_clients), []
    temperature = protocol["fedresolve"]["global_temperature"]
    minimum = protocol["fedresolve"]["minimum_marginal_gain"]
    kept, steps = [], []
    for position, client in enumerate(base_clients):
        if method == "RAGRoute":
            score, slot = full, None
        elif method == "UniversalRAG":
            score, slot = slots.max(axis=0), None
        else:
            slot = trace["gap_steps"][position]["slot"]
            score = slots[slot]
        # Called clients retain probability mass. Only offline or empty
        # clients leave the denominator; there is no conditional renormalization.
        share = float(softmax(np.where(online, score, -np.inf), temperature)[client])
        call = position == 0 or share >= minimum
        steps.append({"client": int(client), "slot": slot, "share": share, "called": call})
        if not call:
            break
        kept.append(int(client))
    return kept, steps


def public_selection(views, clients, budget, protocol):
    """Bind the supplied route to upstream selection without global patches."""
    original = e2e_v3_adapter.fedresolve_v3_adapter
    base = {"clients": list(clients), "trace": {"critical_missing_slots": []}}
    select = FunctionType(original.__code__,
        {**original.__globals__, "fedresolve_adapter": lambda *a, **k: base},
        original.__name__, original.__defaults__)
    return select(views, None, budget, protocol)["rows"]


def public_variants(views, online, budget, protocol):
    for method in METHODS:
        base = ADAPTERS[method](views, online, budget, protocol)
        stopped, steps = stopped_clients(method, views.full_client, views.slot_client,
            online, base["clients"], protocol, base["trace"])
        for variant in VARIANTS + (("topB",) if method == "FedResolve" else ()):
            clients = stopped if variant in ("+stop", "+both") else base["clients"]
            if variant == "base" or (method == "FedResolve" and variant == "+stop"):
                result = dict(base)
            else:
                if variant in ("+sel", "+both"):
                    rows = public_selection(views, clients, budget, protocol)
                elif variant == "topB":
                    pool = [int(r) for c in clients for r in views.rows_by_client[c]]
                    rows = sorted(pool, key=lambda r: (-float(views.evidence_score[r]), r))[:budget]
                else:
                    rows = views.evidence_rows(clients, budget,
                        views.full_page if method == "RAGRoute" else None)
                result = {**base, "clients": list(clients), "rows": rows,
                    "abstain": bool(base["trace"].get("critical_missing_slots")) or not rows}
            result["trace"] = {**base["trace"], "variant": variant,
                "minimum_share_steps": steps if variant in ("+stop", "+both") else []}
            yield method, variant, result


def import_selector(path, expected_hash):
    if digest(path) != expected_hash:
        raise ValueError("Evaluation selector source hash differs from the package protocol")
    spec = importlib.util.spec_from_file_location("frozen_gold_v3", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.v3_select


def gold_variants(runner, select, inputs, views, online, budget, protocol):
    for method in METHODS:
        base = runner.ADAPTERS[method](inputs, views, online, budget, protocol)
        stopped, steps = stopped_clients(method, views["full_client"], views["slot_client"],
            online, base["queried_clients"], protocol, base["trace"])
        for variant in VARIANTS + (("topB",) if method == "FedResolve" else ()):
            clients = stopped if variant in ("+stop", "+both") else base["queried_clients"]
            if variant == "base" or (method == "FedResolve" and variant == "+stop"):
                output = dict(base)
            else:
                if variant in ("+sel", "+both"):
                    evidence = select(inputs, clients, budget, protocol, views["evidence_score"])
                elif variant == "topB":
                    rows = [r for r, c in enumerate(inputs["candidate_clients"]) if c in clients]
                    rows.sort(key=lambda r: (-float(views["evidence_score"][r]), r))
                    evidence = [inputs["candidate_ids"][r] for r in rows[:budget]]
                else:
                    score = inputs["full_scores"] if method == "RAGRoute" else views["evidence_score"]
                    evidence = runner.evidence_for_clients(inputs, clients, budget, score)
                output = {**base, "queried_clients": list(clients), "evidence_ids": evidence,
                    "client_calls": len(clients),
                    "abstain": bool(base["trace"].get("critical_missing_slots")) or not evidence}
            # Confidence is retained for item_metrics compatibility only.
            output["trace"] = {**base["trace"], "variant": variant,
                "minimum_share_steps": steps if variant in ("+stop", "+both") else []}
            yield method, variant, output


def validate_design(design, public, gold):
    if public["seed"] != design["seeds"]["public"] or gold["seed"] != design["seeds"]["evaluation"]:
        raise ValueError("Protocol seed mismatch")
    for key, value in design["constants"].items():
        if gold["fedresolve"][key] != value:
            raise ValueError("Evaluation constant mismatch: " + key)
    for key in ("blend_weight", "dense_weight"):
        if public["retriever"][key] != design["retriever"][key]:
            raise ValueError("Retriever mismatch: " + key)


def pin_modules(design, modules):
    for key, module in modules.items():
        if digest(module.__file__) != design["source_sha256"][key]:
            raise ValueError("Imported source hash mismatch: " + key)



def limit_rows(rows, limit):
    """Keep the first N query IDs in file order for each dataset and setting."""
    if limit is None:
        return rows
    if limit <= 0:
        raise ValueError("Limit must be positive")
    chosen = {}
    for row in rows:
        key = (row["dataset"], row["config"])
        uids = chosen.setdefault(key, [])
        if row["uid"] not in uids and len(uids) < limit:
            uids.append(row["uid"])
    return [row for row in rows if row["uid"] in chosen[row["dataset"], row["config"]]]
