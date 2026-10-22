"""Digests of the data that decide the confirmation (protocol_serf_v3_devconfirm.json, run_validity).

gold_digest: sha256 of the canonical JSON of {uid: sorted gold units} with the gold unit type, from
e2e_records.load_records and run_e2e.gold_units; it pins the relevance labels behind C1, C2, D1 to D5.
pagetext_digest: sha256 of the canonical JSON of e2e_corpus.page_text_map, which builds the prompts
of D4. Neither prints or writes any label or text.
"""
import hashlib
import json


def _sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def gold_digest(ds):
    from e2e_records import load_records
    from run_e2e import gold_units
    records, _, _ = load_records(ds)
    unit = records[0].gold_unit
    return {"records": len(records), "unit": unit,
            "sha256": _sha({"unit": unit, "gold": {r.uid: sorted(map(str, gold_units(r, unit))) for r in records}})}


def pagetext_digest(ds):
    from e2e_corpus import page_text_map
    texts = page_text_map(ds)
    return {"pages": len(texts), "sha256": _sha({str(k): v for k, v in texts.items()})}
