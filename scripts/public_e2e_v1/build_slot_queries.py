#!/usr/bin/env python3
"""Encode claim-slot texts with the same three channels as the shared index.

The page side is untouched; only the query side gains slot vectors.  The
TF-IDF vectorizer is refitted on the identical page corpus with the identical
parameters, which is deterministic and therefore reproduces the vocabulary the
page matrix was built with.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from e2e_common import ROOT, TRACK, gpu_snapshot, record_pid, wait_for_gpu, write_json
from e2e_corpus import CACHE, page_text_map, pages_for
from e2e_records import load_records
from e2e_slots import slot_texts_for, statistics

CLIP_DEFAULT = os.environ.get("SERF_CLIP", str(Path(os.environ.get("SERF_MODELS", "models")) / "openai/clip-vit-base-patch32"))
DENSE_DEFAULT = os.environ.get("SERF_BGE", str(Path(os.environ.get("SERF_MODELS", "models")) / "BAAI/bge-m3"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["finrag", "visdom"])
    parser.add_argument("--clip", default=CLIP_DEFAULT)
    parser.add_argument("--dense", default=DENSE_DEFAULT)
    parser.add_argument("--reserve-mib", type=int, default=4000)
    parser.add_argument("--headroom-mib", type=int, default=1200)
    parser.add_argument("--dense-batch", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=1024)
    args = parser.parse_args()

    record_pid(f"build_slot_queries_{args.dataset}")
    records, _, _ = load_records(args.dataset)
    texts, spans = slot_texts_for(records)
    stats = statistics(records)
    print(json.dumps({"slot_texts": len(texts), **stats}, ensure_ascii=False), flush=True)

    out_dir = CACHE / f"{args.dataset}_slot_queries"
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    # --- sparse channel: refit the identical vectorizer on the page corpus --
    from scipy import sparse
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    from fedresolve_core import mixed_financial_tokens

    pages = pages_for(args.dataset)
    page_texts = page_text_map(args.dataset)
    corpus = [page_texts.get(p.page_uid, "") for p in pages]
    vectorizer = TfidfVectorizer(
        tokenizer=mixed_financial_tokens,
        lowercase=False,
        min_df=2,
        max_df=0.995,
        sublinear_tf=True,
        norm="l2",
        max_features=220000,
        token_pattern=None,
    )
    vectorizer.fit(corpus)
    slot_sparse = normalize(vectorizer.transform(texts)).astype(np.float32)
    sparse.save_npz(out_dir / "slot_text.npz", slot_sparse)
    print(json.dumps({"stage": "sparse", "seconds": round(time.time() - started, 1)}), flush=True)

    chosen = wait_for_gpu(args.reserve_mib, args.headroom_mib)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(chosen["chosen"]["index"])
    import torch
    from transformers import AutoModel, AutoTokenizer, CLIPModel, CLIPProcessor

    # --- visual-text channel -------------------------------------------------
    processor = CLIPProcessor.from_pretrained(args.clip)
    clip = CLIPModel.from_pretrained(args.clip, dtype=torch.float16).eval().cuda()
    visual = []
    for offset in range(0, len(texts), 256):
        batch = texts[offset : offset + 256]
        inputs = processor(
            text=batch, return_tensors="pt", padding=True, truncation=True, max_length=77
        )
        inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            features = clip.get_text_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        visual.append(features.float().cpu().numpy().astype(np.float32))
    np.save(out_dir / "slot_visual.npy", np.concatenate(visual, axis=0))
    del clip
    torch.cuda.empty_cache()

    # --- dense text channel --------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.dense)
    model = AutoModel.from_pretrained(args.dense, dtype=torch.float16).eval().cuda()
    dense = []
    for offset in range(0, len(texts), args.dense_batch):
        batch = texts[offset : offset + args.dense_batch]
        encoded = tokenizer(
            batch, padding=True, truncation=True,
            max_length=args.max_length, return_tensors="pt",
        )
        encoded = {k: v.cuda() for k, v in encoded.items()}
        with torch.no_grad():
            hidden = model(**encoded).last_hidden_state[:, 0]
            hidden = torch.nn.functional.normalize(hidden, p=2, dim=-1)
        dense.append(hidden.float().cpu().numpy().astype(np.float32))
    np.save(out_dir / "slot_dense.npy", np.concatenate(dense, axis=0))

    write_json(
        out_dir / "manifest.json",
        {
            "schema_version": "public-e2e-slot-query-index-v1",
            "dataset": args.dataset,
            "created_unix": time.time(),
            "elapsed_seconds": time.time() - started,
            "slot_texts": len(texts),
            "records": len(records),
            "spans": spans,
            "decomposition": stats,
            "channels": {
                "visual": args.clip,
                "text_sparse": "TF-IDF refitted on the identical page corpus",
                "text_dense": args.dense,
            },
            "gpu": chosen["chosen"],
            "gpu_snapshot_after": gpu_snapshot(),
        },
    )
    print(json.dumps({"output": str(out_dir), "seconds": round(time.time() - started, 1)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
