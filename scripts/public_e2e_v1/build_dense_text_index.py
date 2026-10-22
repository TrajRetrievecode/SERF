#!/usr/bin/env python3
"""Add a dense multilingual text channel to the shared page index.

The first run's retriever was CLIP visual plus TF-IDF text, and the training
splits showed the visual channel carrying no usable signal on either corpus.
This job encodes the same page text with a dense multilingual encoder so the
text channel itself can be upgraded.  It stays query-independent and reads no
qrel or answer.
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

MODEL_DEFAULT = os.environ.get("SERF_BGE", str(Path(os.environ.get("SERF_MODELS", "models")) / "BAAI/bge-m3"))


def save_npy_atomic(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    with tmp.open("wb") as handle:
        np.save(handle, array)
    os.replace(tmp, path)


def encode(
    model,
    tokenizer,
    texts: list[str],
    batch_size: int,
    max_length: int,
    checkpoint_dir: Path,
    log_every: int,
    stage: str,
) -> np.ndarray:
    import torch

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[int, np.ndarray] = {}
    todo: list[int] = []
    for offset in range(0, len(texts), batch_size):
        shard = checkpoint_dir / f"{stage}_{offset:08d}_{batch_size:05d}.npy"
        if shard.is_file():
            try:
                array = np.load(shard)
                if array.ndim == 2 and array.shape[0] == len(
                    texts[offset : offset + batch_size]
                ):
                    outputs[offset] = array.astype(np.float32, copy=False)
                    continue
            except (OSError, ValueError):
                shard.unlink(missing_ok=True)
        todo.append(offset)

    print(
        json.dumps(
            {
                "stage": stage,
                "total": len(texts),
                "reused_shards": len(outputs),
                "pending_batches": len(todo),
            }
        ),
        flush=True,
    )

    started = time.time()
    done = 0
    for offset in todo:
        batch = texts[offset : offset + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {k: v.cuda() for k, v in encoded.items()}
        with torch.no_grad():
            hidden = model(**encoded).last_hidden_state
            # bge-m3 dense retrieval uses the CLS vector.
            vectors = hidden[:, 0]
            vectors = torch.nn.functional.normalize(vectors, p=2, dim=-1)
        array = vectors.float().cpu().numpy().astype(np.float32)
        save_npy_atomic(
            checkpoint_dir / f"{stage}_{offset:08d}_{batch_size:05d}.npy", array
        )
        outputs[offset] = array
        done += len(batch)
        if done % log_every < batch_size:
            rate = done / max(time.time() - started, 1e-6)
            print(
                json.dumps(
                    {
                        "stage": stage,
                        "done": done,
                        "pending": sum(
                            len(texts[o : o + batch_size]) for o in todo
                        ),
                        "per_second": round(rate, 1),
                        "elapsed_seconds": round(time.time() - started, 1),
                    }
                ),
                flush=True,
            )

    matrix = np.concatenate([outputs[k] for k in sorted(outputs)], axis=0)
    if matrix.shape[0] != len(texts):
        raise RuntimeError(f"{stage}: {matrix.shape[0]} rows for {len(texts)} inputs")
    return matrix.astype(np.float32, copy=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["finrag", "visdom"])
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--reserve-mib", type=int, default=4000)
    parser.add_argument("--headroom-mib", type=int, default=1200)
    parser.add_argument("--log-every", type=int, default=2000)
    parser.add_argument("--page-text-char-limit", type=int, default=4000)
    args = parser.parse_args()

    record_pid(f"build_dense_text_index_{args.dataset}")
    pages = pages_for(args.dataset)
    texts_map = page_text_map(args.dataset)
    page_texts = [
        (texts_map.get(p.page_uid) or "").strip()[: args.page_text_char_limit]
        or "(empty page)"
        for p in pages
    ]
    records, _, _ = load_records(args.dataset)
    query_texts = [r.query for r in records]

    out_dir = CACHE / f"{args.dataset}_dense_pages"
    out_dir.mkdir(parents=True, exist_ok=True)

    chosen = wait_for_gpu(args.reserve_mib, args.headroom_mib)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(chosen["chosen"]["index"])
    print(
        json.dumps(
            {
                "stage": "gpu",
                "chosen_gpu": chosen["chosen"],
                "policy": (
                    "least utilised GPU with reservation + headroom free; no "
                    "other process is signalled, killed or reniced"
                ),
            }
        ),
        flush=True,
    )

    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, dtype=torch.float16)
    model.eval().cuda()

    started = time.time()
    page_matrix = encode(
        model,
        tokenizer,
        page_texts,
        args.batch_size,
        args.max_length,
        out_dir / "checkpoints",
        args.log_every,
        "pages",
    )
    query_matrix = encode(
        model,
        tokenizer,
        query_texts,
        max(args.batch_size, 32),
        args.max_length,
        out_dir / "checkpoints",
        args.log_every,
        "queries",
    )

    save_npy_atomic(out_dir / "page_dense.npy", page_matrix)
    save_npy_atomic(out_dir / "query_dense.npy", query_matrix)
    (out_dir / "pages.json").write_text(
        json.dumps([{"page_uid": p.page_uid} for p in pages], ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "queries.json").write_text(
        json.dumps([{"uid": r.uid} for r in records], ensure_ascii=False),
        encoding="utf-8",
    )

    manifest = {
        "schema_version": "public-e2e-dense-text-index-v1",
        "dataset": args.dataset,
        "status": "complete",
        "created_unix": time.time(),
        "elapsed_seconds": time.time() - started,
        "model_path": args.model,
        "model_copied_into_project": False,
        "dtype": "float16",
        "pooling": "cls",
        "normalised": True,
        "max_length": args.max_length,
        "page_text_char_limit": args.page_text_char_limit,
        "counts": {"pages": len(pages), "queries": len(records)},
        "page_shape": list(page_matrix.shape),
        "query_shape": list(query_matrix.shape),
        "query_independent": True,
        "labels_used": "none; qrels and answers are not read by this job",
        "gpu": chosen["chosen"],
        "gpu_snapshot_after": gpu_snapshot(),
    }
    write_json(out_dir / "manifest.json", manifest)
    print(json.dumps({"output": str(out_dir), "elapsed_seconds": manifest["elapsed_seconds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
