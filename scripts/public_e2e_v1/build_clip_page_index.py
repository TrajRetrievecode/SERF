#!/usr/bin/env python3
"""Shared frozen dual-channel page index: CLIP ViT-B/32 visual + TF-IDF text.

This is the retriever every method in the public end-to-end track reads, so
retrieval quality is identical across methods by construction.  Both channels
are query-independent and are built without touching qrels or answers.

The job renders pages in worker processes and encodes them on one GPU inside
the memory the incumbent workloads leave free.  It never signals, kills or
reniced another process.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from e2e_common import ROOT, TRACK, gpu_snapshot, record_pid, wait_for_gpu, write_json
from e2e_corpus import CACHE, page_text_map, pages_for
from e2e_records import load_records

CLIP_DEFAULT = Path(os.environ.get("SERF_CLIP", str(Path(os.environ.get("SERF_MODELS", "models")) / "openai/clip-vit-base-patch32")))


def _render_chunk(payload: tuple) -> tuple[int, list[bytes], list[tuple[int, int]]]:
    """Render a contiguous chunk of pages to raw RGB bytes in a worker."""
    import fitz
    from PIL import Image

    start, rows, max_pixels, side = payload
    cache: dict = {}
    buffers: list[bytes] = []
    sizes: list[tuple[int, int]] = []
    for row in rows:
        path = row["source_path"]
        document = cache.get(path)
        if document is None:
            if len(cache) > 6:
                _, stale = cache.popitem()
                stale.close()
            document = fitz.open(path)
            cache[path] = document
        page = document[int(row["page_number"]) - 1]
        rect = page.rect
        base = max(float(rect.width) * float(rect.height), 1.0)
        zoom = min(max((max_pixels / base) ** 0.5, 0.4), 4.0)
        pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        image = Image.frombytes(
            "RGB", (pixmap.width, pixmap.height), pixmap.samples
        ).resize((side, side), Image.BICUBIC)
        buffers.append(image.tobytes())
        sizes.append((side, side))
    for document in cache.values():
        document.close()
    return start, buffers, sizes


def build_visual(
    dataset: str,
    rows: list[dict],
    model_path: Path,
    out_dir: Path,
    batch_size: int,
    render_workers: int,
    max_pixels: int,
    log_every: int,
) -> np.ndarray:
    import torch
    from transformers import CLIPModel, CLIPProcessor

    processor = CLIPProcessor.from_pretrained(str(model_path))
    model = CLIPModel.from_pretrained(str(model_path), dtype=torch.float16)
    model.eval().cuda()
    side = int(processor.image_processor.crop_size["height"])
    mean = np.array(processor.image_processor.image_mean, dtype=np.float32)
    std = np.array(processor.image_processor.image_std, dtype=np.float32)

    shard_dir = out_dir / "checkpoints"
    shard_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[int, np.ndarray] = {}
    todo: list[tuple] = []
    for start in range(0, len(rows), batch_size):
        shard = shard_dir / f"visual_{start:08d}_{batch_size:05d}.npy"
        if shard.is_file():
            try:
                array = np.load(shard)
                if array.ndim == 2 and array.shape[0] == len(
                    rows[start : start + batch_size]
                ):
                    outputs[start] = array.astype(np.float32, copy=False)
                    continue
            except (OSError, ValueError):
                shard.unlink(missing_ok=True)
        todo.append((start, rows[start : start + batch_size], max_pixels, side))

    print(
        json.dumps(
            {
                "stage": "visual",
                "total_pages": len(rows),
                "reused_shards": len(outputs),
                "pending_batches": len(todo),
            }
        ),
        flush=True,
    )

    if todo:
        result_queue: "queue.Queue" = queue.Queue(maxsize=8)
        error_box: list[Exception] = []

        def producer() -> None:
            try:
                with ProcessPoolExecutor(max_workers=render_workers) as pool:
                    for item in pool.map(_render_chunk, todo, chunksize=1):
                        result_queue.put(item)
            except Exception as error:  # pragma: no cover
                error_box.append(error)
            finally:
                result_queue.put(None)

        thread = threading.Thread(target=producer, daemon=True)
        thread.start()
        started = time.time()
        done = 0
        while True:
            item = result_queue.get()
            if item is None:
                break
            start, buffers, sizes = item
            batch = np.stack(
                [
                    np.frombuffer(buffer, dtype=np.uint8).reshape(h, w, 3)
                    for buffer, (w, h) in zip(buffers, sizes)
                ]
            ).astype(np.float32) / 255.0
            batch = (batch - mean) / std
            tensor = torch.from_numpy(batch.transpose(0, 3, 1, 2)).half().cuda()
            with torch.no_grad():
                features = model.get_image_features(pixel_values=tensor)
                features = features / features.norm(dim=-1, keepdim=True)
            array = features.float().cpu().numpy().astype(np.float32)
            np.save(shard_dir / f"visual_{start:08d}_{batch_size:05d}.npy", array)
            outputs[start] = array
            done += array.shape[0]
            if done % log_every < batch_size:
                rate = done / max(time.time() - started, 1e-6)
                print(
                    json.dumps(
                        {
                            "stage": "visual",
                            "done": done,
                            "pages_per_second": round(rate, 1),
                            "elapsed_seconds": round(time.time() - started, 1),
                        }
                    ),
                    flush=True,
                )
        thread.join(timeout=60)
        if error_box:
            raise error_box[0]

    matrix = np.concatenate([outputs[k] for k in sorted(outputs)], axis=0)
    if matrix.shape[0] != len(rows):
        raise RuntimeError(f"visual rows {matrix.shape[0]} != pages {len(rows)}")
    return matrix.astype(np.float32, copy=False)


def build_visual_queries(model_path: Path, queries: list[str], batch_size: int):
    import torch
    from transformers import CLIPModel, CLIPProcessor

    processor = CLIPProcessor.from_pretrained(str(model_path))
    model = CLIPModel.from_pretrained(str(model_path), dtype=torch.float16)
    model.eval().cuda()
    output = []
    for offset in range(0, len(queries), batch_size):
        batch = queries[offset : offset + batch_size]
        inputs = processor(
            text=batch, return_tensors="pt", padding=True, truncation=True, max_length=77
        )
        inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            features = model.get_text_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        output.append(features.float().cpu().numpy().astype(np.float32))
    return np.concatenate(output, axis=0)


def build_text_channel(dataset: str, rows: list[dict], queries: list[str]):
    from scipy import sparse
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    from fedresolve_core import mixed_financial_tokens

    texts = page_text_map(dataset)
    corpus = [texts.get(row["page_uid"], "") for row in rows]
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
    page_matrix = vectorizer.fit_transform(corpus)
    query_matrix = vectorizer.transform(queries)
    return (
        normalize(page_matrix).astype(np.float32),
        normalize(query_matrix).astype(np.float32),
        {
            "vocabulary_size": len(vectorizer.vocabulary_),
            "nonempty_pages": int(sum(1 for t in corpus if t.strip())),
            "matrix_nnz": int(page_matrix.nnz),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["finrag", "visdom"])
    parser.add_argument("--clip", type=Path, default=CLIP_DEFAULT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--render-workers", type=int, default=6)
    parser.add_argument("--max-pixels", type=int, default=262144)
    parser.add_argument("--reserve-mib", type=int, default=2500)
    parser.add_argument("--headroom-mib", type=int, default=1200)
    parser.add_argument("--log-every", type=int, default=2000)
    args = parser.parse_args()

    record_pid(f"build_clip_page_index_{args.dataset}")
    pages = pages_for(args.dataset)
    rows = [
        {
            "page_uid": p.page_uid,
            "doc_key": p.doc_key,
            "page_number": p.page_number,
            "source_path": p.source_path,
        }
        for p in pages
    ]
    records, _, _ = load_records(args.dataset)
    query_uids = [r.uid for r in records]
    query_texts = [r.query for r in records]

    out_dir = CACHE / f"{args.dataset}_clip_pages"
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

    started = time.time()
    visual = build_visual(
        args.dataset,
        rows,
        args.clip,
        out_dir,
        args.batch_size,
        args.render_workers,
        args.max_pixels,
        args.log_every,
    )
    visual_queries = build_visual_queries(args.clip, query_texts, 256)
    page_text_matrix, query_text_matrix, text_stats = build_text_channel(
        args.dataset, rows, query_texts
    )

    from scipy import sparse

    np.save(out_dir / "page_visual.npy", visual)
    np.save(out_dir / "query_visual.npy", visual_queries)
    sparse.save_npz(out_dir / "page_text.npz", page_text_matrix)
    sparse.save_npz(out_dir / "query_text.npz", query_text_matrix)
    (out_dir / "pages.json").write_text(
        json.dumps(rows, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "queries.json").write_text(
        json.dumps([{"uid": u} for u in query_uids], ensure_ascii=False),
        encoding="utf-8",
    )

    manifest = {
        "schema_version": "public-e2e-clip-tfidf-page-index-v1",
        "dataset": args.dataset,
        "status": "complete",
        "created_unix": time.time(),
        "elapsed_seconds": time.time() - started,
        "visual_model_path": str(args.clip),
        "model_copied_into_project": False,
        "visual_dtype": "float16",
        "render": "PyMuPDF isotropic zoom to the pixel budget, then bicubic to the CLIP crop",
        "max_pixels": args.max_pixels,
        "counts": {"pages": len(rows), "queries": len(query_uids)},
        "visual_shape": list(visual.shape),
        "text_channel": text_stats,
        "text_tokenizer": "fedresolve_core.mixed_financial_tokens",
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
