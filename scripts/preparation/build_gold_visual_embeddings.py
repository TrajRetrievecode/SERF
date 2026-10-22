#!/usr/bin/env python3
"""Build query-independent page and query embeddings for Gold v1.

The script reads only the frozen task inputs and candidate images.  It never
reads Gold answers, stances, sufficient client sets, or participation labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT_DEFAULT = Path(os.environ.get("SERF_ROOT", str(Path(__file__).resolve().parents[2])))
MODEL_DEFAULT = Path(os.environ.get("SERF_QWEN_EMBED", str(Path(os.environ.get("SERF_MODELS", str(ROOT_DEFAULT / "models"))) / "Qwen/Qwen3-VL-Embedding-2B")))
SUBQUESTION_RE = re.compile(r"(?m)^\s*\(([A-Z])\)\s*")


class SharedGpuBecameOccupied(RuntimeError):
    """Signal a cooperative stop when another compute process appears."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def save_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as handle:
        np.save(handle, value)
    os.replace(temporary, path)


def split_question(question: str) -> list[tuple[str, str]]:
    matches = list(SUBQUESTION_RE.finditer(question))
    if not matches:
        return [("Q", question.strip())]
    preamble = question[: matches[0].start()].strip()
    output: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(question)
        body = question[match.end() : end].strip()
        text = f"{preamble}\n({match.group(1)}) {body}".strip()
        output.append((match.group(1), text))
    return output


def gpu_observation(physical_gpu: int) -> dict[str, Any]:
    memory_text = subprocess.run(
        [
            "nvidia-smi",
            f"--id={physical_gpu}",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    process_text = subprocess.run(
        [
            "nvidia-smi",
            f"--id={physical_gpu}",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "physical_gpu": physical_gpu,
        "memory_used_mib": int(memory_text),
        "compute_pids": sorted(
            {
                int(value.strip())
                for value in process_text.splitlines()
                if value.strip().isdigit()
            }
        ),
    }


def preflight_gpu() -> dict[str, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible.isdigit():
        raise RuntimeError("CUDA_VISIBLE_DEVICES must name one physical GPU")
    observation = gpu_observation(int(visible))
    if observation["memory_used_mib"] > 32 or observation["compute_pids"]:
        raise RuntimeError(
            "shared GPU is occupied; no process was modified: "
            + json.dumps(observation)
        )
    return observation


def yield_if_other_process_appears(physical_gpu: int) -> None:
    observation = gpu_observation(physical_gpu)
    other_pids = sorted(set(observation["compute_pids"]) - {os.getpid()})
    if other_pids:
        raise SharedGpuBecameOccupied(
            f"physical_gpu={physical_gpu}, own_pid={os.getpid()}, "
            f"other_pids={other_pids}"
        )


def embed_batches(
    embedder,
    rows: list[dict[str, Any]],
    *,
    kind: str,
    batch_size: int,
    checkpoint_dir: Path,
    checkpoint_salt: str,
    physical_gpu: int,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    started = time.time()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset : offset + batch_size]
        end = offset + len(batch)
        cursor = offset
        pieces: list[np.ndarray] = []
        reused_rows = 0
        while cursor < end:
            found: tuple[int, np.ndarray] | None = None
            # Accept a chain of already validated sub-batches.  This makes a
            # safe throughput-only batch-size change resume without changing
            # or recomputing completed embeddings.
            for length in range(end - cursor, 0, -1):
                subset = rows[cursor : cursor + length]
                signature = hashlib.sha256(
                    json.dumps(
                        {
                            "salt": checkpoint_salt,
                            "kind": kind,
                            "rows": subset,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ).encode("utf-8")
                ).hexdigest()
                checkpoint = checkpoint_dir / (
                    f"{cursor:06d}_{length:04d}_{signature}.npy"
                )
                if not checkpoint.is_file():
                    continue
                try:
                    cached = np.load(checkpoint)
                    if cached.ndim != 2 or cached.shape[0] != length:
                        raise ValueError("checkpoint shape mismatch")
                    found = (
                        length,
                        cached.astype(np.float32, copy=False),
                    )
                    break
                except (OSError, ValueError):
                    continue
            if found is None:
                break
            length, cached = found
            pieces.append(cached)
            cursor += length
            reused_rows += length

        if cursor < end:
            missing = rows[cursor:end]
            yield_if_other_process_appears(physical_gpu)
            if kind == "images":
                inputs = [
                    {
                        "image": row["path"],
                        "instruction": (
                            "Represent the financial report page for question retrieval."
                        ),
                    }
                    for row in missing
                ]
            else:
                inputs = [
                    {
                        "text": row["text"],
                        "instruction": (
                            "Represent the financial question for report-page retrieval."
                        ),
                    }
                    for row in missing
                ]
            tensor = embedder.process(inputs, normalize=True)
            computed = (
                tensor.detach()
                .float()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
            signature = hashlib.sha256(
                json.dumps(
                    {
                        "salt": checkpoint_salt,
                        "kind": kind,
                        "rows": missing,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()
            checkpoint = checkpoint_dir / (
                f"{cursor:06d}_{len(missing):04d}_{signature}.npy"
            )
            save_npy(checkpoint, computed)
            pieces.append(computed)
            yield_if_other_process_appears(physical_gpu)
        value = np.concatenate(pieces, axis=0)
        if value.shape[0] != len(batch):
            raise ValueError("assembled checkpoint shape mismatch")
        outputs.append(value)
        done = min(offset + len(batch), len(rows))
        print(
            json.dumps(
                {
                    "stage": kind,
                    "done": done,
                    "total": len(rows),
                    "checkpoint_rows_reused": reused_rows,
                    "elapsed_seconds": round(time.time() - started, 2),
                }
            ),
            flush=True,
        )
    return np.concatenate(outputs, axis=0).astype(np.float32, copy=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    parser.add_argument("--model", type=Path, default=MODEL_DEFAULT)
    parser.add_argument("--image-batch-size", type=int, default=1)
    parser.add_argument("--query-batch-size", type=int, default=8)
    parser.add_argument("--max-pixels", type=int, default=1024 * 1024)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the formal visual cache")
    root = args.root.resolve()
    tasks_path = Path(os.environ.get("SERF_GOLD_TASKS", str(root / "data/annotation/gold_v1/tasks_gold_v1.json")))
    media_root = Path(os.environ.get("SERF_GOLD_MEDIA", str(root / "data/annotation/gold_v1")))
    output = Path(os.environ.get("SERF_GOLD_CACHE", str(root / "artifacts/gold_v3/cache")))
    tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
    if len(tasks) != 45:
        raise ValueError(f"expected 45 frozen tasks, got {len(tasks)}")

    image_by_candidate: dict[str, dict[str, Any]] = {}
    for task in tasks:
        for candidate in task["candidate_pages"]:
            candidate_id = candidate["candidate_id"]
            image_path = (media_root / candidate["media_path"]).resolve()
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            row = {
                "candidate_id": candidate_id,
                "media_path": candidate["media_path"],
                "path": str(image_path),
                "sha256": sha256(image_path),
            }
            previous = image_by_candidate.get(candidate_id)
            if previous and previous != row:
                raise ValueError(f"inconsistent candidate mapping: {candidate_id}")
            image_by_candidate[candidate_id] = row
    images = [image_by_candidate[key] for key in sorted(image_by_candidate)]

    queries: list[dict[str, Any]] = []
    for task in sorted(tasks, key=lambda row: row["canonical_id"]):
        canonical_id = task["canonical_id"]
        question = task["question"].strip()
        queries.append(
            {
                "query_key": f"{canonical_id}::full",
                "canonical_id": canonical_id,
                "kind": "full",
                "slot_id": "full",
                "text": question,
            }
        )
        for slot_id, text in split_question(question):
            queries.append(
                {
                    "query_key": f"{canonical_id}::{slot_id}",
                    "canonical_id": canonical_id,
                    "kind": "claim_slot",
                    "slot_id": slot_id,
                    "text": text,
                }
            )

    first_preflight = preflight_gpu()
    time.sleep(5)
    second_preflight = preflight_gpu()
    physical_gpu = second_preflight["physical_gpu"]
    import qwen_vl_utils
    qwen_vl_utils_path = Path(qwen_vl_utils.__file__).resolve().parent
    sys.path.insert(0, str(args.model / "scripts"))
    from qwen3_vl_embedding import Qwen3VLEmbedder

    started = time.time()
    embedder = Qwen3VLEmbedder(
        model_name_or_path=str(args.model),
        max_length=4096,
        max_pixels=args.max_pixels,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
    )
    salt = (
        f"model={args.model.resolve()}|max_length=4096|"
        f"max_pixels={args.max_pixels}|dtype=float16|attention=sdpa"
    )
    try:
        image_embeddings = embed_batches(
            embedder,
            images,
            kind="images",
            batch_size=args.image_batch_size,
            checkpoint_dir=output / "checkpoints/images",
            checkpoint_salt=salt,
            physical_gpu=physical_gpu,
        )
        query_embeddings = embed_batches(
            embedder,
            queries,
            kind="queries",
            batch_size=args.query_batch_size,
            checkpoint_dir=output / "checkpoints/queries",
            checkpoint_salt=salt,
            physical_gpu=physical_gpu,
        )
    except SharedGpuBecameOccupied as exc:
        print(
            json.dumps(
                {
                    "status": "COOPERATIVE_GPU_YIELD",
                    "reason": str(exc),
                    "checkpoint_dir": str(output / "checkpoints"),
                }
            ),
            flush=True,
        )
        return 75

    save_npy(output / "image_embeddings.npy", image_embeddings)
    save_npy(output / "query_embeddings.npy", query_embeddings)
    write_json(output / "images.json", images)
    write_json(output / "queries.json", queries)
    manifest = {
        "schema_version": "fedresolve-gold-v1-visual-cache-v1",
        "status": "COMPLETE",
        "dataset_version": "fedfinrag-resolve-gold-v1.0-20260730",
        "tasks_sha256": sha256(tasks_path),
        "model_path": str(args.model.resolve()),
        "model_copied_into_task": False,
        "model_license": "Apache-2.0 per local official model metadata",
        "qwen_vl_utils_reference": {
            "task_path": str(qwen_vl_utils_path),
            "resolved_path": str(qwen_vl_utils_path.resolve()),
            "copied_into_task": False,
        },
        "images": len(images),
        "queries": len(queries),
        "image_shape": list(image_embeddings.shape),
        "query_shape": list(query_embeddings.shape),
        "max_pixels": args.max_pixels,
        "torch_dtype": "float16",
        "attention": "sdpa",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_preflight": [first_preflight, second_preflight],
        "runtime_guard": "yield on any other compute PID; checkpoints retained",
        "gold_labels_read": False,
        "inputs_read": [
            "frozen question text",
            "candidate_id",
            "candidate page image",
        ],
        "elapsed_seconds": time.time() - started,
        "artifacts": {
            "image_embeddings.npy": sha256(output / "image_embeddings.npy"),
            "query_embeddings.npy": sha256(output / "query_embeddings.npy"),
            "images.json": sha256(output / "images.json"),
            "queries.json": sha256(output / "queries.json"),
        },
    }
    write_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
