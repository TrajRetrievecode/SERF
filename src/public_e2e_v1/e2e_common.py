#!/usr/bin/env python3
"""Shared utilities for the public end-to-end experiment track (public_e2e_v1).

Nothing in this module reads gold answers, qrels or participation labels at
inference time.  Label-consuming helpers live in the scoring section and are
only called by the evaluator.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(os.environ.get("SERF_ROOT", str(Path(__file__).resolve().parents[2])))
TRACK = "public_e2e_v1"


# --------------------------------------------------------------------------
# hashing / io
# --------------------------------------------------------------------------
def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)
    return path


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class JsonlAppender:
    """Append writer used for resumability.

    Every record is flushed to the operating system immediately, so a process
    that dies loses nothing.  ``fsync`` is issued periodically rather than per
    record: syncing every line costs more than the whole computation on this
    filesystem, and only a machine-level crash could lose the last few lines,
    which the resume logic re-derives anyway.
    """

    def __init__(self, path: Path, sync_every: int = 500):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")
        self._sync_every = max(1, int(sync_every))
        self._since_sync = 0

    def write(self, row: dict) -> None:
        self._handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._handle.flush()
        self._since_sync += 1
        if self._since_sync >= self._sync_every:
            os.fsync(self._handle.fileno())
            self._since_sync = 0

    def close(self) -> None:
        try:
            self._handle.flush()
            os.fsync(self._handle.fileno())
        except Exception:
            pass
        try:
            self._handle.close()
        except Exception:
            pass

    def __enter__(self) -> "JsonlAppender":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


# --------------------------------------------------------------------------
# GPU cooperation.  We never kill, suspend or renice anything.
# --------------------------------------------------------------------------
NVIDIA_QUERY = (
    "index,name,memory.total,memory.free,memory.used,utilization.gpu"
)


def gpu_snapshot() -> list[dict]:
    try:
        raw = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={NVIDIA_QUERY}",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        ).stdout
    except Exception as error:  # pragma: no cover - environment dependent
        return [{"error": f"{type(error).__name__}: {error}"}]
    rows = []
    for line in raw.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 6:
            continue
        rows.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "memory_total_mib": int(parts[2]),
                "memory_free_mib": int(parts[3]),
                "memory_used_mib": int(parts[4]),
                "utilization_percent": int(parts[5]),
            }
        )
    return rows


def gpu_processes() -> list[dict]:
    try:
        raw = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        ).stdout
    except Exception as error:  # pragma: no cover
        return [{"error": f"{type(error).__name__}: {error}"}]
    rows = []
    for line in raw.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue
        rows.append(
            {
                "gpu_uuid": parts[0],
                "pid": int(parts[1]),
                "process_name": parts[2],
                "used_memory_mib": int(parts[3]),
            }
        )
    return rows


def pick_gpu(required_free_mib: int, headroom_mib: int = 4096) -> dict:
    """Return the least-loaded GPU that leaves the incumbent workload room.

    Never returns a GPU whose free memory would drop below ``headroom_mib``
    after our own reservation.  Raises when no GPU qualifies so the caller can
    wait rather than squeeze an existing job.
    """
    snapshot = gpu_snapshot()
    usable = [
        row
        for row in snapshot
        if "error" not in row
        and row["memory_free_mib"] >= required_free_mib + headroom_mib
    ]
    if not usable:
        raise RuntimeError(
            "no GPU has "
            f"{required_free_mib + headroom_mib} MiB free; snapshot={snapshot}"
        )
    usable.sort(
        key=lambda row: (row["utilization_percent"], -row["memory_free_mib"])
    )
    return {"chosen": usable[0], "snapshot": snapshot, "processes": gpu_processes()}


def wait_for_gpu(
    required_free_mib: int,
    headroom_mib: int = 4096,
    poll_seconds: int = 60,
    max_wait_seconds: int = 7200,
    log=print,
) -> dict:
    started = time.time()
    attempt = 0
    while True:
        attempt += 1
        try:
            return pick_gpu(required_free_mib, headroom_mib)
        except RuntimeError as error:
            waited = time.time() - started
            if waited > max_wait_seconds:
                raise
            log(
                f"[gpu-wait] attempt={attempt} waited={waited:.0f}s {error}",
            )
            time.sleep(poll_seconds)


def record_pid(name: str, extra: dict | None = None) -> Path:
    payload = {
        "task": TRACK,
        "name": name,
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "started_unix": time.time(),
        "cwd": os.getcwd(),
        "argv_hint": name,
    }
    if extra:
        payload.update(extra)
    return write_json(ROOT / "artifacts" / TRACK / "pids" / f"{name}.json", payload)


# --------------------------------------------------------------------------
# answer scoring (unified across every method)
# --------------------------------------------------------------------------
_ARTICLES = {"a", "an", "the"}
_CJK = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
_PUNCT = re.compile(r"[^\w\s㐀-鿿.\-%/]", re.UNICODE)


def normalize_answer(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    text = _PUNCT.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def answer_tokens(text: str) -> list[str]:
    """Whitespace tokens for Latin script, character tokens for CJK runs."""
    normalized = normalize_answer(text)
    tokens: list[str] = []
    for chunk in normalized.split():
        if _CJK.search(chunk):
            buffer = ""
            for char in chunk:
                if _CJK.match(char):
                    if buffer:
                        tokens.append(buffer)
                        buffer = ""
                    tokens.append(char)
                else:
                    buffer += char
            if buffer:
                tokens.append(buffer)
        else:
            if chunk in _ARTICLES:
                continue
            tokens.append(chunk)
    return tokens


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def token_f1(prediction: str, reference: str) -> float:
    pred = answer_tokens(prediction)
    ref = answer_tokens(reference)
    if not pred or not ref:
        return float(pred == ref)
    common = Counter(pred) & Counter(ref)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def token_recall(prediction: str, reference: str) -> float:
    pred = answer_tokens(prediction)
    ref = answer_tokens(reference)
    if not ref:
        return float(not pred)
    common = Counter(pred) & Counter(ref)
    return sum(common.values()) / len(ref)


_NUMBER_RE = re.compile(r"-?\d+(?:[.,]\d+)*%?")


def extract_numbers(text: str) -> list[str]:
    output = []
    for raw in _NUMBER_RE.findall(unicodedata.normalize("NFKC", str(text))):
        value = raw.replace(",", "")
        percent = value.endswith("%")
        if percent:
            value = value[:-1]
        try:
            number = float(value)
        except ValueError:
            continue
        # Canonicalise to at most six decimals so 1.50 and 1.5 agree.
        output.append(f"{number:.6f}".rstrip("0").rstrip(".") + ("%" if percent else ""))
    return output


def numeric_recall(prediction: str, reference: str) -> float | None:
    ref = extract_numbers(reference)
    if not ref:
        return None
    pred = Counter(extract_numbers(prediction))
    common = pred & Counter(ref)
    return sum(common.values()) / len(ref)


def numeric_accuracy(prediction: str, reference: str) -> float | None:
    """1.0 when every reference number appears in the prediction."""
    value = numeric_recall(prediction, reference)
    if value is None:
        return None
    return float(value >= 1.0)


# --------------------------------------------------------------------------
# retrieval / citation scoring
# --------------------------------------------------------------------------
def recall_at_k(ranked: Sequence[str], gold: Iterable[str], k: int) -> float:
    gold_set = set(gold)
    if not gold_set:
        return float("nan")
    hit = len(gold_set.intersection(ranked[:k]))
    return hit / len(gold_set)


def prf(selected: Iterable[str], gold: Iterable[str]) -> dict[str, float]:
    selected_set = set(selected)
    gold_set = set(gold)
    if not gold_set:
        return {"precision": float("nan"), "recall": float("nan"), "f1": float("nan")}
    hit = len(selected_set & gold_set)
    precision = hit / len(selected_set) if selected_set else 0.0
    recall = hit / len(gold_set)
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def package_bytes(payload: Any) -> int:
    """Actual serialized byte count of a request/response payload."""
    return len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
