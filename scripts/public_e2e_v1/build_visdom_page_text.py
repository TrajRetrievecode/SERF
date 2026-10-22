#!/usr/bin/env python3
"""Extract per-page text for every VisDoM PDF in the official checkout.

CPU only, idempotent, resumable.  Failures are recorded per document instead
of aborting the run.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from e2e_common import ROOT, TRACK, record_pid, write_json
from e2e_corpus import visdom_extract_document, visdom_pdf_paths


def worker(item: tuple[str, str]) -> dict:
    doc_key, pdf_path = item
    return visdom_extract_document(doc_key, Path(pdf_path))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    record_pid("build_visdom_page_text")
    jobs = [(key, str(path)) for key, path in visdom_pdf_paths()]
    if args.limit:
        jobs = jobs[: args.limit]

    started = time.time()
    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(worker, job): job[0] for job in jobs}
        done = 0
        for future in as_completed(futures):
            results.append(future.result())
            done += 1
            if done % 100 == 0 or done == len(jobs):
                print(
                    f"[visdom-text] {done}/{len(jobs)} "
                    f"elapsed={time.time() - started:.0f}s",
                    flush=True,
                )

    status_counts: dict[str, int] = {}
    for row in results:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    failures = [row for row in results if row["status"] == "failed"]
    payload = {
        "schema_version": "visdom-page-text-manifest-v1",
        "created_unix": time.time(),
        "elapsed_seconds": time.time() - started,
        "workers": args.workers,
        "documents_seen": len(jobs),
        "status_counts": status_counts,
        "total_pages": sum(int(r["page_count"]) for r in results),
        "total_text_characters": sum(int(r["text_characters"]) for r in results),
        "failures": failures,
        "documents": sorted(results, key=lambda r: r["doc_key"]),
    }
    out = write_json(
        ROOT / "artifacts" / TRACK / "visdom_page_text_manifest.json", payload
    )
    print(
        json.dumps(
            {
                "output": str(out),
                "documents": len(jobs),
                "pages": payload["total_pages"],
                "failed": len(failures),
                "elapsed_seconds": payload["elapsed_seconds"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
