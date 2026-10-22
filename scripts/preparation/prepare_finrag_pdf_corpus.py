#!/usr/bin/env python3
"""Safely materialize and index the verified FinRAGBench-V QA PDFs.

All generated files remain below the task's data/ directory. The extractor rejects
links, devices, absolute paths, and parent traversal. Page text is stored per PDF
as gzip JSONL with explicit source provenance so indexing can resume safely.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import shutil
import tarfile
import time
import unicodedata
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

import fitz


ROOT_DEFAULT = Path(os.environ.get("SERF_ROOT", str(Path(__file__).resolve().parents[2])))


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative_member(name: str) -> Path:
    pure = PurePosixPath(name)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"unsafe archive member: {name!r}")
    parts = list(pure.parts)
    if parts and parts[0] == "pdfs_selected_for_queries":
        parts = parts[1:]
    if not parts:
        raise ValueError(f"empty archive member after prefix removal: {name!r}")
    return Path(*parts)


def extract_archive(archive: Path, destination: Path, lang: str) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    extracted = 0
    skipped = 0
    logical_bytes = 0
    rejected: list[str] = []
    started = time.time()
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            if member.isdir():
                continue
            if not member.isfile():
                rejected.append(f"{member.name}:type={member.type!r}")
                continue
            try:
                relative = safe_relative_member(member.name)
            except ValueError as exc:
                rejected.append(str(exc))
                continue
            if relative.suffix.casefold() != ".pdf":
                rejected.append(f"{member.name}:not_pdf")
                continue
            target = destination / relative
            target_resolved = target.resolve(strict=False)
            destination_resolved = destination.resolve()
            if destination_resolved not in target_resolved.parents:
                rejected.append(f"{member.name}:resolved_outside_destination")
                continue
            logical_bytes += member.size
            if target.is_file() and target.stat().st_size == member.size:
                skipped += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = tar.extractfile(member)
            if source is None:
                rejected.append(f"{member.name}:extractfile_none")
                continue
            partial = target.with_suffix(target.suffix + ".partial")
            with partial.open("wb") as output:
                shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
                output.flush()
                os.fsync(output.fileno())
            if partial.stat().st_size != member.size:
                raise RuntimeError(
                    f"size mismatch for {member.name}: "
                    f"{partial.stat().st_size} != {member.size}"
                )
            os.replace(partial, target)
            extracted += 1
    return {
        "lang": lang,
        "archive": str(archive),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": sha256_file(archive),
        "destination": str(destination),
        "pdfs_extracted": extracted,
        "pdfs_already_complete": skipped,
        "logical_pdf_bytes": logical_bytes,
        "rejected_members": rejected,
        "elapsed_seconds": time.time() - started,
    }


def pdf_output_name(relative_path: str) -> str:
    digest = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()
    return f"{digest}.jsonl.gz"


def index_one_pdf(job: tuple[str, str, str]) -> dict:
    lang, source_string, output_string = job
    source = Path(source_string)
    output = Path(output_string)
    if output.is_file() and output.stat().st_size > 64:
        page_count = 0
        text_characters = 0
        with gzip.open(output, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                page_count += 1
                text_characters += int(row.get("text_characters", 0))
        return {
            "lang": lang,
            "source_path": str(source),
            "output_path": str(output),
            "status": "already_indexed",
            "page_count": page_count,
            "text_characters": text_characters,
            "source_bytes": source.stat().st_size,
            "output_bytes": output.stat().st_size,
        }
    started = time.time()
    temp = output.with_suffix(output.suffix + ".partial")
    output.parent.mkdir(parents=True, exist_ok=True)
    page_count = 0
    text_characters = 0
    metadata: dict = {}
    try:
        with fitz.open(source) as document:
            page_count = document.page_count
            metadata = {
                str(key): str(value)
                for key, value in (document.metadata or {}).items()
                if value is not None
            }
            with gzip.open(temp, "wt", encoding="utf-8", compresslevel=5) as handle:
                for page_index in range(page_count):
                    page = document.load_page(page_index)
                    text = page.get_text("text", sort=True)
                    text_characters += len(text)
                    record = {
                        "schema_version": "finrag-page-text-v1",
                        "lang": lang,
                        "source_path": str(source),
                        "pdf_stem": source.stem,
                        "page_index_0": page_index,
                        "page_number_1": page_index + 1,
                        "text": text,
                        "text_characters": len(text),
                        "image_count": len(page.get_images(full=True)),
                        "width": float(page.rect.width),
                        "height": float(page.rect.height),
                    }
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.replace(temp, output)
        return {
            "lang": lang,
            "source_path": str(source),
            "output_path": str(output),
            "status": "indexed",
            "page_count": page_count,
            "text_characters": text_characters,
            "metadata": metadata,
            "source_bytes": source.stat().st_size,
            "output_bytes": output.stat().st_size,
            "elapsed_seconds": time.time() - started,
        }
    except Exception as exc:  # preserve all failures in the manifest
        return {
            "lang": lang,
            "source_path": str(source),
            "output_path": str(output),
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "elapsed_seconds": time.time() - started,
        }


def iter_pdf_files(root: Path) -> list[Path]:
    """Enumerate PDFs without assuming a lowercase filename extension."""
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.casefold() == ".pdf"
    )


def index_pdfs(
    extracted_root: Path, index_root: Path, workers: int, limit: int | None
) -> dict:
    jobs: list[tuple[str, str, str]] = []
    for lang in ("ch", "en"):
        source_root = extracted_root / lang
        for source in iter_pdf_files(source_root):
            relative = source.relative_to(extracted_root).as_posix()
            output = index_root / lang / pdf_output_name(relative)
            jobs.append((lang, str(source), str(output)))
    if limit is not None:
        jobs = jobs[:limit]
    started = time.time()
    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(index_one_pdf, job) for job in jobs]
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda row: row["source_path"])
    return {
        "schema_version": "finrag-page-index-manifest-v1",
        "created_unix": time.time(),
        "workers": workers,
        "job_count": len(jobs),
        "status_counts": {
            status: sum(row["status"] == status for row in results)
            for status in sorted({row["status"] for row in results})
        },
        "total_pages": sum(int(row.get("page_count", 0)) for row in results),
        "total_text_characters": sum(
            int(row.get("text_characters", 0)) for row in results
        ),
        "elapsed_seconds": time.time() - started,
        "documents": results,
    }


def normalized_alias(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return "".join(char for char in text if char.isalnum())


def corpus_document_key(corpus_id: str) -> str:
    value = corpus_id.removesuffix(".png")
    value, multipage_count = re.subn(
        r"_multipage_\d+(?:-\d+)+$", "", value, flags=re.I
    )
    if multipage_count == 0:
        value = re.sub(r"_\s*\d+$", "", value)
    return value


def corpus_page_label(corpus_id: str) -> str:
    value = corpus_id.removesuffix(".png")
    multipage = re.search(
        r"_multipage_(\d+(?:-\d+)+)$", value, flags=re.I
    )
    if multipage:
        return multipage.group(1)
    return value.rsplit("_", 1)[-1]


def qrel_mapping_audit(
    dataset_root: Path, extracted_root: Path, output_jsonl: Path
) -> dict:
    pdfs = [
        (lang, path)
        for lang in ("ch", "en")
        for path in iter_pdf_files(extracted_root / lang)
    ]
    alias_to_pdf: dict[tuple[str, str], list[Path]] = {}
    for lang, path in pdfs:
        alias_to_pdf.setdefault((lang, normalized_alias(path.stem)), []).append(path)

    rows: list[dict] = []
    for lang in ("ch", "en"):
        qrel_path = dataset_root / "qrels" / f"qrels_{lang}.tsv"
        with qrel_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                corpus_id = row["corpus-id"]
                doc_key = corpus_document_key(corpus_id)
                matches = alias_to_pdf.get((lang, normalized_alias(doc_key)), [])
                rows.append(
                    {
                        "lang": lang,
                        "query_id": row["query-id"],
                        "corpus_id": corpus_id,
                        "document_key": doc_key,
                        "page_number": corpus_page_label(corpus_id),
                        "mapping_status": (
                            "exact_unique"
                            if len(matches) == 1
                            else ("exact_ambiguous" if matches else "unmapped")
                        ),
                        "pdf_matches": [str(path) for path in matches],
                    }
                )
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    temp = output_jsonl.with_suffix(output_jsonl.suffix + ".partial")
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temp, output_jsonl)
    statuses = {
        status: sum(row["mapping_status"] == status for row in rows)
        for status in sorted({row["mapping_status"] for row in rows})
    }
    unique_docs = {
        (row["lang"], row["document_key"], row["mapping_status"])
        for row in rows
    }
    doc_statuses = {
        status: sum(item[2] == status for item in unique_docs)
        for status in sorted({item[2] for item in unique_docs})
    }
    return {
        "schema_version": "finrag-qrel-pdf-mapping-audit-v1",
        "qrel_rows": len(rows),
        "qrel_mapping_status_counts": statuses,
        "unique_document_status_counts": doc_statuses,
        "pdf_count": len(pdfs),
        "output_jsonl": str(output_jsonl),
        "important_scope": (
            "Only normalized exact filename matches are accepted. Unmapped rows "
            "must not be auto-promoted using answer text."
        ),
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    parser.add_argument(
        "--actions",
        nargs="+",
        choices=["extract", "index", "audit"],
        default=["extract", "index", "audit"],
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("workers must be in [1, 8] for this extractor")

    dataset_root = Path(os.environ.get("SERF_FINRAG", str(args.root / "data" / "datasets" / "FinRAGBench-V")))
    processed_root = args.root / "data" / "processed" / "FinRAGBench-V"
    extracted_root = processed_root / "pdfs_for_QA"
    index_root = processed_root / "page_text"
    artifact_root = args.root / "artifacts" / "stage2" / "pdf_corpus"
    summary: dict = {
        "schema_version": "finrag-pdf-corpus-preparation-v1",
        "created_unix": time.time(),
        "actions": args.actions,
        "workers": args.workers,
        "limit": args.limit,
        "source_license": "Apache-2.0 per official FinRAGBench-V dataset card",
    }

    if "extract" in args.actions:
        extraction = []
        for lang in ("ch", "en"):
            archive = dataset_root / "pdfs_for_QA" / f"pdf_{lang}.tar.gz"
            extraction.append(
                extract_archive(archive, extracted_root / lang, lang)
            )
        summary["extraction"] = extraction
        write_json(artifact_root / "extraction_manifest.json", summary)
    elif (artifact_root / "extraction_manifest.json").is_file():
        previous = json.loads(
            (artifact_root / "extraction_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        summary["extraction"] = previous.get("extraction", [])

    if "index" in args.actions:
        index_manifest = index_pdfs(
            extracted_root, index_root, workers=args.workers, limit=args.limit
        )
        summary["index"] = {
            key: value
            for key, value in index_manifest.items()
            if key != "documents"
        }
        write_json(Path(os.environ.get("SERF_FINRAG_MANIFEST", str(artifact_root / "page_index_manifest.json"))), index_manifest)
    elif Path(os.environ.get("SERF_FINRAG_MANIFEST", str(artifact_root / "page_index_manifest.json"))).is_file():
        previous_index = json.loads(
            Path(os.environ.get("SERF_FINRAG_MANIFEST", str(artifact_root / "page_index_manifest.json"))).read_text(
                encoding="utf-8"
            )
        )
        summary["index"] = {
            key: value
            for key, value in previous_index.items()
            if key != "documents"
        }

    if "audit" in args.actions:
        mapping = qrel_mapping_audit(
            dataset_root,
            extracted_root,
            artifact_root / "qrel_pdf_mapping.jsonl",
        )
        summary["mapping_audit"] = mapping
        write_json(artifact_root / "mapping_summary.json", mapping)

    write_json(artifact_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
