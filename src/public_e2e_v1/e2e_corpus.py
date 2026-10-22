#!/usr/bin/env python3
"""Page-level corpora for the two public datasets.

A page is the retrievable unit for both datasets.  Every page carries the
document key used by the simulated federation, so a client owns whole
documents and therefore whole page sets.  No qrel, answer or participation
label is consulted here.
"""

from __future__ import annotations

import gzip
import os
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from e2e_common import ROOT, TRACK, read_json, write_json

FINRAG_ROOT = Path(os.environ.get("SERF_FINRAG", str(ROOT / "data" / "datasets" / "FinRAGBench-V")))
FINRAG_PROCESSED = ROOT / "data" / "processed" / "FinRAGBench-V"
VISDOM_ROOT = Path(os.environ.get("SERF_VISDOM", str(ROOT / "data" / "datasets" / "VisDoM")))
CACHE = ROOT / "data" / "cache" / TRACK

VISDOM_TASKS = {
    "feta_tab": "feta_tab.csv",
    "paper_tab": "paper_tab.csv",
    "scigraphvqa": "scigraphqa.csv",
    "slidevqa": "slidevqa.csv",
    "spiqa": "spiqa.csv",
}


@dataclass(frozen=True)
class PageRecord:
    page_uid: str
    doc_key: str
    lang: str
    page_number: int
    source_path: str
    text_characters: int


def page_uid(doc_key: str, page_number: int) -> str:
    return f"{doc_key}#p{page_number}"


# --------------------------------------------------------------------------
# FinRAGBench-V
# --------------------------------------------------------------------------
def finrag_page_manifest() -> list[dict]:
    manifest = read_json(
        Path(os.environ.get("SERF_FINRAG_MANIFEST", str(ROOT / "artifacts" / "stage2" / "pdf_corpus" / "page_index_manifest.json")))
    )
    return manifest["documents"]


def finrag_pages() -> list[PageRecord]:
    """Every extracted page of the official QA PDF corpus, in a stable order."""
    pages: list[PageRecord] = []
    for entry in sorted(finrag_page_manifest(), key=lambda d: d["source_path"]):
        source = Path(entry["source_path"])
        doc_key = source.stem
        text_path = Path(entry["output_path"])
        if not text_path.exists():
            continue
        with gzip.open(text_path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                number = int(row["page_number_1"])
                pages.append(
                    PageRecord(
                        page_uid=page_uid(doc_key, number),
                        doc_key=doc_key,
                        lang=str(row.get("lang") or entry.get("lang") or ""),
                        page_number=number,
                        source_path=str(source),
                        text_characters=int(row.get("text_characters") or 0),
                    )
                )
    pages.sort(key=lambda p: (p.doc_key, p.page_number))
    return pages


def finrag_page_texts() -> Iterator[tuple[str, str]]:
    for entry in sorted(finrag_page_manifest(), key=lambda d: d["source_path"]):
        doc_key = Path(entry["source_path"]).stem
        text_path = Path(entry["output_path"])
        if not text_path.exists():
            continue
        with gzip.open(text_path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                yield page_uid(doc_key, int(row["page_number_1"])), str(
                    row.get("text") or ""
                )


# --------------------------------------------------------------------------
# VisDoM
# --------------------------------------------------------------------------
def visdom_pdf_paths() -> list[tuple[str, Path]]:
    """(doc_key, pdf_path) for every PDF present in the official checkout."""
    output: list[tuple[str, Path]] = []
    for task in sorted(VISDOM_TASKS):
        docs = VISDOM_ROOT / task / "docs"
        if not docs.is_dir():
            continue
        for pdf in sorted(docs.iterdir()):
            if pdf.suffix.lower() != ".pdf":
                continue
            output.append((f"{task}/docs/{pdf.name}", pdf))
    return output


def visdom_text_cache_path(doc_key: str) -> Path:
    import hashlib

    digest = hashlib.sha256(doc_key.encode("utf-8")).hexdigest()
    return CACHE / "visdom_page_text" / digest[:2] / f"{digest}.jsonl.gz"


def visdom_extract_document(doc_key: str, pdf_path: Path) -> dict:
    """Extract per-page text once and cache it.  Idempotent."""
    target = visdom_text_cache_path(doc_key)
    if target.exists():
        try:
            with gzip.open(target, "rt", encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
            return {
                "doc_key": doc_key,
                "status": "already_indexed",
                "page_count": len(rows),
                "text_characters": sum(int(r["text_characters"]) for r in rows),
            }
        except Exception:
            target.unlink(missing_ok=True)

    import fitz  # PyMuPDF

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp.gz")
    total_chars = 0
    pages = 0
    try:
        with fitz.open(pdf_path) as document, gzip.open(
            tmp, "wt", encoding="utf-8"
        ) as handle:
            for index, page in enumerate(document):
                text = page.get_text("text") or ""
                rect = page.rect
                row = {
                    "schema_version": "visdom-page-text-v1",
                    "doc_key": doc_key,
                    "source_path": str(pdf_path),
                    "page_index_0": index,
                    "page_number_1": index + 1,
                    "text": text,
                    "text_characters": len(text),
                    "width": float(rect.width),
                    "height": float(rect.height),
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                total_chars += len(text)
                pages += 1
    except Exception as error:
        tmp.unlink(missing_ok=True)
        return {
            "doc_key": doc_key,
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
            "page_count": 0,
            "text_characters": 0,
        }
    tmp.replace(target)
    return {
        "doc_key": doc_key,
        "status": "indexed",
        "page_count": pages,
        "text_characters": total_chars,
    }


def visdom_pages() -> list[PageRecord]:
    pages: list[PageRecord] = []
    for doc_key, pdf_path in visdom_pdf_paths():
        cache = visdom_text_cache_path(doc_key)
        if not cache.exists():
            continue
        with gzip.open(cache, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                number = int(row["page_number_1"])
                pages.append(
                    PageRecord(
                        page_uid=page_uid(doc_key, number),
                        doc_key=doc_key,
                        lang="en",
                        page_number=number,
                        source_path=str(pdf_path),
                        text_characters=int(row.get("text_characters") or 0),
                    )
                )
    pages.sort(key=lambda p: (p.doc_key, p.page_number))
    return pages


def visdom_page_texts() -> Iterator[tuple[str, str]]:
    for doc_key, _ in visdom_pdf_paths():
        cache = visdom_text_cache_path(doc_key)
        if not cache.exists():
            continue
        with gzip.open(cache, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                yield page_uid(doc_key, int(row["page_number_1"])), str(
                    row.get("text") or ""
                )


# --------------------------------------------------------------------------
# generic accessors
# --------------------------------------------------------------------------
def _pages_uncached(dataset: str) -> list[PageRecord]:
    if dataset == "finrag":
        return finrag_pages()
    if dataset == "visdom":
        return visdom_pages()
    raise ValueError(f"unknown dataset: {dataset}")


def page_texts_for(dataset: str) -> Iterator[tuple[str, str]]:
    """Stream (page_uid, text) straight from the per-document sources."""
    if dataset == "finrag":
        return finrag_page_texts()
    if dataset == "visdom":
        return visdom_page_texts()
    raise ValueError(f"unknown dataset: {dataset}")


def consolidated_paths(dataset: str) -> tuple[Path, Path]:
    return (
        CACHE / f"{dataset}_pages.json",
        CACHE / f"{dataset}_page_text.jsonl.gz",
    )


def build_consolidated_cache(dataset: str) -> dict:
    """Collapse thousands of per-document files into two fast-loading files."""
    pages_path, text_path = consolidated_paths(dataset)
    pages_path.parent.mkdir(parents=True, exist_ok=True)
    pages = _pages_uncached(dataset)
    tmp = pages_path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(
            [
                {
                    "page_uid": p.page_uid,
                    "doc_key": p.doc_key,
                    "lang": p.lang,
                    "page_number": p.page_number,
                    "source_path": p.source_path,
                    "text_characters": p.text_characters,
                }
                for p in pages
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    tmp.replace(pages_path)

    tmp_text = text_path.with_suffix(".tmp.gz")
    written = 0
    with gzip.open(tmp_text, "wt", encoding="utf-8") as handle:
        for uid, text in page_texts_for(dataset):
            handle.write(json.dumps({"u": uid, "t": text}, ensure_ascii=False) + "\n")
            written += 1
    tmp_text.replace(text_path)
    return {
        "dataset": dataset,
        "pages": len(pages),
        "documents": len({p.doc_key for p in pages}),
        "text_rows": written,
        "pages_path": str(pages_path),
        "text_path": str(text_path),
    }


_PAGE_CACHE: dict[str, list[PageRecord]] = {}


def pages_for(dataset: str) -> list[PageRecord]:
    if dataset in _PAGE_CACHE:
        return _PAGE_CACHE[dataset]
    pages_path, _ = consolidated_paths(dataset)
    if pages_path.exists():
        rows = json.loads(pages_path.read_text(encoding="utf-8"))
        pages = [
            PageRecord(
                page_uid=row["page_uid"],
                doc_key=row["doc_key"],
                lang=row["lang"],
                page_number=int(row["page_number"]),
                source_path=row["source_path"],
                text_characters=int(row["text_characters"]),
            )
            for row in rows
        ]
    else:
        pages = _pages_uncached(dataset)
    _PAGE_CACHE[dataset] = pages
    return pages


def page_text_map(dataset: str) -> dict[str, str]:
    _, text_path = consolidated_paths(dataset)
    if text_path.exists():
        output: dict[str, str] = {}
        with gzip.open(text_path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                output[row["u"]] = row["t"]
        return output
    return {uid: text for uid, text in page_texts_for(dataset)}


_GOLD_PAGE_RE = re.compile(r"_(\d+)\.png$", re.IGNORECASE)
_GOLD_MULTIPAGE_RE = re.compile(r"_multipage_(\d+(?:-\d+)+)\.png$", re.IGNORECASE)


def finrag_gold_pages(corpus_ids: Iterable[str], doc_keys: Iterable[str]) -> list[str]:
    """Map official qrel corpus ids onto page uids of the extracted corpus."""
    output: list[str] = []
    for corpus_id, doc_key in zip(corpus_ids, doc_keys):
        multipage = _GOLD_MULTIPAGE_RE.search(corpus_id)
        if multipage:
            for part in multipage.group(1).split("-"):
                output.append(page_uid(doc_key, int(part)))
            continue
        single = _GOLD_PAGE_RE.search(corpus_id)
        if single:
            output.append(page_uid(doc_key, int(single.group(1))))
    return list(dict.fromkeys(output))
