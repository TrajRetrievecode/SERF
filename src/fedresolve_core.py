#!/usr/bin/env python3
"""Core components for a reproducible FedResolve routing/certification audit.

This module intentionally separates:
  * official benchmark observations (queries and qrels),
  * a deterministic, query-independent simulated federation,
  * protocol-level adapters for external methods, and
  * the FedResolve implementation evaluated in this repository.

The simulated federation is not an official split from FinRAGBench-V.
"""

from __future__ import annotations

import csv
import ast
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import joblib
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize


YEAR_RE = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
PAGE_RE = re.compile(r"_(\d+)\.png$", re.IGNORECASE)
MULTIPAGE_RE = re.compile(
    r"_multipage_(\d+(?:-\d+)+)\.png$", re.IGNORECASE
)
NASDAQ_RE = re.compile(r"(?:^|_)NASDAQ_([A-Z0-9.\-]+)(?:_|$)", re.IGNORECASE)
LSE_RE = re.compile(r"(?:^|_)LSE_([A-Z0-9.\-]+)(?:_|$)", re.IGNORECASE)
CIK_RE = re.compile(r"(?:^|_)(\d{10})-\d{2}-\d{6}(?:_|$)")
STOCK_CODE_RE = re.compile(r"(?:^|__)([036]\d{5})(?:__|$)")


def stable_u64(text: str, seed: int = 0) -> int:
    digest = hashlib.blake2b(
        f"{seed}\0{text}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def mixed_financial_tokens(text: str) -> list[str]:
    """Lightweight multilingual tokenizer for persisted lexical evidence indexes."""
    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    words = re.findall(r"[a-z0-9]+(?:[._%/\-][a-z0-9]+)*", normalized)
    tokens = [f"w:{word}" for word in words if len(word) >= 2]
    tokens.extend(
        f"w2:{left}_{right}"
        for left, right in zip(words[:-1], words[1:])
        if len(left) >= 2 and len(right) >= 2
    )
    for run in re.findall(r"[\u3400-\u9fff]+", normalized):
        tokens.extend(f"c:{char}" for char in run)
        tokens.extend(
            f"c2:{run[index:index+2]}" for index in range(len(run) - 1)
        )
    return tokens


def document_key(corpus_id: str) -> str:
    value = re.sub(r"\.png$", "", corpus_id, flags=re.IGNORECASE)
    value, multipage_count = re.subn(
        r"_multipage_\d+(?:-\d+)+$", "", value, flags=re.IGNORECASE
    )
    # A multipage suffix already contains the page label.  Do not then strip
    # the document year that becomes the final numeric component.
    if multipage_count == 0:
        value = re.sub(r"_\s*\d+$", "", value)
    return value


def page_number(corpus_id: str) -> int | None:
    multipage = MULTIPAGE_RE.search(corpus_id)
    if multipage:
        return int(multipage.group(1).split("-", 1)[0])
    match = PAGE_RE.search(corpus_id)
    return int(match.group(1)) if match else None


def family_key(doc_key: str) -> str:
    """Derive a source/entity family without consulting any query or qrel label."""
    normalized = unicodedata.normalize("NFKC", doc_key)
    match = NASDAQ_RE.search(normalized)
    if match:
        return f"NASDAQ::{match.group(1).upper()}"
    match = LSE_RE.search(normalized)
    if match:
        return f"LSE::{match.group(1).upper()}"
    match = CIK_RE.search(normalized)
    if match:
        return f"SEC-CIK::{match.group(1)}"
    match = STOCK_CODE_RE.search(normalized)
    if match:
        return f"CN-STOCK::{match.group(1)}"

    # Chinese annual-report identifiers often contain date/company/code/name/year.
    parts = [p.strip() for p in normalized.split("__") if p.strip()]
    if len(parts) >= 3:
        without_dates = [p for p in parts if not YEAR_RE.fullmatch(p)]
        if len(without_dates) >= 2:
            return "CN-ENTITY::" + normalize_text("::".join(without_dates[1:3]))

    reduced = YEAR_RE.sub("<YEAR>", normalized)
    reduced = re.sub(r"\b(?:q[1-4]|quarter|annual|report|reports)\b", " ", reduced, flags=re.I)
    reduced = re.sub(r"[_\W]+", " ", reduced, flags=re.UNICODE)
    reduced = normalize_text(reduced)
    if re.fullmatch(r"[0-9a-f]{24,}", reduced.replace(" ", "")):
        return f"OPAQUE::{reduced[:16]}"
    return f"TITLE::{reduced[:80] or normalize_text(doc_key)[:80]}"


@dataclass(frozen=True)
class QueryRecord:
    uid: str
    lang: str
    query_id: str
    query: str
    answer: str
    category: str
    answer_type: str
    from_pages: str
    corpus_ids: tuple[str, ...]
    doc_keys: tuple[str, ...]
    page_numbers: tuple[int | None, ...]


def load_finrag_records(dataset_root: Path) -> tuple[list[QueryRecord], dict]:
    records: list[QueryRecord] = []
    audit: dict[str, dict] = {}
    for lang in ("ch", "en"):
        qpath = dataset_root / "queries" / f"queries_{lang}.json"
        rpath = dataset_root / "qrels" / f"qrels_{lang}.tsv"
        with qpath.open("r", encoding="utf-8") as handle:
            rows = json.load(handle)
        qrels: dict[str, list[str]] = defaultdict(list)
        with rpath.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                if float(row["score"]) > 0:
                    qrels[row["query-id"]].append(row["corpus-id"])

        duplicate_ids = len(rows) - len({str(row["query-id"]) for row in rows})
        missing_qrel = 0
        for ordinal, row in enumerate(rows):
            qid = str(row["query-id"])
            corpus_ids = tuple(dict.fromkeys(qrels.get(qid, [])))
            if not corpus_ids:
                missing_qrel += 1
                continue
            # Prefix language and ordinal so the two known duplicate query ids remain
            # separately auditable instead of being silently overwritten.
            uid = f"{lang}:{qid}:{ordinal}"
            records.append(
                QueryRecord(
                    uid=uid,
                    lang=lang,
                    query_id=qid,
                    query=str(row.get("query", "")),
                    answer=str(row.get("answer", "")),
                    category=str(row.get("category", "")),
                    answer_type=str(row.get("answer_type", "")),
                    from_pages=str(row.get("from_pages", "")),
                    corpus_ids=corpus_ids,
                    doc_keys=tuple(document_key(x) for x in corpus_ids),
                    page_numbers=tuple(page_number(x) for x in corpus_ids),
                )
            )
        audit[lang] = {
            "query_rows": len(rows),
            "duplicate_query_id_rows": duplicate_ids,
            "positive_qrel_query_ids": len(qrels),
            "query_rows_without_positive_qrel_excluded": missing_qrel,
        }
    return records, audit


def load_visdom_records(
    dataset_root: Path,
) -> tuple[list[QueryRecord], dict, tuple[str, ...]]:
    """Load the five official VisDoM CSV tasks without requiring PDF parsing."""
    task_files = {
        "feta_tab": dataset_root / "feta_tab" / "feta_tab.csv",
        "paper_tab": dataset_root / "paper_tab" / "paper_tab.csv",
        "scigraphvqa": dataset_root / "scigraphvqa" / "scigraphqa.csv",
        "slidevqa": dataset_root / "slidevqa" / "slidevqa.csv",
        "spiqa": dataset_root / "spiqa" / "spiqa.csv",
    }
    records: list[QueryRecord] = []
    inventory: set[str] = set()
    audit: dict[str, dict] = {}
    for task, path in task_files.items():
        task_rows = 0
        bad_candidates = 0
        missing_gold = 0
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for ordinal, row in enumerate(csv.DictReader(handle)):
                task_rows += 1
                gold_name = str(row.get("doc_path") or "").strip()
                if not gold_name:
                    missing_gold += 1
                    continue
                gold_doc = f"{task}/docs/{gold_name}"
                inventory.add(gold_doc)
                raw_candidates = str(row.get("documents") or "[]")
                try:
                    candidates = ast.literal_eval(raw_candidates)
                    if not isinstance(candidates, (list, tuple)):
                        raise ValueError("candidate field is not a list")
                    for name in candidates:
                        inventory.add(f"{task}/docs/{str(name).strip()}")
                except (ValueError, SyntaxError):
                    bad_candidates += 1
                qid = str(row.get("q_id") or ordinal)
                records.append(
                    QueryRecord(
                        uid=f"{task}:{qid}:{ordinal}",
                        lang="en",
                        query_id=qid,
                        query=str(row.get("question") or ""),
                        answer=str(row.get("answer") or ""),
                        category=task,
                        answer_type=str(row.get("answer_type") or ""),
                        from_pages=str(
                            row.get("evidence_pages")
                            or row.get("image_file")
                            or ""
                        ),
                        corpus_ids=(gold_doc,),
                        doc_keys=(gold_doc,),
                        page_numbers=(None,),
                    )
                )
        audit[task] = {
            "csv": str(path),
            "rows": task_rows,
            "records_used": sum(1 for r in records if r.category == task),
            "candidate_fields_unparsed": bad_candidates,
            "rows_without_gold_excluded": missing_gold,
        }
    return records, audit, tuple(sorted(inventory))


@dataclass(frozen=True)
class Federation:
    num_clients: int
    doc_to_client: Mapping[str, int]
    family_to_client: Mapping[str, int]
    client_documents: Mapping[int, tuple[str, ...]]
    family_loads: Mapping[int, int]

    def gold_clients(self, record: QueryRecord) -> tuple[int, ...]:
        return tuple(sorted({self.doc_to_client[d] for d in record.doc_keys}))


def build_federation(
    doc_keys: Iterable[str],
    num_clients: int,
    seed: int,
    assignment_unit: str = "family",
) -> Federation:
    """Build a query-independent capacity-balanced simulated federation.

    ``family`` co-locates entity/source families and is used for the main routing
    tracks. ``document`` distributes individual documents and is used only for the
    Resolve challenge, where cross-client complementary evidence must be possible.
    """
    if assignment_unit not in {"family", "document"}:
        raise ValueError(f"unsupported assignment_unit: {assignment_unit}")
    unique_docs = sorted(set(doc_keys))
    if assignment_unit == "document":
        loads = [0] * num_clients
        doc_to_client: dict[str, int] = {}
        for doc in sorted(
            unique_docs, key=lambda value: (stable_u64(value, seed), value)
        ):
            min_load = min(loads)
            candidates = [i for i, value in enumerate(loads) if value == min_load]
            choice = candidates[
                stable_u64(doc, seed + 17) % len(candidates)
            ]
            doc_to_client[doc] = choice
            loads[choice] += 1
        client_documents = {
            client: tuple(
                sorted(
                    doc for doc, owner in doc_to_client.items() if owner == client
                )
            )
            for client in range(num_clients)
        }
        # The legacy field name is retained for compatibility; document-qualified
        # keys make the non-atomic nature explicit and collision-free.
        qualified_assignment = {
            f"{family_key(doc)}\0{doc}": owner
            for doc, owner in doc_to_client.items()
        }
        return Federation(
            num_clients=num_clients,
            doc_to_client=doc_to_client,
            family_to_client=qualified_assignment,
            client_documents=client_documents,
            family_loads={i: loads[i] for i in range(num_clients)},
        )

    families: dict[str, list[str]] = defaultdict(list)
    for doc_key in unique_docs:
        families[family_key(doc_key)].append(doc_key)

    loads = [0] * num_clients
    family_to_client: dict[str, int] = {}
    ordered = sorted(
        families,
        key=lambda fam: (-len(families[fam]), stable_u64(fam, seed), fam),
    )
    for fam in ordered:
        min_load = min(loads)
        candidates = [i for i, value in enumerate(loads) if value == min_load]
        choice = candidates[stable_u64(fam, seed + 17) % len(candidates)]
        family_to_client[fam] = choice
        loads[choice] += len(families[fam])

    doc_to_client = {
        doc: family_to_client[fam]
        for fam, docs in families.items()
        for doc in docs
    }
    client_documents: dict[int, tuple[str, ...]] = {}
    for client in range(num_clients):
        client_documents[client] = tuple(
            sorted(doc for doc, owner in doc_to_client.items() if owner == client)
        )
    return Federation(
        num_clients=num_clients,
        doc_to_client=doc_to_client,
        family_to_client=family_to_client,
        client_documents=client_documents,
        family_loads={i: loads[i] for i in range(num_clients)},
    )


def split_records(
    records: Sequence[QueryRecord],
    track: str,
    seed: int,
) -> dict[str, list[QueryRecord]]:
    if track not in {"seen_query", "unseen_family"}:
        raise ValueError(f"unsupported track: {track}")
    output = {"train": [], "dev": [], "test": []}
    for record in records:
        if track == "seen_query":
            key = record.uid
        else:
            key = "|".join(sorted({family_key(d) for d in record.doc_keys}))
        bucket = stable_u64(key, seed) % 100
        split = "train" if bucket < 70 else ("dev" if bucket < 82 else "test")
        output[split].append(record)
    return output


class SupervisedClientRouter:
    """Supervised query-to-client classifier kept for the routing interface. The reported
    RAGRoute adaptation ranks clients by the full-query page maxima (e2e_slot_adapters)."""

    def __init__(self, num_clients: int, seed: int):
        self.num_clients = num_clients
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            min_df=2,
            max_df=0.995,
            sublinear_tf=True,
            max_features=70_000,
            dtype=np.float32,
        )
        self.model = LogisticRegression(
            C=4.0,
            max_iter=1200,
            class_weight="balanced",
            random_state=seed,
            solver="lbfgs",
        )

    @staticmethod
    def _text(record: QueryRecord) -> str:
        return f"{record.query} [CATEGORY] {record.category} [LANG] {record.lang}"

    def fit(self, records: Sequence[QueryRecord], federation: Federation) -> None:
        texts = [self._text(r) for r in records]
        labels = [federation.gold_clients(r)[0] for r in records]
        matrix = self.vectorizer.fit_transform(texts)
        self.model.fit(matrix, labels)

    def predict_proba(self, records: Sequence[QueryRecord]) -> np.ndarray:
        matrix = self.vectorizer.transform([self._text(r) for r in records])
        raw = self.model.predict_proba(matrix)
        output = np.full((len(records), self.num_clients), 1e-8, dtype=np.float64)
        output[:, self.model.classes_.astype(int)] = raw
        output /= output.sum(axis=1, keepdims=True)
        return output


class CapabilitySketchRouter:
    """Client capability sketches derived from query-independent document inventory."""

    def __init__(self, num_clients: int):
        self.num_clients = num_clients
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            min_df=1,
            sublinear_tf=True,
            max_features=90_000,
            dtype=np.float32,
        )
        self.centroids: sparse.csr_matrix | None = None

    @staticmethod
    def _describe_doc(doc_key: str) -> str:
        fam = family_key(doc_key)
        years = " ".join(YEAR_RE.findall(doc_key))
        return f"{doc_key} [FAMILY] {fam} [YEARS] {years}"

    def fit(self, federation: Federation, train_records: Sequence[QueryRecord]) -> None:
        doc_texts = [
            self._describe_doc(doc)
            for client in range(self.num_clients)
            for doc in federation.client_documents[client]
        ]
        train_texts = [r.query for r in train_records]
        self.vectorizer.fit(doc_texts + train_texts)
        rows = []
        for client in range(self.num_clients):
            texts = [self._describe_doc(d) for d in federation.client_documents[client]]
            matrix = self.vectorizer.transform(texts)
            centroid = sparse.csr_matrix(matrix.mean(axis=0))
            rows.append(centroid)
        self.centroids = normalize(sparse.vstack(rows).tocsr(), norm="l2", axis=1)

    def score(self, records: Sequence[QueryRecord]) -> np.ndarray:
        if self.centroids is None:
            raise RuntimeError("capability sketch is not fitted")
        query_matrix = normalize(
            self.vectorizer.transform([r.query for r in records]), norm="l2", axis=1
        )
        scores = (query_matrix @ self.centroids.T).toarray().astype(np.float64)
        # Convert similarities to a stable simplex while retaining small differences.
        scores -= scores.max(axis=1, keepdims=True)
        scores = np.exp(scores / 0.12)
        scores /= scores.sum(axis=1, keepdims=True)
        return scores


class TextEvidenceRouter:
    """Query-to-client router backed by query-independent document content."""

    def __init__(self, vectorizer, matrix: sparse.csr_matrix, documents: list[dict]):
        self.vectorizer = vectorizer
        self.matrix = normalize(matrix.tocsr(), norm="l2", axis=1)
        self.documents = documents

    @classmethod
    def load(cls, index_root: Path) -> "TextEvidenceRouter":
        vectorizer = joblib.load(index_root / "vectorizer.joblib")
        matrix = sparse.load_npz(index_root / "document_matrix.npz")
        documents = json.loads(
            (index_root / "documents.json").read_text(encoding="utf-8")
        )
        if matrix.shape[0] != len(documents):
            raise RuntimeError(
                f"text index row mismatch: {matrix.shape[0]} != {len(documents)}"
            )
        return cls(vectorizer, matrix, documents)

    def score(
        self, records: Sequence[QueryRecord], federation: Federation
    ) -> np.ndarray:
        query_matrix = normalize(
            self.vectorizer.transform([record.query for record in records]),
            norm="l2",
            axis=1,
        )
        similarities = (query_matrix @ self.matrix.T).toarray()
        scores = np.zeros((len(records), federation.num_clients), dtype=np.float64)
        for doc_row, document in enumerate(self.documents):
            doc_key = str(document["document_key"])
            if doc_key not in federation.doc_to_client:
                continue
            client = federation.doc_to_client[doc_key]
            scores[:, client] = np.maximum(
                scores[:, client], similarities[:, doc_row]
            )
        # Retain the strongest matching document while producing calibrated-input
        # positive values for every client.
        scores -= scores.max(axis=1, keepdims=True)
        scores = np.exp(scores / 0.055)
        scores /= scores.sum(axis=1, keepdims=True)
        return scores


class CachedDenseEvidenceRouter:
    """Client scores from a reproducibly cached dense document/query feature set."""

    def __init__(
        self,
        document_embeddings: np.ndarray,
        query_embeddings: np.ndarray,
        documents: list[dict],
        query_rows: list[dict],
    ):
        self.document_embeddings = normalize(
            np.asarray(document_embeddings, dtype=np.float32), norm="l2", axis=1
        )
        self.query_embeddings = normalize(
            np.asarray(query_embeddings, dtype=np.float32), norm="l2", axis=1
        )
        self.documents = documents
        self.query_uid_to_row = {
            str(row["uid"]): int(row["row"]) for row in query_rows
        }

    @classmethod
    def load(cls, cache_root: Path) -> "CachedDenseEvidenceRouter":
        return cls(
            np.load(cache_root / "document_embeddings.npy"),
            np.load(cache_root / "query_embeddings.npy"),
            json.loads(
                (cache_root / "documents.json").read_text(encoding="utf-8")
            ),
            json.loads(
                (cache_root / "queries.json").read_text(encoding="utf-8")
            ),
        )

    def score(
        self, records: Sequence[QueryRecord], federation: Federation
    ) -> np.ndarray:
        missing = [
            record.uid
            for record in records
            if record.uid not in self.query_uid_to_row
        ]
        if missing:
            raise KeyError(
                f"dense query cache missing {len(missing)} uids; sample={missing[:3]}"
            )
        rows = [self.query_uid_to_row[record.uid] for record in records]
        similarities = (
            self.query_embeddings[rows] @ self.document_embeddings.T
        )
        scores = np.zeros((len(records), federation.num_clients), dtype=np.float64)
        for doc_row, document in enumerate(self.documents):
            doc_key = str(document["document_key"])
            if doc_key not in federation.doc_to_client:
                continue
            client = federation.doc_to_client[doc_key]
            scores[:, client] = np.maximum(
                scores[:, client], similarities[:, doc_row]
            )
        scores -= scores.max(axis=1, keepdims=True)
        scores = np.exp(scores / 0.045)
        scores /= scores.sum(axis=1, keepdims=True)
        return scores


class ResidualCertifier:
    """Global calibration head plus smoothed client-specific logit residuals."""

    def __init__(self, num_clients: int, seed: int):
        self.num_clients = num_clients
        self.residuals = np.zeros(num_clients, dtype=np.float64)
        self.priors = np.full(num_clients, 1.0 / num_clients, dtype=np.float64)
        self.temperature = 1.0
        self.seed = seed

    @staticmethod
    def _temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
        logits = np.log(np.clip(probabilities, 1e-9, 1.0)) / temperature
        logits -= logits.max(axis=1, keepdims=True)
        output = np.exp(logits)
        output /= output.sum(axis=1, keepdims=True)
        return output

    @staticmethod
    def _targets(
        records: Sequence[QueryRecord], federation: Federation
    ) -> np.ndarray:
        target = np.zeros((len(records), federation.num_clients), dtype=np.int8)
        for row, record in enumerate(records):
            target[row, list(federation.gold_clients(record))] = 1
        return target

    def fit(
        self,
        records: Sequence[QueryRecord],
        federation: Federation,
        supervised: np.ndarray,
        sketch: np.ndarray,
    ) -> None:
        target = self._targets(records, federation)
        self.priors = np.clip(target.mean(axis=0), 1e-4, 1.0)
        # The input is the dev-selected router distribution. Temperature scaling
        # is monotonic, so global calibration cannot erase cold-start sketch order.
        best = (float("inf"), 1.0, supervised)
        for temperature in np.linspace(0.35, 3.0, 54):
            scaled = self._temperature_scale(supervised, float(temperature))
            nll = -float(
                np.sum(target * np.log(np.clip(scaled, 1e-9, 1.0)))
                / max(1, target.sum())
            )
            if nll < best[0]:
                best = (nll, float(temperature), scaled)
        self.temperature = best[1]
        global_prob = best[2]

        # Beta-smoothed client correction. Strong shrinkage retains personalization
        # without letting a small dev split overwrite evidence ranking.
        for client in range(self.num_clients):
            observed = float(target[:, client].sum())
            expected = float(global_prob[:, client].sum())
            total = float(len(records))
            obs_rate = (observed + 2.0) / (total + 4.0)
            exp_rate = (expected + 2.0) / (total + 4.0)
            correction = math.log(obs_rate / (1 - obs_rate)) - math.log(
                exp_rate / (1 - exp_rate)
            )
            shrinkage = observed / (observed + 50.0)
            self.residuals[client] = float(
                np.clip(correction * shrinkage, -0.25, 0.25)
            )

    def predict(self, supervised: np.ndarray, sketch: np.ndarray) -> np.ndarray:
        logits = np.log(np.clip(supervised, 1e-9, 1.0)) / self.temperature
        logits += self.residuals[None, :]
        logits -= logits.max(axis=1, keepdims=True)
        calibrated = np.exp(logits)
        calibrated /= calibrated.sum(axis=1, keepdims=True)
        return calibrated

    def predict_global(self, probabilities: np.ndarray) -> np.ndarray:
        """Rank-preserving global certificate used by the routing negotiator."""
        return self._temperature_scale(probabilities, self.temperature)


@dataclass(frozen=True)
class NegotiationPolicy:
    max_calls: int
    accept_threshold: float
    stop_mass: float
    offline_mass_threshold: float


class MissingnessAwareNegotiator:
    """Cost-normalized sequential client selection with explicit abstention."""

    def __init__(self, policy: NegotiationPolicy):
        self.policy = policy

    def select(
        self,
        probabilities: np.ndarray,
        online: np.ndarray | None = None,
        costs: np.ndarray | None = None,
    ) -> tuple[list[int], bool, dict]:
        num_clients = len(probabilities)
        online = (
            np.ones(num_clients, dtype=bool)
            if online is None
            else np.asarray(online, dtype=bool)
        )
        costs = (
            np.ones(num_clients, dtype=np.float64)
            if costs is None
            else np.asarray(costs, dtype=np.float64)
        )
        offline_mass = float(probabilities[~online].sum())
        offline_peak = (
            float(probabilities[~online].max()) if (~online).any() else 0.0
        )
        online_peak = float(probabilities[online].max()) if online.any() else 0.0
        critical_offline = (
            offline_peak >= self.policy.offline_mass_threshold
            and offline_peak > 1.05 * online_peak
        )
        if not online.any() or critical_offline:
            return [], True, {
                "offline_probability_mass": offline_mass,
                "offline_probability_peak": offline_peak,
                "stop_reason": "critical_capability_offline",
            }

        utility = np.where(online, probabilities / np.maximum(costs, 1e-8), -np.inf)
        order = np.argsort(-utility)
        selected: list[int] = []
        selected_mass = 0.0
        for client in order:
            if not online[client]:
                continue
            selected.append(int(client))
            selected_mass += float(probabilities[client])
            if (
                probabilities[selected[0]] >= self.policy.accept_threshold
                and selected_mass >= self.policy.stop_mass
            ):
                break
            if len(selected) >= self.policy.max_calls:
                break

        abstain = (
            not selected
            or probabilities[selected[0]] < self.policy.accept_threshold
        )
        return selected, abstain, {
            "offline_probability_mass": offline_mass,
            "offline_probability_peak": offline_peak,
            "selected_probability_mass": selected_mass,
            "stop_reason": "low_confidence" if abstain else "sufficient_mass",
        }


def tune_policy(
    probabilities: np.ndarray,
    records: Sequence[QueryRecord],
    federation: Federation,
) -> NegotiationPolicy:
    best: tuple[float, NegotiationPolicy] | None = None
    for max_calls in (1, 2, 3):
        for accept, mass, offline_threshold in product(
            # Both FinRAGBench-V and VisDoM rows used here are labeled answerable.
            # Semantic rejection cannot be tuned honestly until the Resolve track
            # supplies unanswerable examples; only missingness can trigger abstain.
            (0.0,),
            (0.25, 0.45, 0.65),
            (0.08, 0.16, 0.30),
        ):
            policy = NegotiationPolicy(
                max_calls=max_calls,
                accept_threshold=accept,
                stop_mass=mass,
                offline_mass_threshold=offline_threshold,
            )
            negotiator = MissingnessAwareNegotiator(policy)
            hits = 0
            calls = 0
            accepted = 0
            for row, record in enumerate(records):
                selected, abstain, _ = negotiator.select(probabilities[row])
                gold = set(federation.gold_clients(record))
                hits += int(bool(gold.intersection(selected)))
                calls += len(selected)
                accepted += int(not abstain)
            recall = hits / max(1, len(records))
            avg_calls = calls / max(1, len(records))
            coverage = accepted / max(1, len(records))
            stress = adaptive_metrics(
                probabilities,
                records,
                federation,
                policy,
                participation=0.5,
                seed=9173,
            )
            balanced_missingness = 0.5 * (
                stress["answerable_success"]
                + stress["unanswerable_abstain_accuracy"]
            )
            # Prefer normal-condition recall, communication efficiency, and
            # balanced answer/abstain behavior under a dev-only stress mask.
            objective = (
                recall
                - 0.035 * avg_calls
                + 0.12 * balanced_missingness
            )
            candidate = (objective, policy)
            if best is None or candidate[0] > best[0]:
                best = candidate
    assert best is not None
    return best[1]


def ranking_metrics(
    probabilities: np.ndarray,
    records: Sequence[QueryRecord],
    federation: Federation,
    ks: Sequence[int] = (1, 2, 3),
) -> dict:
    order = np.argsort(-probabilities, axis=1)
    metrics: dict[str, float] = {}
    reciprocal_ranks: list[float] = []
    for row, record in enumerate(records):
        gold = set(federation.gold_clients(record))
        rank = next(
            (idx + 1 for idx, client in enumerate(order[row]) if int(client) in gold),
            federation.num_clients + 1,
        )
        reciprocal_ranks.append(1.0 / rank)
    for k in ks:
        hits = []
        for row, record in enumerate(records):
            gold = set(federation.gold_clients(record))
            hits.append(bool(gold.intersection(map(int, order[row, :k]))))
        metrics[f"client_recall@{k}"] = float(np.mean(hits))
    metrics["mrr"] = float(np.mean(reciprocal_ranks))
    return metrics


def calibration_metrics(
    probabilities: np.ndarray,
    records: Sequence[QueryRecord],
    federation: Federation,
    bins: int = 10,
) -> dict:
    target = np.zeros_like(probabilities)
    for row, record in enumerate(records):
        gold = federation.gold_clients(record)
        target[row, list(gold)] = 1.0 / len(gold)
    nll = -float(
        np.sum(target * np.log(np.clip(probabilities, 1e-9, 1.0)))
        / max(1, len(records))
    )
    brier = float(np.mean(np.sum((probabilities - target) ** 2, axis=1)))
    prediction = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = np.array(
        [
            int(int(prediction[row]) in federation.gold_clients(record))
            for row, record in enumerate(records)
        ],
        dtype=np.float64,
    )
    ece = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidence >= lower) & (
            confidence <= upper if upper == 1.0 else confidence < upper
        )
        if mask.any():
            ece += float(mask.mean()) * abs(
                float(correct[mask].mean()) - float(confidence[mask].mean())
            )
    return {
        "nll": nll,
        "brier": brier,
        "ece_10bin": ece,
        "top1_accuracy": float(correct.mean()),
        "mean_confidence": float(confidence.mean()),
    }


def adaptive_metrics(
    probabilities: np.ndarray,
    records: Sequence[QueryRecord],
    federation: Federation,
    policy: NegotiationPolicy,
    participation: float = 1.0,
    seed: int = 0,
) -> dict:
    negotiator = MissingnessAwareNegotiator(policy)
    hit = 0
    calls = 0
    abstain = 0
    answerable = 0
    answerable_hit = 0
    unanswerable = 0
    unanswerable_abstain = 0
    for row, record in enumerate(records):
        if participation >= 1.0:
            online = np.ones(federation.num_clients, dtype=bool)
        else:
            rng = np.random.default_rng(stable_u64(record.uid, seed))
            online = rng.random(federation.num_clients) < participation
        selected, did_abstain, _ = negotiator.select(probabilities[row], online=online)
        gold = set(federation.gold_clients(record))
        is_answerable = all(online[g] for g in gold)
        is_hit = bool(gold.intersection(selected)) and not did_abstain
        hit += int(is_hit)
        calls += len(selected)
        abstain += int(did_abstain)
        if is_answerable:
            answerable += 1
            answerable_hit += int(is_hit)
        else:
            unanswerable += 1
            unanswerable_abstain += int(did_abstain)
    total = max(1, len(records))
    return {
        "participation": participation,
        "end_to_end_client_success": hit / total,
        "mean_client_calls": calls / total,
        "abstention_rate": abstain / total,
        "answerable_count": answerable,
        "answerable_success": answerable_hit / max(1, answerable),
        "unanswerable_count": unanswerable,
        "unanswerable_abstain_accuracy": unanswerable_abstain
        / max(1, unanswerable),
    }
