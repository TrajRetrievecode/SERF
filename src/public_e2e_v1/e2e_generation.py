#!/usr/bin/env python3
"""Shared frozen answer generator for the public end-to-end track.

One generator, one prompt template, one parser and one sampling
configuration are used by every method.  Requests are keyed by their content,
so two methods that deliver the same evidence package for the same question
consume one generation and are guaranteed to receive an identical answer.

The generator is also the only abstention mechanism available to every
method: any system may return INSUFFICIENT_EVIDENCE.  Methods that also carry
their own availability test are recorded separately so the two effects never
get conflated.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from e2e_common import JsonlAppender, sha256_bytes

INSUFFICIENT = "INSUFFICIENT_EVIDENCE"

# ---------------------------------------------------------------------------
# Prompt registry.
#
# v1 is the prompt of the reported runs.  v2 and v3 keep the identical output
# contract and parser and only change how the abstention instruction is
# worded.  The active version is chosen on training-split records only; see
# scripts/public_e2e_v1/select_prompt.py.
# ---------------------------------------------------------------------------

SYSTEM_V1 = (
    "You answer questions about documents using only the evidence pages that "
    "are provided. Copy values from the evidence rather than recalling them. "
    "Answer in the same language as the question. Keep the answer as short as "
    "the question allows. If the evidence pages do not contain what the "
    f"question asks for, make the answer exactly {INSUFFICIENT}."
)

SYSTEM_V2 = (
    "You answer questions about documents using only the evidence pages that "
    "are provided. Copy values from the evidence rather than recalling them. "
    "Answer in the same language as the question. Keep the answer as short as "
    "the question allows.\n"
    "Answer whenever the pages let you: derive the value if it needs a "
    "calculation, pick the right row or column of a table, and answer from a "
    "partially garbled or partially extracted page as long as the needed part "
    "is readable. A page that is hard to read is not a reason to refuse.\n"
    f"Use exactly {INSUFFICIENT} only when the pages are about something else "
    "entirely, or when the specific quantity the question asks for appears "
    "nowhere on them."
)

SYSTEM_V3 = (
    "You answer questions about documents using only the evidence pages that "
    "are provided. Copy values from the evidence rather than recalling them. "
    "Answer in the same language as the question. Keep the answer as short as "
    "the question allows.\n"
    "Your default is to answer. Compute the value if the question needs "
    "arithmetic, read it out of the table or chart if it is there, and give "
    "your best supported answer when the page is only partially extracted.\n"
    f"Use exactly {INSUFFICIENT} only if none of the pages is about the "
    "subject of the question. Do not use it because you are unsure, because "
    "the page is messy, or because the answer needs a calculation."
)

USER_TEMPLATE = """Question:
{query}

Evidence pages:
{evidence}

Reply with one JSON object and nothing else:
{{"answer": "<answer or {insufficient}>", "citations": ["P1", ...]}}"""

PROMPT_VERSIONS = {
    "v1": SYSTEM_V1,
    "v2": SYSTEM_V2,
    "v3": SYSTEM_V3,
}
DEFAULT_PROMPT_VERSION = "v1"

# Kept so older imports keep working; the runner always passes a version.
SYSTEM_PROMPT = SYSTEM_V1

EMPTY_EVIDENCE_NOTE = "(no evidence page was delivered)"


@dataclass(frozen=True)
class GenerationRequest:
    key: str
    prompt: str
    evidence_labels: tuple[str, ...]  # label -> page uid, index aligned
    evidence_page_uids: tuple[str, ...]


def build_prompt(
    query: str,
    evidence_page_uids: tuple[str, ...],
    page_text: dict[str, str],
    text_char_limit: int,
) -> tuple[str, tuple[str, ...]]:
    blocks = []
    labels = []
    for position, uid in enumerate(evidence_page_uids, start=1):
        label = f"P{position}"
        labels.append(label)
        text = (page_text.get(uid) or "").strip()
        text = re.sub(r"\n{3,}", "\n\n", text)[:text_char_limit]
        if not text:
            text = "(this page has no extractable text layer)"
        blocks.append(f"[{label}]\n{text}")
    evidence = "\n\n".join(blocks) if blocks else EMPTY_EVIDENCE_NOTE
    return (
        USER_TEMPLATE.format(
            query=query.strip(), evidence=evidence, insufficient=INSUFFICIENT
        ),
        tuple(labels),
    )


def request_for(
    query: str,
    evidence_page_uids: tuple[str, ...],
    page_text: dict[str, str],
    text_char_limit: int,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> GenerationRequest:
    if prompt_version not in PROMPT_VERSIONS:
        raise ValueError(f"unknown prompt version: {prompt_version}")
    prompt, labels = build_prompt(
        query, evidence_page_uids, page_text, text_char_limit
    )
    # The system prompt is part of the request identity, so two prompt
    # versions never share a cache entry.
    key = sha256_bytes(
        (prompt_version + "\0" + PROMPT_VERSIONS[prompt_version] + "\0" + prompt).encode(
            "utf-8"
        )
    )
    return GenerationRequest(
        key=key,
        prompt=prompt,
        evidence_labels=labels,
        evidence_page_uids=tuple(evidence_page_uids),
    )


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_output(text: str, labels: tuple[str, ...]) -> dict:
    """Extract the answer and citation labels, tolerating imperfect JSON."""
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw).strip()
    answer = None
    citations: list[str] = []
    parse_status = "json"
    match = _JSON_RE.search(raw)
    if match:
        try:
            payload = json.loads(match.group(0))
            if isinstance(payload, dict):
                answer = payload.get("answer")
                raw_citations = payload.get("citations")
                if isinstance(raw_citations, list):
                    citations = [str(c).strip() for c in raw_citations]
                elif isinstance(raw_citations, str):
                    citations = [raw_citations.strip()]
        except json.JSONDecodeError:
            parse_status = "json_decode_failed"
    else:
        parse_status = "no_json_object"
    if answer is None:
        parse_status = "fallback_raw_text" if parse_status == "json" else parse_status
        answer = raw
    answer = str(answer).strip()
    valid = [c for c in citations if c in labels]
    return {
        "answer": answer,
        "citations": valid,
        "citations_dropped": [c for c in citations if c not in labels],
        "parse_status": parse_status,
        "generator_abstained": answer.strip().upper().startswith(INSUFFICIENT),
    }


class GenerationCache:
    """Append-only content-addressed cache of generator outputs."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.entries: dict[str, str] = {}
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    self.entries[row["key"]] = row["output"]
        self._writer = JsonlAppender(self.path)

    def __contains__(self, key: str) -> bool:
        return key in self.entries

    def get(self, key: str) -> str | None:
        return self.entries.get(key)

    def put(self, key: str, output: str) -> None:
        self.entries[key] = output
        self._writer.write({"key": key, "output": output})

    def close(self) -> None:
        self._writer.close()


class VllmGenerator:
    """Thin wrapper so the sampling configuration is stated exactly once."""

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        max_model_len: int,
        seed: int,
        max_tokens: int,
        swap_space_gb: int = 2,
        enforce_eager: bool = True,
    ):
        from vllm import LLM, SamplingParams

        self.model_path = model_path
        self.max_tokens = max_tokens
        self.seed = seed
        self.llm = LLM(
            model=model_path,
            tokenizer=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype="float16",
            seed=seed,
            swap_space=swap_space_gb,
            enforce_eager=enforce_eager,
            trust_remote_code=False,
            disable_log_stats=True,
        )
        self.sampling = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=max_tokens,
            seed=seed,
            n=1,
        )
        self.tokenizer = self.llm.get_tokenizer()

    def chat_prompt(
        self, user_prompt: str, prompt_version: str = DEFAULT_PROMPT_VERSION
    ) -> str:
        return self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": PROMPT_VERSIONS[prompt_version]},
                {"role": "user", "content": user_prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

    def generate(
        self, prompts: list[str], prompt_version: str = DEFAULT_PROMPT_VERSION
    ) -> list[str]:
        rendered = [self.chat_prompt(p, prompt_version) for p in prompts]
        outputs = self.llm.generate(rendered, self.sampling)
        return [output.outputs[0].text for output in outputs]

    def configuration(self, prompt_version: str = DEFAULT_PROMPT_VERSION) -> dict:
        return {
            "model_path": self.model_path,
            "model_copied_into_project": False,
            "dtype": "float16",
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
            "prompt_version": prompt_version,
            "system_prompt": PROMPT_VERSIONS[prompt_version],
            "system_prompt_sha256": sha256_bytes(
                PROMPT_VERSIONS[prompt_version].encode("utf-8")
            ),
            "user_template_sha256": sha256_bytes(USER_TEMPLATE.encode("utf-8")),
            "shared_across_methods": True,
        }
