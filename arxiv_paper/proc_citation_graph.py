#!/usr/bin/env python3
"""Convert arXiv citation graphs into deterministic research-proposal SFT data.

The input graph format is produced by ``build_arxiv_citation_graph.py``:
one topic directory per graph, each containing ``nodes.jsonl``,
``edges.jsonl``, and ``manifest.json``. Edges are oriented from an older cited
paper to a later citing paper, so they can be read as parent -> child research
steps.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import hashlib
import html
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, TextIO


DEFAULT_INPUT_ROOT = Path("data/arxiv_citation_graph_premier_tcs_depth5_s2")
DEFAULT_OUTPUT_DIR = Path("data/arxiv_research_midtraining_premier_tcs_depth5_s2_v2_markdown_gemini")

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "for",
    "from",
    "has",
    "have",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "our",
    "that",
    "the",
    "their",
    "this",
    "to",
    "using",
    "via",
    "we",
    "with",
}

QUANTITY_WORDS = {
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
    "hundred",
    "thousand",
    "million",
    "billion",
    "trillion",
    "first",
    "second",
    "third",
    "fourth",
    "fifth",
    "sixth",
    "seventh",
    "eighth",
    "ninth",
    "tenth",
}

CONTRIBUTION_CUES = (
    "we show",
    "we prove",
    "we present",
    "we propose",
    "we introduce",
    "we give",
    "we study",
    "this paper",
    "in this paper",
    "our main",
)


JsonDict = dict[str, Any]
AbstractProcessor = Callable[[dict[str, JsonDict], "ConversionConfig"], dict[str, JsonDict]]


@dataclass(frozen=True)
class ConversionConfig:
    input_root: Path = DEFAULT_INPUT_ROOT
    output_dir: Path = DEFAULT_OUTPUT_DIR
    seed: int = 0
    topic_limit: int = 0
    topics: tuple[str, ...] = ()
    max_examples: int = 0
    max_single_examples_per_topic: int = 0
    max_trajectories_per_topic: int = 100
    min_parents: int = 2
    max_parents: int = 5
    min_trajectory_turns: int = 3
    max_trajectory_turns: int = 5
    valid_fraction: float = 0.01
    test_fraction: float = 0.01
    max_parent_abstract_chars: int = 1600
    max_child_abstract_chars: int = 2200
    max_insight_chars: int = 360
    max_user_chars: int = 24000
    write_sft_parquet: bool = True
    write_sft_jsonl: bool = True
    sft_output_dir: str = "sft"
    include_trajectories: bool = True
    abstract_processing_mode: str = "deterministic"
    abstract_model: str = "gemini-3.1-flash-lite-preview"
    abstract_batch_size: int = 8
    abstract_batch_max_chars: int = 60000
    abstract_max_concurrent: int = 10
    abstract_max_completion_tokens: int = 8192
    abstract_cache_jsonl: str = "abstract_processing_cache.jsonl"
    paper_source_max_chars: int = 80000
    progress_interval: float = 15.0
    no_progress: bool = False
    abstract_processor: AbstractProcessor | None = field(default=None, compare=False, repr=False)


@dataclass
class TopicGraph:
    name: str
    path: Path
    manifest: JsonDict
    nodes: dict[str, JsonDict]
    incoming: dict[str, list[str]]
    outgoing: dict[str, list[str]]
    edge_records: dict[tuple[str, str], JsonDict]


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


class ProgressReporter:
    def __init__(
        self,
        *,
        enabled: bool,
        interval_seconds: float,
        stream: TextIO = sys.stderr,
    ) -> None:
        self.enabled = enabled
        self.interval_seconds = interval_seconds
        self.stream = stream
        self.stage_name = ""
        self.stage_unit = "items"
        self.stage_started_at = time.monotonic()
        self.last_report_at = 0.0

    def log(self, message: str) -> None:
        if not self.enabled:
            return
        timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}", file=self.stream, flush=True)

    def start_stage(
        self,
        name: str,
        *,
        total: int | None = None,
        unit: str = "items",
        detail: str | None = None,
    ) -> None:
        self.stage_name = name
        self.stage_unit = unit
        self.stage_started_at = time.monotonic()
        self.last_report_at = 0.0

        parts = [f"{name}: started"]
        if total is not None:
            parts.append(f"target {total:,} {unit}")
        if detail:
            parts.append(detail)
        self.log(" | ".join(parts))

    def update(
        self,
        completed: int,
        *,
        total: int | None = None,
        detail: str | None = None,
        force: bool = False,
    ) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        if (
            not force
            and self.last_report_at
            and now - self.last_report_at < self.interval_seconds
        ):
            return

        elapsed = now - self.stage_started_at
        rate = completed / elapsed if elapsed > 0 else 0.0
        parts = [self.stage_name or "Progress"]
        if total is not None:
            percent = (completed / total * 100.0) if total else 100.0
            parts.append(f"{completed:,}/{total:,} {self.stage_unit} ({percent:.1f}%)")
            if rate > 0 and completed < total:
                parts.append(f"eta {format_duration((total - completed) / rate)}")
        else:
            parts.append(f"{completed:,} {self.stage_unit}")
        parts.append(f"elapsed {format_duration(elapsed)}")
        if rate > 0:
            parts.append(f"{rate:.2f} {self.stage_unit}/s")
        if detail:
            parts.append(detail)
        self.log(" | ".join(parts))
        self.last_report_at = now

    def finish_stage(
        self,
        *,
        completed: int | None = None,
        total: int | None = None,
        detail: str | None = None,
    ) -> None:
        elapsed = time.monotonic() - self.stage_started_at
        parts = [
            f"{self.stage_name or 'Progress'}: finished",
            f"elapsed {format_duration(elapsed)}",
        ]
        if completed is not None and total is not None:
            parts.append(f"{completed:,}/{total:,} {self.stage_unit}")
        elif completed is not None:
            parts.append(f"{completed:,} {self.stage_unit}")
        if detail:
            parts.append(detail)
        self.log(" | ".join(parts))


def make_progress_reporter(config: ConversionConfig) -> ProgressReporter:
    return ProgressReporter(
        enabled=not config.no_progress,
        interval_seconds=config.progress_interval,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build deterministic citation-graph mid-training JSONL and VERL SFT Parquet files."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=15.0,
        help="Minimum seconds between non-forced progress updates.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable timestamped progress updates on stderr.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--topic-limit", type=int, default=0)
    parser.add_argument("--topics", nargs="*", default=[])
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--max-single-examples-per-topic", type=int, default=0)
    parser.add_argument("--max-trajectories-per-topic", type=int, default=100)
    parser.add_argument("--min-parents", type=int, default=2)
    parser.add_argument("--max-parents", type=int, default=5)
    parser.add_argument("--min-trajectory-turns", type=int, default=3)
    parser.add_argument("--max-trajectory-turns", type=int, default=5)
    parser.add_argument("--valid-fraction", type=float, default=0.01)
    parser.add_argument("--test-fraction", type=float, default=0.01)
    parser.add_argument("--max-parent-abstract-chars", type=int, default=1600)
    parser.add_argument("--max-child-abstract-chars", type=int, default=2200)
    parser.add_argument("--max-insight-chars", type=int, default=360)
    parser.add_argument("--max-user-chars", type=int, default=24000)
    parser.add_argument(
        "--write-sft-parquet",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write sft/{train,valid,test}.parquet with a messages column.",
    )
    parser.add_argument(
        "--write-sft-jsonl",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write inspectable sft/{train,valid,test}.jsonl mirrors.",
    )
    parser.add_argument("--sft-output-dir", default="sft")
    parser.add_argument(
        "--include-trajectories",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--abstract-processing-mode",
        choices=("none", "deterministic", "gemini"),
        default="deterministic",
        help="How to rewrite abstracts before emitting training data.",
    )
    parser.add_argument("--abstract-model", default="gemini-3.1-flash-lite-preview")
    parser.add_argument("--abstract-batch-size", type=int, default=8)
    parser.add_argument("--abstract-batch-max-chars", type=int, default=60000)
    parser.add_argument("--abstract-max-concurrent", type=int, default=10)
    parser.add_argument("--abstract-max-completion-tokens", type=int, default=8192)
    parser.add_argument("--abstract-cache-jsonl", default="abstract_processing_cache.jsonl")
    parser.add_argument(
        "--paper-source-max-chars",
        type=int,
        default=80000,
        help="Maximum characters of full-paper HTML/text to send for one summary.",
    )
    args = parser.parse_args()

    if args.topic_limit < 0:
        parser.error("--topic-limit must be >= 0")
    if args.max_examples < 0:
        parser.error("--max-examples must be >= 0")
    if args.max_single_examples_per_topic < 0:
        parser.error("--max-single-examples-per-topic must be >= 0")
    if args.max_trajectories_per_topic < 0:
        parser.error("--max-trajectories-per-topic must be >= 0")
    if args.min_parents < 1:
        parser.error("--min-parents must be >= 1")
    if args.max_parents < args.min_parents:
        parser.error("--max-parents must be >= --min-parents")
    if args.min_trajectory_turns < 1:
        parser.error("--min-trajectory-turns must be >= 1")
    if args.max_trajectory_turns < args.min_trajectory_turns:
        parser.error("--max-trajectory-turns must be >= --min-trajectory-turns")
    if args.valid_fraction < 0 or args.test_fraction < 0:
        parser.error("split fractions must be non-negative")
    if args.valid_fraction + args.test_fraction >= 1:
        parser.error("--valid-fraction + --test-fraction must be < 1")
    if args.max_parent_abstract_chars <= 0 or args.max_child_abstract_chars <= 0:
        parser.error("abstract character limits must be positive")
    if args.max_user_chars <= 0:
        parser.error("--max-user-chars must be positive")
    if args.abstract_batch_size <= 0:
        parser.error("--abstract-batch-size must be positive")
    if args.abstract_batch_max_chars <= 0:
        parser.error("--abstract-batch-max-chars must be positive")
    if args.abstract_max_concurrent <= 0:
        parser.error("--abstract-max-concurrent must be positive")
    if args.abstract_max_completion_tokens <= 0:
        parser.error("--abstract-max-completion-tokens must be positive")
    if args.paper_source_max_chars <= 0:
        parser.error("--paper-source-max-chars must be positive")
    if args.progress_interval < 0:
        parser.error("--progress-interval must be non-negative")
    return args


def config_from_args(args: argparse.Namespace) -> ConversionConfig:
    return ConversionConfig(
        input_root=args.input_root,
        output_dir=args.output_dir,
        seed=args.seed,
        topic_limit=args.topic_limit,
        topics=tuple(args.topics or ()),
        max_examples=args.max_examples,
        max_single_examples_per_topic=args.max_single_examples_per_topic,
        max_trajectories_per_topic=args.max_trajectories_per_topic,
        min_parents=args.min_parents,
        max_parents=args.max_parents,
        min_trajectory_turns=args.min_trajectory_turns,
        max_trajectory_turns=args.max_trajectory_turns,
        valid_fraction=args.valid_fraction,
        test_fraction=args.test_fraction,
        max_parent_abstract_chars=args.max_parent_abstract_chars,
        max_child_abstract_chars=args.max_child_abstract_chars,
        max_insight_chars=args.max_insight_chars,
        max_user_chars=args.max_user_chars,
        write_sft_parquet=args.write_sft_parquet,
        write_sft_jsonl=args.write_sft_jsonl,
        sft_output_dir=args.sft_output_dir,
        include_trajectories=args.include_trajectories,
        abstract_processing_mode=args.abstract_processing_mode,
        abstract_model=args.abstract_model,
        abstract_batch_size=args.abstract_batch_size,
        abstract_batch_max_chars=args.abstract_batch_max_chars,
        abstract_max_concurrent=args.abstract_max_concurrent,
        abstract_max_completion_tokens=args.abstract_max_completion_tokens,
        abstract_cache_jsonl=args.abstract_cache_jsonl,
        paper_source_max_chars=args.paper_source_max_chars,
        progress_interval=args.progress_interval,
        no_progress=args.no_progress,
    )


def stable_digest(value: str, *, length: int = 16) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def compact_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def truncate_text(value: object, max_chars: int) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) <= max_chars:
        return text
    if max_chars <= 16:
        return text[:max_chars]
    cutoff = max_chars - 14
    trimmed = text[:cutoff]
    space_index = trimmed.rfind(" ")
    if space_index >= max_chars // 2:
        trimmed = trimmed[:space_index]
    return trimmed.rstrip() + " ... [truncated]"


def tokenize(value: object) -> set[str]:
    text = compact_text(value).lower()
    return {
        token
        for token in re.findall(r"[a-z0-9][a-z0-9-]*", text)
        if len(token) > 2 and token not in STOPWORDS
    }


def as_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def read_json(path: Path) -> JsonDict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def iter_jsonl(path: Path) -> Iterable[JsonDict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_num} is not a JSON object")
            yield payload


def write_jsonl(path: Path, rows: Iterable[JsonDict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def contains_digit(text: object) -> bool:
    return any(char.isdigit() for char in str(text or ""))


def token_has_quantity_word(token: str) -> bool:
    parts = [part for part in re.split(r"[^A-Za-z]+", token.lower()) if part]
    return any(part in QUANTITY_WORDS for part in parts)


def numeric_fragments(text: object) -> list[str]:
    text = compact_text(text)
    if not text:
        return []
    fragments = re.findall(r"\b\S*\d\S*\b", text)
    fragments.extend(
        token
        for token in re.findall(r"\b[A-Za-z]+(?:-[A-Za-z]+)*\b", text)
        if token_has_quantity_word(token)
    )
    seen: set[str] = set()
    unique = []
    for fragment in fragments:
        cleaned = fragment.strip(".,;:()[]{}")
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            unique.append(cleaned)
    return unique


def strip_html_text(text: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", text)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    text = html.unescape(text)
    return compact_text(text)


def resolve_source_path(node: JsonDict) -> Path | None:
    graph_dir = Path(str(node.get("_graph_dir") or "")) if node.get("_graph_dir") else None
    for key in ("arxiv_html_path", "paper_text_path", "text_path", "markdown_path"):
        value = node.get(key)
        if not value:
            continue
        path = Path(str(value))
        if not path.is_absolute() and graph_dir is not None:
            path = graph_dir / path
        if path.exists() and path.is_file():
            return path
    return None


def read_paper_source_text(node: JsonDict, config: ConversionConfig) -> tuple[str, str, str | None]:
    path = resolve_source_path(node)
    if path is None:
        return compact_text(node.get("abstract")), "abstract", None
    text = path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix.lower() in {".html", ".htm"}:
        text = strip_html_text(text)
    else:
        text = compact_text(text)
    return truncate_text(text, config.paper_source_max_chars), "full_text", str(path)


def processing_source(node: JsonDict, config: ConversionConfig) -> tuple[str, str, str | None]:
    if "_processing_source_text" in node:
        return (
            str(node.get("_processing_source_text") or ""),
            str(node.get("_processing_source_kind") or "abstract"),
            str(node.get("_processing_source_path") or "") or None,
        )
    return read_paper_source_text(node, config)


def deterministic_process_abstract(abstract: object) -> str:
    text = compact_text(abstract)
    if not text:
        return ""
    tokens = []
    for token in text.split(" "):
        cleaned = token.strip(".,;:!?()[]{}\"'")
        if contains_digit(cleaned) or token_has_quantity_word(cleaned):
            continue
        tokens.append(token)
    text = " ".join(tokens)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"\(\s*\)", "", text)
    text = re.sub(r"\[\s*\]", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip(" ,;:-")


def abstract_cache_path(config: ConversionConfig) -> Path:
    path = Path(config.abstract_cache_jsonl)
    if path.is_absolute():
        return path
    return config.output_dir / path


def abstract_cache_key(node_id: str, abstract: object) -> str:
    raw_hash = hashlib.sha1(compact_text(abstract).encode("utf-8")).hexdigest()
    return stable_digest(f"{node_id}|{raw_hash}", length=40)


def load_abstract_cache(path: Path) -> dict[str, JsonDict]:
    if not path.exists():
        return {}
    rows: dict[str, JsonDict] = {}
    for row in iter_jsonl(path):
        key = str(row.get("cache_key") or "")
        if key:
            rows[key] = row
    return rows


def save_abstract_cache(
    path: Path,
    rows_by_key: dict[str, JsonDict],
    progress: ProgressReporter | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    total = len(rows_by_key)
    if progress:
        progress.start_stage(
            "Saving abstract cache",
            total=total,
            unit="records",
            detail=str(path),
        )
    with tmp_path.open("w", encoding="utf-8") as handle:
        for index, key in enumerate(sorted(rows_by_key), start=1):
            handle.write(json.dumps(rows_by_key[key], ensure_ascii=False, sort_keys=True) + "\n")
            if progress:
                progress.update(index, total=total)
    tmp_path.replace(path)
    if progress:
        progress.finish_stage(completed=total, total=total)


def validate_processed_abstract(raw_abstract: object, processed: object) -> tuple[bool, str]:
    processed_text = compact_text(processed)
    raw_text = compact_text(raw_abstract)
    if not processed_text:
        return False, "empty processed abstract"
    if contains_digit(processed_text):
        return False, "processed abstract still contains digits"
    lower = processed_text.lower()
    boilerplate_prefixes = (
        "here is",
        "here's",
        "processed abstract",
        "the processed abstract",
        "rewritten abstract",
    )
    if any(lower.startswith(prefix) for prefix in boilerplate_prefixes):
        return False, "processed abstract contains response boilerplate"
    max_length = max(300, int(len(raw_text) * 1.35) + 120)
    if len(processed_text) > max_length:
        return False, "processed abstract is too long"
    return True, ""


def build_abstract_record(
    node_id: str,
    source_text: object,
    processed_abstract: object,
    *,
    mode: str,
    model: str | None = None,
    source_kind: str = "abstract",
    source_path: str | None = None,
    removed_quantities: list[str] | None = None,
    quality_flags: list[str] | None = None,
) -> JsonDict:
    processed_text = compact_text(processed_abstract)
    ok, reason = validate_processed_abstract(source_text, processed_text)
    if not ok:
        raise ValueError(f"Invalid processed abstract for {node_id}: {reason}")
    raw_text = compact_text(source_text)
    raw_hash = hashlib.sha1(raw_text.encode("utf-8")).hexdigest()
    return {
        "cache_key": abstract_cache_key(node_id, raw_text),
        "node_id": node_id,
        "source_sha1": raw_hash,
        "raw_abstract_sha1": raw_hash,
        "source_kind": source_kind,
        "source_path": source_path,
        "processed_abstract": processed_text,
        "removed_quantities": list(removed_quantities or numeric_fragments(raw_text)),
        "quality_flags": list(quality_flags or [source_kind]),
        "mode": mode,
        "model": model,
    }


def deterministic_abstract_records(
    nodes_by_id: dict[str, JsonDict],
    config: ConversionConfig,
    progress: ProgressReporter | None = None,
) -> dict[str, JsonDict]:
    records = {}
    total = len(nodes_by_id)
    if progress:
        progress.start_stage("Deterministic abstract processing", total=total, unit="papers")
    for index, (node_id, node) in enumerate(nodes_by_id.items(), start=1):
        source_text, source_kind, source_path = processing_source(node, config)
        records[node_id] = build_abstract_record(
            node_id,
            source_text,
            deterministic_process_abstract(source_text),
            mode="deterministic",
            model=None,
            source_kind=source_kind,
            source_path=source_path,
        )
        if progress:
            progress.update(index, total=total)
    if progress:
        progress.finish_stage(completed=len(records), total=total)
    return records


def none_abstract_records(
    nodes_by_id: dict[str, JsonDict],
    config: ConversionConfig,
    progress: ProgressReporter | None = None,
) -> dict[str, JsonDict]:
    records = {}
    total = len(nodes_by_id)
    if progress:
        progress.start_stage("Raw abstract/source records", total=total, unit="papers")
    for index, (node_id, node) in enumerate(nodes_by_id.items(), start=1):
        raw, source_kind, source_path = processing_source(node, config)
        records[node_id] = {
            "cache_key": abstract_cache_key(node_id, raw),
            "node_id": node_id,
            "source_sha1": hashlib.sha1(raw.encode("utf-8")).hexdigest(),
            "raw_abstract_sha1": hashlib.sha1(raw.encode("utf-8")).hexdigest(),
            "source_kind": source_kind,
            "source_path": source_path,
            "processed_abstract": raw,
            "removed_quantities": [],
            "quality_flags": [f"raw_{source_kind}"],
            "mode": "none",
            "model": None,
        }
        if progress:
            progress.update(index, total=total)
    if progress:
        progress.finish_stage(completed=len(records), total=total)
    return records


def build_abstract_processing_prompt(items: list[JsonDict]) -> str:
    payload = [
        {
            "node_id": item["node_id"],
            "title": item["title"],
            "source_kind": item["source_kind"],
            "document": item["document"],
        }
        for item in items
    ]
    return (
        "Summarize academic papers for a high-quality supervised fine-tuning dataset.\n\n"
        "Rules:\n"
        "- Summarize the document, clearly describing the method used and highlighting the key insights or findings.\n"
        "- Provide sufficient detail so that the approach and main contributions are fully understood.\n"
        "- If the source_kind is abstract, summarize only the abstract because no full paper text is available.\n"
        "- Remove exact quantitative values, counts, thresholds, years, percentages, dimensions, sample sizes, and numbered claims.\n"
        "- Remove exact quantities whether they are written with digits, symbols, or words.\n"
        "- The processed_abstract must contain no digit characters at all and should not spell out exact quantities as number words.\n"
        "- Do not add facts that are not in the original source document.\n"
        "- Do not include commentary, markdown, or prose outside JSON.\n\n"
        "Return strict JSON with this shape:\n"
        "{\"items\":[{\"node_id\":\"...\",\"processed_abstract\":\"...\",\"removed_quantities\":[\"...\"],\"quality_flags\":[]}]}\n\n"
        "Input documents:\n"
        + json.dumps({"items": payload}, ensure_ascii=False)
    )


def extract_json_payload(text: str) -> JsonDict:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end < start:
        raise ValueError("No JSON object found")
    payload = json.loads(stripped[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("JSON response is not an object")
    return payload


def parse_abstract_processing_response(
    response_text: str,
    batch: list[JsonDict],
    *,
    mode: str,
    model: str,
) -> dict[str, JsonDict]:
    payload = extract_json_payload(response_text)
    raw_items = payload.get("items")
    if not isinstance(raw_items, list):
        raise ValueError("JSON response missing items list")
    by_id: dict[str, JsonDict] = {}
    batch_by_id = {str(item["node_id"]): item for item in batch}
    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            raise ValueError("JSON item is not an object")
        node_id = str(raw_item.get("node_id") or "")
        if node_id not in batch_by_id:
            continue
        processed = raw_item.get("processed_abstract")
        removed = raw_item.get("removed_quantities")
        flags = raw_item.get("quality_flags")
        by_id[node_id] = build_abstract_record(
            node_id,
            batch_by_id[node_id]["document"],
            processed,
            mode=mode,
            model=model,
            source_kind=str(batch_by_id[node_id].get("source_kind") or "abstract"),
            source_path=batch_by_id[node_id].get("source_path"),
            removed_quantities=removed if isinstance(removed, list) else None,
            quality_flags=flags if isinstance(flags, list) else None,
        )
    missing = sorted(set(batch_by_id) - set(by_id))
    if missing:
        raise ValueError(f"JSON response missing node_ids: {missing[:5]}")
    return by_id


def build_repair_prompt(original_prompt: str, response_text: str, error: str) -> str:
    return (
        "Repair the previous JSON response for abstract processing.\n\n"
        f"Validation error: {error}\n\n"
        "Return only strict JSON with the required shape. Every processed_abstract must contain no digit characters "
        "and should not spell out exact quantities as number words.\n\n"
        "Original request:\n"
        f"{original_prompt}\n\n"
        "Invalid response:\n"
        f"{response_text}"
    )


async def call_gemini_for_abstract_batch(
    client: Any,
    batch: list[JsonDict],
    config: ConversionConfig,
) -> dict[str, JsonDict]:
    from inference.collect_llm import generate_response

    prompt = build_abstract_processing_prompt(batch)
    responses = await generate_response(
        client,
        prompt,
        N=1,
        max_completion_tokens=config.abstract_max_completion_tokens,
        use_cache=True,
        model_name=config.abstract_model,
        provider="gemini",
    )
    response_text = responses[0] if responses else ""
    try:
        return parse_abstract_processing_response(
            response_text,
            batch,
            mode="gemini",
            model=config.abstract_model,
        )
    except Exception as first_error:
        repair_prompt = build_repair_prompt(prompt, response_text, str(first_error))
        repair_responses = await generate_response(
            client,
            repair_prompt,
            N=1,
            max_completion_tokens=config.abstract_max_completion_tokens,
            use_cache=True,
            model_name=config.abstract_model,
            provider="gemini",
        )
        repair_text = repair_responses[0] if repair_responses else ""
        try:
            return parse_abstract_processing_response(
                repair_text,
                batch,
                mode="gemini",
                model=config.abstract_model,
            )
        except Exception:
            if len(batch) <= 1:
                raise
            repaired: dict[str, JsonDict] = {}
            for item in batch:
                repaired.update(await call_gemini_for_abstract_batch(client, [item], config))
            return repaired


def batch_abstract_processing_items(items: list[JsonDict], config: ConversionConfig) -> list[list[JsonDict]]:
    batches: list[list[JsonDict]] = []
    current: list[JsonDict] = []
    current_chars = 0
    for item in items:
        item_chars = len(str(item.get("document") or "")) + len(str(item.get("title") or ""))
        if (
            current
            and (
                len(current) >= config.abstract_batch_size
                or current_chars + item_chars > config.abstract_batch_max_chars
            )
        ):
            batches.append(current)
            current = []
            current_chars = 0
        current.append(item)
        current_chars += item_chars
    if current:
        batches.append(current)
    return batches


async def gemini_abstract_records_async(
    nodes_by_id: dict[str, JsonDict],
    config: ConversionConfig,
    progress: ProgressReporter | None = None,
) -> dict[str, JsonDict]:
    from inference.collect_llm import create_async_client

    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY must be set for --abstract-processing-mode gemini")

    client = create_async_client("gemini")
    items = []
    for node_id, node in sorted(nodes_by_id.items()):
        source_text, source_kind, source_path = processing_source(node, config)
        items.append(
            {
                "node_id": node_id,
                "title": compact_text(node.get("title")),
                "source_kind": source_kind,
                "source_path": source_path,
                "document": source_text,
            }
        )
    batches = batch_abstract_processing_items(items, config)
    semaphore = asyncio.Semaphore(config.abstract_max_concurrent)

    async def process_batch(batch: list[JsonDict]) -> dict[str, JsonDict]:
        async with semaphore:
            return await call_gemini_for_abstract_batch(client, batch, config)

    records: dict[str, JsonDict] = {}
    if progress:
        progress.start_stage(
            "Gemini abstract processing",
            total=len(batches),
            unit="batches",
            detail=f"papers={len(items):,}, concurrency={config.abstract_max_concurrent:,}",
        )
    tasks = [asyncio.create_task(process_batch(batch)) for batch in batches]
    processed_papers = 0
    for index, task in enumerate(asyncio.as_completed(tasks), start=1):
        result = await task
        records.update(result)
        processed_papers += len(result)
        if progress:
            progress.update(
                index,
                total=len(batches),
                detail=f"papers={processed_papers:,}/{len(items):,}",
            )
    if progress:
        progress.finish_stage(
            completed=len(batches),
            total=len(batches),
            detail=f"papers={len(records):,}/{len(items):,}",
        )
    return records


def gemini_abstract_records(
    nodes_by_id: dict[str, JsonDict],
    config: ConversionConfig,
    progress: ProgressReporter | None = None,
) -> dict[str, JsonDict]:
    return asyncio.run(gemini_abstract_records_async(nodes_by_id, config, progress))


def process_abstract_records(
    nodes_by_id: dict[str, JsonDict],
    config: ConversionConfig,
    progress: ProgressReporter | None = None,
) -> tuple[dict[str, JsonDict], JsonDict]:
    if not nodes_by_id:
        return {}, {"mode": config.abstract_processing_mode, "processed_count": 0}

    if config.abstract_processor is not None:
        if progress:
            progress.start_stage(
                "Injected abstract processing",
                total=len(nodes_by_id),
                unit="papers",
                detail="processor=injected",
            )
        records = config.abstract_processor(nodes_by_id, config)
        if progress:
            progress.finish_stage(completed=len(records), total=len(nodes_by_id))
            progress.start_stage("Validating injected abstracts", total=len(records), unit="papers")
        for index, (node_id, record) in enumerate(records.items(), start=1):
            node = nodes_by_id.get(node_id)
            if node is None:
                continue
            source_text, _, _ = read_paper_source_text(node, config)
            ok, reason = validate_processed_abstract(source_text, record.get("processed_abstract"))
            if not ok:
                raise ValueError(f"Injected abstract processor returned invalid abstract for {node_id}: {reason}")
            if progress:
                progress.update(index, total=len(records))
        if progress:
            progress.finish_stage(completed=len(records), total=len(records))
        return records, {
            "mode": config.abstract_processing_mode,
            "processed_count": len(records),
            "processor": "injected",
        }

    if config.abstract_processing_mode == "none":
        return none_abstract_records(nodes_by_id, config, progress), {
            "mode": "none",
            "processed_count": len(nodes_by_id),
        }

    cache_path = abstract_cache_path(config)
    cache_rows = load_abstract_cache(cache_path)
    records: dict[str, JsonDict] = {}
    missing_nodes: dict[str, JsonDict] = {}
    total = len(nodes_by_id)
    if progress:
        progress.start_stage(
            "Abstract source/cache scan",
            total=total,
            unit="papers",
            detail=f"cache={cache_path}",
        )
    for index, (node_id, node) in enumerate(nodes_by_id.items(), start=1):
        source_text, source_kind, source_path = read_paper_source_text(node, config)
        key = abstract_cache_key(node_id, source_text)
        cached = cache_rows.get(key)
        if cached and cached.get("processed_abstract"):
            ok, _ = validate_processed_abstract(source_text, cached["processed_abstract"])
            if ok and cached.get("mode") == config.abstract_processing_mode:
                records[node_id] = cached
                if progress:
                    progress.update(
                        index,
                        total=total,
                        detail=f"hits={len(records):,}, misses={len(missing_nodes):,}",
                    )
                continue
        node_with_source = dict(node)
        node_with_source["_processing_source_text"] = source_text
        node_with_source["_processing_source_kind"] = source_kind
        if source_path:
            node_with_source["_processing_source_path"] = source_path
        missing_nodes[node_id] = node_with_source
        if progress:
            progress.update(
                index,
                total=total,
                detail=f"hits={len(records):,}, misses={len(missing_nodes):,}",
            )
    if progress:
        progress.finish_stage(
            completed=total,
            total=total,
            detail=f"hits={len(records):,}, misses={len(missing_nodes):,}",
        )

    if not missing_nodes:
        new_records = {}
    elif config.abstract_processing_mode == "deterministic":
        new_records = deterministic_abstract_records(missing_nodes, config, progress)
    elif config.abstract_processing_mode == "gemini":
        new_records = gemini_abstract_records(missing_nodes, config, progress)
    else:
        raise ValueError(f"Unknown abstract processing mode: {config.abstract_processing_mode}")

    records.update(new_records)
    for record in new_records.values():
        cache_rows[str(record["cache_key"])] = record
    save_abstract_cache(cache_path, cache_rows, progress)

    return records, {
        "mode": config.abstract_processing_mode,
        "model": config.abstract_model if config.abstract_processing_mode == "gemini" else None,
        "cache_jsonl": str(cache_path.resolve()),
        "processed_count": len(records),
        "cache_hit_count": len(records) - len(new_records),
        "cache_miss_count": len(new_records),
        "source_kind_counts": dict(Counter(str(record.get("source_kind") or "unknown") for record in records.values())),
    }


def discover_topic_dirs(input_root: Path, topics: tuple[str, ...], topic_limit: int) -> list[Path]:
    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    requested = set(topics)
    topic_dirs = [
        path
        for path in sorted(input_root.iterdir())
        if path.is_dir()
        and (not requested or path.name in requested)
        and (path / "nodes.jsonl").exists()
        and (path / "edges.jsonl").exists()
    ]
    if requested:
        found = {path.name for path in topic_dirs}
        missing = sorted(requested - found)
        if missing:
            raise FileNotFoundError(f"Requested topics are missing graph files: {missing}")
    if topic_limit:
        topic_dirs = topic_dirs[:topic_limit]
    if not topic_dirs:
        raise ValueError(f"No topic graph directories found under {input_root}")
    return topic_dirs


def load_topic_graph(topic_dir: Path) -> TopicGraph:
    nodes: dict[str, JsonDict] = {}
    for node in iter_jsonl(topic_dir / "nodes.jsonl"):
        node_id = str(node.get("node_id") or "")
        if not node_id:
            continue
        nodes[node_id] = node

    incoming: dict[str, list[str]] = defaultdict(list)
    outgoing: dict[str, list[str]] = defaultdict(list)
    edge_records: dict[tuple[str, str], JsonDict] = {}
    for edge in iter_jsonl(topic_dir / "edges.jsonl"):
        source = str(edge.get("source") or "")
        target = str(edge.get("target") or "")
        if not source or not target or source not in nodes or target not in nodes:
            continue
        incoming[target].append(source)
        outgoing[source].append(target)
        edge_records[(source, target)] = edge

    for values in incoming.values():
        values.sort()
    for values in outgoing.values():
        values.sort()

    return TopicGraph(
        name=topic_dir.name,
        path=topic_dir,
        manifest=read_json(topic_dir / "manifest.json"),
        nodes=nodes,
        incoming=dict(incoming),
        outgoing=dict(outgoing),
        edge_records=edge_records,
    )


def topic_tokens(graph: TopicGraph) -> set[str]:
    query = graph.manifest.get("query") or graph.manifest.get("s2_search_query") or graph.name
    return tokenize(f"{graph.name.replace('_', ' ')} {query}")


def node_text(node: JsonDict) -> str:
    return f"{node.get('title') or ''} {node.get('abstract') or ''}"


def has_title_abstract_year(node: JsonDict) -> bool:
    return bool(compact_text(node.get("title"))) and bool(compact_text(node.get("abstract"))) and as_int(node.get("year")) is not None


def is_valid_parent(parent: JsonDict, child: JsonDict) -> bool:
    if not compact_text(parent.get("title")) or not compact_text(parent.get("abstract")):
        return False
    parent_year = as_int(parent.get("year"))
    child_year = as_int(child.get("year"))
    if parent_year is not None and child_year is not None and parent_year > child_year:
        return False
    return True


def parent_score(
    parent_id: str,
    parent: JsonDict,
    child_tokens: set[str],
    topic_terms: set[str],
) -> tuple[int, int, int, int, int, str]:
    tokens = tokenize(node_text(parent))
    overlap = len(tokens & child_tokens)
    topic_overlap = len(tokens & topic_terms)
    has_arxiv = 1 if parent.get("arxiv_id") else 0
    citation_count = as_int(parent.get("citation_count")) or 0
    year = as_int(parent.get("year")) or 0
    return (topic_overlap, overlap, has_arxiv, citation_count, year, parent_id)


def select_parent_ids(
    child_id: str,
    graph: TopicGraph,
    terms: set[str],
    config: ConversionConfig,
    *,
    required_parent_id: str | None = None,
) -> list[str]:
    child = graph.nodes[child_id]
    child_tokens = tokenize(node_text(child))
    scored: list[tuple[tuple[int, int, int, int, int, str], str]] = []
    required_ok = False

    for parent_id in graph.incoming.get(child_id, []):
        parent = graph.nodes.get(parent_id)
        if parent is None or not is_valid_parent(parent, child):
            continue
        score = parent_score(parent_id, parent, child_tokens, terms)
        scored.append((score, parent_id))
        if parent_id == required_parent_id:
            required_ok = True

    if required_parent_id is not None and not required_ok:
        return []
    if len(scored) < config.min_parents:
        return []

    scored.sort(reverse=True)
    selected: list[str] = []
    if required_parent_id is not None:
        selected.append(required_parent_id)
    for _, parent_id in scored:
        if parent_id in selected:
            continue
        selected.append(parent_id)
        if len(selected) >= config.max_parents:
            break
    if len(selected) < config.min_parents:
        return []

    # Keep a stable, chronological reading order after selecting by quality.
    return sorted(
        selected,
        key=lambda node_id: (
            as_int(graph.nodes[node_id].get("year")) or 9999,
            compact_text(graph.nodes[node_id].get("title")).lower(),
            node_id,
        ),
    )


def split_sentences(text: str) -> list[str]:
    text = compact_text(text)
    if not text:
        return []
    pieces = re.split(r"(?<=[.!?])\s+", text)
    return [piece.strip() for piece in pieces if piece.strip()]


def extract_key_insight(node: JsonDict, max_chars: int) -> str:
    title = compact_text(node.get("title"))
    sentences = split_sentences(str(node.get("abstract") or ""))
    selected = sentences[0] if sentences else ""
    for sentence in sentences:
        lower = sentence.lower()
        if any(cue in lower for cue in CONTRIBUTION_CUES):
            selected = sentence
            break
    if not selected:
        selected = title
    return truncate_text(f"{title}: {selected}", max_chars)


def abstract_label(config: ConversionConfig) -> str:
    return "Processed summary" if config.abstract_processing_mode != "none" else "Abstract"


def paper_block(node: JsonDict, index: int, max_abstract_chars: int, config: ConversionConfig) -> str:
    title = compact_text(node.get("title"))
    year = as_int(node.get("year"))
    abstract = truncate_text(node.get("abstract"), max_abstract_chars)
    arxiv_id = compact_text(node.get("arxiv_id"))
    metadata = [f"year={year if year is not None else 'unknown'}"]
    if arxiv_id:
        metadata.append(f"arxiv={arxiv_id}")
    return "\n".join(
        [
            f"### Parent Paper {index}: {title}",
            f"- Metadata: {', '.join(metadata)}",
            f"- {abstract_label(config)}: {abstract}",
        ]
    )


def child_metadata(node: JsonDict) -> JsonDict:
    return {
        "node_id": node.get("node_id"),
        "paper_id": node.get("paper_id"),
        "corpus_id": node.get("corpus_id"),
        "arxiv_id": node.get("arxiv_id"),
        "doi": node.get("doi"),
        "title": compact_text(node.get("title")),
        "abstract": compact_text(node.get("abstract")),
        "year": as_int(node.get("year")),
        "publication_date": node.get("publication_date"),
        "venue": node.get("venue"),
        "url": node.get("url"),
        "citation_count": as_int(node.get("citation_count")),
        "seed_reference_depth": node.get("seed_reference_depth"),
    }


def parent_metadata(node: JsonDict, config: ConversionConfig) -> JsonDict:
    payload = child_metadata(node)
    payload["abstract"] = truncate_text(node.get("abstract"), config.max_parent_abstract_chars)
    return payload


def build_parent_hypothesis(parent_nodes: list[JsonDict], terms: set[str]) -> str:
    title_tokens: list[str] = []
    seen: set[str] = set()
    preferred_terms = terms - STOPWORDS
    for node in parent_nodes:
        tokens = tokenize(node.get("title"))
        ordered = sorted(tokens & preferred_terms) + sorted(tokens - preferred_terms)
        for token in ordered:
            if token not in seen:
                seen.add(token)
                title_tokens.append(token.replace("-", " "))
            if len(title_tokens) >= 8:
                break
        if len(title_tokens) >= 8:
            break
    if not title_tokens:
        title_tokens = ["these", "methods"]
    theme = ", ".join(title_tokens[:6])
    return (
        "A plausible next research move is to combine and extend the ideas around "
        f"{theme}, looking for a sharper theorem, algorithmic framework, or conceptual bridge "
        "that addresses limitations shared by the parent papers."
    )


def build_single_step_prompt(
    topic_names: list[str],
    parent_nodes: list[JsonDict],
    insights: list[str],
    hypothesis: str,
    config: ConversionConfig,
) -> str:
    parent_sections = [
        paper_block(node, index + 1, config.max_parent_abstract_chars, config)
        for index, node in enumerate(parent_nodes)
    ]
    insight_lines = [f"- {insight}" for insight in insights]
    prompt = "\n\n".join(
        [
            "# Research Direction Proposal",
            "## Task\n"
            "Given the parent papers below, propose the next plausible child paper that could cite them. "
            "Focus on a concrete hypothesis or key insight, not a literature survey.",
            f"## Research Area\n{', '.join(topic_names)}",
            "## Parent Papers\n" + "\n\n".join(parent_sections),
            "## Extracted Insights\n" + "\n".join(insight_lines),
            "## Synthesis Hypothesis\n" + hypothesis,
            "## Response Format\n"
            "Write the proposed child paper with a title, processed summary, and brief rationale connecting it to the parents.",
        ]
    )
    return truncate_text(prompt, config.max_user_chars)


def build_single_step_completion(child: JsonDict, parent_nodes: list[JsonDict], config: ConversionConfig) -> str:
    title = compact_text(child.get("title"))
    year = as_int(child.get("year"))
    abstract = truncate_text(child.get("abstract"), config.max_child_abstract_chars)
    parent_titles = "; ".join(compact_text(node.get("title")) for node in parent_nodes[:3])
    return "\n\n".join(
        [
            "## Proposed Child Paper",
            f"**Title:** {title}",
            f"**Year:** {year if year is not None else 'unknown'}",
            f"## {abstract_label(config)}\n{abstract}",
            "## Why This Follows\n"
            "This child paper is a plausible next step because it cites and builds on "
            f"the parent context ({parent_titles}), extending their shared methods or questions "
            "into a more specific research contribution.",
        ]
    )


def build_citation_edges(graph: TopicGraph, parent_ids: list[str], child_id: str) -> list[JsonDict]:
    edges = []
    for parent_id in parent_ids:
        edge = graph.edge_records.get((parent_id, child_id), {})
        edges.append(
            {
                "source": parent_id,
                "target": child_id,
                "relation": edge.get("relation", "cited_by"),
                "orientation": edge.get("orientation", "cited_reference_to_citing_paper"),
            }
        )
    return edges


def child_title_leaks(prompt: str, child: JsonDict) -> bool:
    title = compact_text(child.get("title")).lower()
    return bool(title) and title in prompt.lower()


def make_single_step_example(
    graph: TopicGraph,
    child_id: str,
    parent_ids: list[str],
    terms: set[str],
    config: ConversionConfig,
) -> JsonDict | None:
    child = graph.nodes[child_id]
    parent_nodes = [graph.nodes[parent_id] for parent_id in parent_ids]
    insights = [extract_key_insight(node, config.max_insight_chars) for node in parent_nodes]
    hypothesis = build_parent_hypothesis(parent_nodes, terms)
    topics = [graph.name]
    prompt = build_single_step_prompt(topics, parent_nodes, insights, hypothesis, config)
    if child_title_leaks(prompt, child):
        return None
    completion = build_single_step_completion(child, parent_nodes, config)
    digest = stable_digest("single|" + child_id + "|" + "|".join(parent_ids))
    return {
        "id": f"single:{digest}",
        "kind": "single_step",
        "topics": topics,
        "topic": graph.name,
        "parent_papers": [parent_metadata(node, config) for node in parent_nodes],
        "parent_key_insights": insights,
        "parent_hypothesis": hypothesis,
        "prompt": prompt,
        "completion": completion,
        "child_paper": child_metadata(child),
        "citation_edges": build_citation_edges(graph, parent_ids, child_id),
    }


def candidate_child_score(
    child_id: str,
    graph: TopicGraph,
    terms: set[str],
    config: ConversionConfig,
) -> tuple[int, int, int, int, int, str]:
    child = graph.nodes[child_id]
    valid_parent_count = sum(
        1
        for parent_id in graph.incoming.get(child_id, [])
        if parent_id in graph.nodes and is_valid_parent(graph.nodes[parent_id], child)
    )
    tokens = tokenize(node_text(child))
    topic_overlap = len(tokens & terms)
    citation_count = as_int(child.get("citation_count")) or 0
    year = as_int(child.get("year")) or 0
    depth = as_int(child.get("seed_reference_depth"))
    depth_score = 100 - depth if depth is not None else 0
    return (valid_parent_count, topic_overlap, citation_count, year, depth_score, child_id)


def build_single_step_examples(
    graph: TopicGraph,
    terms: set[str],
    config: ConversionConfig,
    skipped: Counter[str],
) -> list[JsonDict]:
    candidates = []
    for child_id in graph.incoming:
        child = graph.nodes.get(child_id)
        if child is None or not has_title_abstract_year(child):
            skipped["single_child_missing_title_abstract_year"] += 1
            continue
        parent_ids = select_parent_ids(child_id, graph, terms, config)
        if not parent_ids:
            skipped["single_not_enough_valid_parents"] += 1
            continue
        candidates.append((candidate_child_score(child_id, graph, terms, config), child_id, parent_ids))

    candidates.sort(reverse=True)
    if config.max_single_examples_per_topic:
        candidates = candidates[: config.max_single_examples_per_topic]

    examples = []
    for _, child_id, parent_ids in candidates:
        example = make_single_step_example(graph, child_id, parent_ids, terms, config)
        if example is None:
            skipped["single_child_title_leaks_in_prompt"] += 1
            continue
        examples.append(example)
    return examples


def depth_value(node: JsonDict) -> int | None:
    return as_int(node.get("seed_reference_depth"))


def trajectory_next_child(
    current_id: str,
    path_ids: list[str],
    graph: TopicGraph,
    terms: set[str],
    config: ConversionConfig,
) -> tuple[str, list[str]] | None:
    current = graph.nodes[current_id]
    current_depth = depth_value(current)
    current_year = as_int(current.get("year"))
    scored = []
    for child_id in graph.outgoing.get(current_id, []):
        if child_id in path_ids:
            continue
        child = graph.nodes[child_id]
        if not has_title_abstract_year(child):
            continue
        child_year = as_int(child.get("year"))
        if current_year is not None and child_year is not None and child_year < current_year:
            continue
        child_depth = depth_value(child)
        if current_depth is not None and child_depth is not None and child_depth > current_depth:
            continue
        parent_ids = select_parent_ids(
            child_id,
            graph,
            terms,
            config,
            required_parent_id=current_id,
        )
        if not parent_ids:
            continue
        progress = 0
        if current_depth is not None and child_depth is not None:
            progress = current_depth - child_depth
        tokens = tokenize(node_text(child))
        score = (
            progress,
            len(tokens & terms),
            len(parent_ids),
            as_int(child.get("citation_count")) or 0,
            child_year or 0,
            child_id,
        )
        scored.append((score, child_id, parent_ids))
    if not scored:
        return None
    scored.sort(reverse=True)
    _, child_id, parent_ids = scored[0]
    return child_id, parent_ids


def start_node_score(node_id: str, graph: TopicGraph, terms: set[str]) -> tuple[int, int, int, int, str]:
    node = graph.nodes[node_id]
    tokens = tokenize(node_text(node))
    depth = depth_value(node) or 0
    return (
        depth,
        len(tokens & terms),
        len(graph.outgoing.get(node_id, [])),
        as_int(node.get("citation_count")) or 0,
        node_id,
    )


def make_trajectory_example(
    graph: TopicGraph,
    path_ids: list[str],
    turn_parent_ids: list[list[str]],
    terms: set[str],
    config: ConversionConfig,
) -> JsonDict | None:
    turns = []
    completion_sections = []
    for turn_index, child_id in enumerate(path_ids[1:], start=1):
        parent_ids = turn_parent_ids[turn_index - 1]
        child = graph.nodes[child_id]
        parent_nodes = [graph.nodes[parent_id] for parent_id in parent_ids]
        insights = [extract_key_insight(node, config.max_insight_chars) for node in parent_nodes]
        hypothesis = build_parent_hypothesis(parent_nodes, terms)
        turns.append(
            {
                "turn_index": turn_index,
                "parent_papers": [parent_metadata(node, config) for node in parent_nodes],
                "parent_key_insights": insights,
                "parent_hypothesis": hypothesis,
                "child_paper": child_metadata(child),
                "citation_edges": build_citation_edges(graph, parent_ids, child_id),
            }
        )
        completion_sections.append(
            "\n\n".join(
                [
                    f"## Step {turn_index}: {compact_text(child.get('title'))}",
                    f"**Year:** {as_int(child.get('year')) if as_int(child.get('year')) is not None else 'unknown'}",
                    f"### {abstract_label(config)}\n{truncate_text(child.get('abstract'), config.max_child_abstract_chars)}",
                ]
            )
        )

    start = graph.nodes[path_ids[0]]
    start_prompt = "\n\n".join(
        [
            "# Multi-Step Research Trajectory",
            "## Task\n"
            "Start from this paper and propose a sequence of later child papers that could reasonably cite it, "
            "where each step should build on the previous research direction.",
            f"## Research Area\n{graph.name}",
            "## Starting Paper\n" + paper_block(start, 1, config.max_parent_abstract_chars, config),
            "## Response Format\nWrite the sequence as concrete child papers with titles, processed summaries, and years.",
        ]
    )
    prompt = truncate_text(start_prompt, config.max_user_chars)
    for child_id in path_ids[1:]:
        if child_title_leaks(prompt, graph.nodes[child_id]):
            return None

    completion = "\n\n".join(completion_sections)
    digest = stable_digest("trajectory|" + "|".join(path_ids))
    return {
        "id": f"trajectory:{digest}",
        "kind": "trajectory",
        "topic": graph.name,
        "topics": [graph.name],
        "path_node_ids": path_ids,
        "turns": turns,
        "prompt": prompt,
        "completion": completion,
        "child_paper": child_metadata(graph.nodes[path_ids[-1]]),
        "parent_papers": [parent_metadata(graph.nodes[path_ids[0]], config)],
    }


def build_trajectory_examples(
    graph: TopicGraph,
    terms: set[str],
    config: ConversionConfig,
    skipped: Counter[str],
) -> list[JsonDict]:
    if not config.include_trajectories or config.max_trajectories_per_topic == 0:
        return []

    start_ids = [
        node_id
        for node_id, node in graph.nodes.items()
        if has_title_abstract_year(node) and graph.outgoing.get(node_id)
    ]
    start_ids.sort(key=lambda node_id: start_node_score(node_id, graph, terms), reverse=True)

    examples = []
    seen_paths: set[tuple[str, ...]] = set()
    for start_id in start_ids:
        path_ids = [start_id]
        turn_parent_ids: list[list[str]] = []
        while len(turn_parent_ids) < config.max_trajectory_turns:
            next_step = trajectory_next_child(path_ids[-1], path_ids, graph, terms, config)
            if next_step is None:
                break
            child_id, parent_ids = next_step
            path_ids.append(child_id)
            turn_parent_ids.append(parent_ids)
            if depth_value(graph.nodes[child_id]) == 0:
                break

        if len(turn_parent_ids) < config.min_trajectory_turns:
            skipped["trajectory_too_short"] += 1
            continue
        key = tuple(path_ids)
        if key in seen_paths:
            skipped["trajectory_duplicate_path"] += 1
            continue
        seen_paths.add(key)
        example = make_trajectory_example(graph, path_ids, turn_parent_ids, terms, config)
        if example is None:
            skipped["trajectory_child_title_leaks_in_prompt"] += 1
            continue
        examples.append(example)
        if len(examples) >= config.max_trajectories_per_topic:
            break
    return examples


def example_node_ids(example: JsonDict) -> set[str]:
    node_ids: set[str] = set()
    for paper in example.get("parent_papers") or []:
        node_id = str(paper.get("node_id") or "")
        if node_id:
            node_ids.add(node_id)
    child = example.get("child_paper") or {}
    child_id = str(child.get("node_id") or "")
    if child_id:
        node_ids.add(child_id)
    for turn in example.get("turns") or []:
        for paper in turn.get("parent_papers") or []:
            node_id = str(paper.get("node_id") or "")
            if node_id:
                node_ids.add(node_id)
        turn_child = turn.get("child_paper") or {}
        turn_child_id = str(turn_child.get("node_id") or "")
        if turn_child_id:
            node_ids.add(turn_child_id)
    return node_ids


def record_example_node_sources(
    example: JsonDict,
    graph_nodes: dict[str, JsonDict],
    graph_path: Path,
    node_sources: dict[str, JsonDict],
) -> None:
    for node_id in example_node_ids(example):
        node = graph_nodes.get(node_id)
        if node is not None and compact_text(node.get("abstract")):
            node_copy = dict(node)
            node_copy["_graph_dir"] = str(graph_path)
            node_sources.setdefault(node_id, node_copy)


def apply_processed_abstract_to_paper(
    paper: JsonDict,
    abstract_records: dict[str, JsonDict],
    config: ConversionConfig,
    *,
    max_chars: int,
) -> None:
    node_id = str(paper.get("node_id") or "")
    record = abstract_records.get(node_id)
    if not record:
        return
    processed = compact_text(record.get("processed_abstract"))
    paper["abstract"] = truncate_text(processed, max_chars)
    paper["abstract_processing"] = {
        "mode": record.get("mode"),
        "model": record.get("model"),
        "source_kind": record.get("source_kind"),
        "source_path": record.get("source_path"),
        "source_sha1": record.get("source_sha1"),
        "raw_abstract_sha1": record.get("raw_abstract_sha1"),
        "removed_quantities": record.get("removed_quantities") or [],
        "quality_flags": record.get("quality_flags") or [],
    }


def refresh_single_step_example(example: JsonDict, config: ConversionConfig) -> JsonDict | None:
    parent_nodes = list(example.get("parent_papers") or [])
    child = dict(example.get("child_paper") or {})
    insights = [extract_key_insight(node, config.max_insight_chars) for node in parent_nodes]
    topics = list(example.get("topics") or [example.get("topic", "")])
    terms = tokenize(" ".join(str(topic).replace("_", " ") for topic in topics))
    hypothesis = build_parent_hypothesis(parent_nodes, terms)
    prompt = build_single_step_prompt(topics, parent_nodes, insights, hypothesis, config)
    if child_title_leaks(prompt, child):
        return None
    example["parent_key_insights"] = insights
    example["parent_hypothesis"] = hypothesis
    example["prompt"] = prompt
    example["completion"] = build_single_step_completion(child, parent_nodes, config)
    return example


def refresh_trajectory_example(example: JsonDict, config: ConversionConfig) -> JsonDict | None:
    topics = list(example.get("topics") or [example.get("topic", "")])
    start_papers = list(example.get("parent_papers") or [])
    if not start_papers:
        return None
    start = start_papers[0]
    prompt = "\n\n".join(
        [
            "# Multi-Step Research Trajectory",
            "## Task\n"
            "Start from this paper and propose a sequence of later child papers that could reasonably cite it, "
            "where each step should build on the previous research direction.",
            f"## Research Area\n{', '.join(str(topic) for topic in topics if topic)}",
            "## Starting Paper\n" + paper_block(start, 1, config.max_parent_abstract_chars, config),
            "## Response Format\nWrite the sequence as concrete child papers with titles, processed summaries, and years.",
        ]
    )
    prompt = truncate_text(prompt, config.max_user_chars)
    completion_sections = []
    terms = tokenize(" ".join(str(topic).replace("_", " ") for topic in topics))
    for turn in example.get("turns") or []:
        parent_nodes = list(turn.get("parent_papers") or [])
        child = dict(turn.get("child_paper") or {})
        if child_title_leaks(prompt, child):
            return None
        turn["parent_key_insights"] = [extract_key_insight(node, config.max_insight_chars) for node in parent_nodes]
        turn["parent_hypothesis"] = build_parent_hypothesis(parent_nodes, terms)
        turn_index = int(turn.get("turn_index") or len(completion_sections) + 1)
        completion_sections.append(
            "\n\n".join(
                [
                    f"## Step {turn_index}: {compact_text(child.get('title'))}",
                    f"**Year:** {as_int(child.get('year')) if as_int(child.get('year')) is not None else 'unknown'}",
                    f"### {abstract_label(config)}\n{truncate_text(child.get('abstract'), config.max_child_abstract_chars)}",
                ]
            )
        )
    example["prompt"] = prompt
    example["completion"] = "\n\n".join(completion_sections)
    return example


def apply_processed_abstracts(
    examples: list[JsonDict],
    abstract_records: dict[str, JsonDict],
    config: ConversionConfig,
    skipped: Counter[str],
    progress: ProgressReporter | None = None,
) -> list[JsonDict]:
    refreshed = []
    total = len(examples)
    if progress:
        progress.start_stage("Applying processed abstracts", total=total, unit="examples")
    for index, example in enumerate(examples, start=1):
        missing = sorted(node_id for node_id in example_node_ids(example) if node_id not in abstract_records)
        if missing:
            skipped["abstract_processing_missing_record"] += 1
            if progress:
                progress.update(index, total=total, detail=f"kept={len(refreshed):,}")
            continue
        for paper in example.get("parent_papers") or []:
            apply_processed_abstract_to_paper(
                paper,
                abstract_records,
                config,
                max_chars=config.max_parent_abstract_chars,
            )
        child = example.get("child_paper") or {}
        apply_processed_abstract_to_paper(
            child,
            abstract_records,
            config,
            max_chars=config.max_child_abstract_chars,
        )
        for turn in example.get("turns") or []:
            for paper in turn.get("parent_papers") or []:
                apply_processed_abstract_to_paper(
                    paper,
                    abstract_records,
                    config,
                    max_chars=config.max_parent_abstract_chars,
                )
            turn_child = turn.get("child_paper") or {}
            apply_processed_abstract_to_paper(
                turn_child,
                abstract_records,
                config,
                max_chars=config.max_child_abstract_chars,
            )
        if example["kind"] == "single_step":
            next_example = refresh_single_step_example(example, config)
        else:
            next_example = refresh_trajectory_example(example, config)
        if next_example is None:
            skipped["processed_child_title_leaks_in_prompt"] += 1
            if progress:
                progress.update(index, total=total, detail=f"kept={len(refreshed):,}")
            continue
        refreshed.append(next_example)
        if progress:
            progress.update(index, total=total, detail=f"kept={len(refreshed):,}")
    if progress:
        progress.finish_stage(completed=total, total=total, detail=f"kept={len(refreshed):,}")
    return refreshed


def merge_duplicate_example(target: JsonDict, source: JsonDict) -> None:
    target_topics = list(target.get("topics") or [])
    for topic in source.get("topics") or []:
        if topic not in target_topics:
            target_topics.append(topic)
    target["topics"] = target_topics
    if len(target_topics) > 1:
        target["topic"] = ",".join(target_topics)


def collect_examples(
    config: ConversionConfig,
    progress: ProgressReporter | None = None,
) -> tuple[list[JsonDict], JsonDict]:
    topic_dirs = discover_topic_dirs(config.input_root, config.topics, config.topic_limit)
    examples_by_id: dict[str, JsonDict] = {}
    node_sources_by_id: dict[str, JsonDict] = {}
    skipped: Counter[str] = Counter()
    topic_summaries = []

    if progress:
        progress.log(
            f"Discovered {len(topic_dirs):,} topic graph directories under {config.input_root}"
        )
        progress.start_stage("Topic graph conversion", total=len(topic_dirs), unit="topics")
    for index, topic_dir in enumerate(topic_dirs, start=1):
        graph = load_topic_graph(topic_dir)
        terms = topic_tokens(graph)
        single_examples = build_single_step_examples(graph, terms, config, skipped)
        trajectory_examples = build_trajectory_examples(graph, terms, config, skipped)
        for example in single_examples + trajectory_examples:
            existing = examples_by_id.get(example["id"])
            if existing is None:
                examples_by_id[example["id"]] = example
                record_example_node_sources(example, graph.nodes, graph.path, node_sources_by_id)
            else:
                skipped["duplicate_example_merged"] += 1
                merge_duplicate_example(existing, example)
                record_example_node_sources(existing, graph.nodes, graph.path, node_sources_by_id)

        topic_summaries.append(
            {
                "topic": graph.name,
                "nodes": len(graph.nodes),
                "edges": len(graph.edge_records),
                "single_examples": len(single_examples),
                "trajectory_examples": len(trajectory_examples),
            }
        )
        if progress:
            progress.update(
                index,
                total=len(topic_dirs),
                detail=(
                    f"{topic_dir.name}: nodes={len(graph.nodes):,}, edges={len(graph.edge_records):,}, "
                    f"topic_examples={len(single_examples) + len(trajectory_examples):,}, "
                    f"unique_examples={len(examples_by_id):,}"
                ),
            )
    if progress:
        progress.finish_stage(
            completed=len(topic_dirs),
            total=len(topic_dirs),
            detail=f"unique_examples={len(examples_by_id):,}",
        )

    examples = sorted(examples_by_id.values(), key=lambda row: (row["kind"], row["id"]))
    if config.max_examples:
        examples = examples[: config.max_examples]

    final_node_ids: set[str] = set()
    for example in examples:
        final_node_ids.update(example_node_ids(example))
    final_node_sources = {
        node_id: node_sources_by_id[node_id]
        for node_id in sorted(final_node_ids)
        if node_id in node_sources_by_id
    }
    if progress:
        progress.log(
            f"Selected {len(examples):,} examples using {len(final_node_sources):,} unique paper sources"
        )
    abstract_records, abstract_processing_manifest = process_abstract_records(final_node_sources, config, progress)
    examples = apply_processed_abstracts(examples, abstract_records, config, skipped, progress)

    manifest = {
        "input_root": str(config.input_root.resolve()),
        "output_dir": str(config.output_dir.resolve()),
        "seed": config.seed,
        "topic_count": len(topic_dirs),
        "topics": [path.name for path in topic_dirs],
        "num_examples": len(examples),
        "num_single_step_examples": sum(1 for row in examples if row["kind"] == "single_step"),
        "num_trajectory_examples": sum(1 for row in examples if row["kind"] == "trajectory"),
        "topic_summaries": topic_summaries,
        "abstract_processing": abstract_processing_manifest,
        "skipped": dict(sorted(skipped.items())),
        "config": {
            "max_examples": config.max_examples,
            "max_single_examples_per_topic": config.max_single_examples_per_topic,
            "max_trajectories_per_topic": config.max_trajectories_per_topic,
            "min_parents": config.min_parents,
            "max_parents": config.max_parents,
            "min_trajectory_turns": config.min_trajectory_turns,
            "max_trajectory_turns": config.max_trajectory_turns,
            "valid_fraction": config.valid_fraction,
            "test_fraction": config.test_fraction,
            "max_parent_abstract_chars": config.max_parent_abstract_chars,
            "max_child_abstract_chars": config.max_child_abstract_chars,
            "max_user_chars": config.max_user_chars,
            "write_sft_parquet": config.write_sft_parquet,
            "write_sft_jsonl": config.write_sft_jsonl,
            "abstract_processing_mode": config.abstract_processing_mode,
            "abstract_model": config.abstract_model if config.abstract_processing_mode == "gemini" else None,
            "abstract_batch_size": config.abstract_batch_size,
            "abstract_batch_max_chars": config.abstract_batch_max_chars,
            "abstract_max_concurrent": config.abstract_max_concurrent,
            "abstract_cache_jsonl": config.abstract_cache_jsonl,
            "paper_source_max_chars": config.paper_source_max_chars,
            "progress_interval": config.progress_interval,
            "no_progress": config.no_progress,
        },
    }
    return examples, manifest


def assign_split(example_id: str, valid_fraction: float, test_fraction: float) -> str:
    bucket = int(hashlib.sha1(example_id.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    if bucket < test_fraction:
        return "test"
    if bucket < test_fraction + valid_fraction:
        return "valid"
    return "train"


def split_examples(examples: list[JsonDict], config: ConversionConfig) -> dict[str, list[JsonDict]]:
    splits = {"train": [], "valid": [], "test": []}
    for example in examples:
        split = assign_split(example["id"], config.valid_fraction, config.test_fraction)
        splits[split].append(example)
    return splits


def example_to_sft_row(example: JsonDict) -> JsonDict:
    parent_ids = []
    if example["kind"] == "trajectory":
        parent_ids = list(example.get("path_node_ids") or [])[:-1]
    else:
        parent_ids = [str(parent.get("node_id")) for parent in example.get("parent_papers") or []]
    child = example.get("child_paper") or {}
    topics = example.get("topics") or [example.get("topic", "")]
    return {
        "messages": [
            {"role": "user", "content": str(example["prompt"])},
            {"role": "assistant", "content": str(example["completion"])},
        ],
        "id": str(example["id"]),
        "kind": str(example["kind"]),
        "topic": ",".join(str(topic) for topic in topics if topic),
        "child_node_id": str(child.get("node_id") or ""),
        "parent_node_ids": [str(parent_id) for parent_id in parent_ids if parent_id],
    }


def sft_schema() -> Any:
    import pyarrow as pa

    return pa.schema(
        [
            pa.field(
                "messages",
                pa.list_(
                    pa.struct(
                        [
                            pa.field("role", pa.string()),
                            pa.field("content", pa.string()),
                        ]
                    )
                ),
            ),
            pa.field("id", pa.string()),
            pa.field("kind", pa.string()),
            pa.field("topic", pa.string()),
            pa.field("child_node_id", pa.string()),
            pa.field("parent_node_ids", pa.list_(pa.string())),
        ]
    )


def write_sft_parquet(path: Path, rows: list[JsonDict]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=sft_schema())
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, tmp_path)
    tmp_path.replace(path)


def write_outputs(
    examples: list[JsonDict],
    manifest: JsonDict,
    config: ConversionConfig,
    progress: ProgressReporter | None = None,
) -> JsonDict:
    splits = split_examples(examples, config)
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    total_writes = 1 + len(splits) + 1
    if config.write_sft_jsonl:
        total_writes += len(splits)
    if config.write_sft_parquet:
        total_writes += len(splits)
    completed_writes = 0
    if progress:
        progress.start_stage("Writing outputs", total=total_writes, unit="files", detail=str(output_dir))

    write_jsonl(output_dir / "examples.jsonl", examples)
    completed_writes += 1
    if progress:
        progress.update(completed_writes, total=total_writes, detail="examples.jsonl", force=True)
    split_counts = {}
    for split, rows in splits.items():
        split_counts[split] = write_jsonl(output_dir / f"{split}.jsonl", rows)
        completed_writes += 1
        if progress:
            progress.update(completed_writes, total=total_writes, detail=f"{split}.jsonl", force=True)

    sft_counts = {}
    sft_dir = Path(config.sft_output_dir)
    if not sft_dir.is_absolute():
        sft_dir = output_dir / sft_dir
    if config.write_sft_jsonl or config.write_sft_parquet:
        sft_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        sft_rows = [example_to_sft_row(row) for row in rows]
        sft_counts[split] = len(sft_rows)
        if config.write_sft_jsonl:
            write_jsonl(sft_dir / f"{split}.jsonl", sft_rows)
            completed_writes += 1
            if progress:
                progress.update(
                    completed_writes,
                    total=total_writes,
                    detail=f"sft/{split}.jsonl",
                    force=True,
                )
        if config.write_sft_parquet:
            write_sft_parquet(sft_dir / f"{split}.parquet", sft_rows)
            completed_writes += 1
            if progress:
                progress.update(
                    completed_writes,
                    total=total_writes,
                    detail=f"sft/{split}.parquet",
                    force=True,
                )

    final_manifest = {
        **manifest,
        "files": {
            "examples_jsonl": str((output_dir / "examples.jsonl").resolve()),
            "train_jsonl": str((output_dir / "train.jsonl").resolve()),
            "valid_jsonl": str((output_dir / "valid.jsonl").resolve()),
            "test_jsonl": str((output_dir / "test.jsonl").resolve()),
            "sft_dir": str(sft_dir.resolve()),
            "sft_train_parquet": str((sft_dir / "train.parquet").resolve()) if config.write_sft_parquet else None,
            "sft_valid_parquet": str((sft_dir / "valid.parquet").resolve()) if config.write_sft_parquet else None,
            "sft_test_parquet": str((sft_dir / "test.parquet").resolve()) if config.write_sft_parquet else None,
        },
        "split_counts": split_counts,
        "sft_counts": sft_counts,
    }
    write_json(output_dir / "manifest.json", final_manifest)
    completed_writes += 1
    if progress:
        progress.update(completed_writes, total=total_writes, detail="manifest.json", force=True)
        progress.finish_stage(completed=completed_writes, total=total_writes)
    return final_manifest


def run_conversion(
    config: ConversionConfig,
    progress: ProgressReporter | None = None,
) -> JsonDict:
    progress = progress or make_progress_reporter(config)
    progress.log(
        "Starting citation graph conversion | "
        f"input={config.input_root} | output={config.output_dir} | "
        f"abstract_mode={config.abstract_processing_mode}"
    )
    examples, manifest = collect_examples(config, progress)
    final_manifest = write_outputs(examples, manifest, config, progress)
    progress.log(
        "Finished citation graph conversion | "
        f"examples={final_manifest['num_examples']:,} | output={final_manifest['output_dir']}"
    )
    return final_manifest


def main() -> None:
    manifest = run_conversion(config_from_args(parse_args()))
    print(
        "Wrote {num_examples} examples to {output_dir} "
        "(train={train}, valid={valid}, test={test})".format(
            num_examples=manifest["num_examples"],
            output_dir=manifest["output_dir"],
            train=manifest["split_counts"]["train"],
            valid=manifest["split_counts"]["valid"],
            test=manifest["split_counts"]["test"],
        )
    )


if __name__ == "__main__":
    main()
