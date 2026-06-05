#!/usr/bin/env python3
"""Build a forward-time citation graph from arXiv-backed papers.

The script can use the arXiv Atom API or Semantic Scholar search to collect
candidate arXiv IDs, enriches those candidates with Semantic Scholar metadata,
keeps the best-cited papers, and writes a citation graph with edges from
cited/reference papers to the later papers that cite them.

Example:
    python arxiv_paper/build_arxiv_citation_graph.py \
        --query "cat:cs.LG AND all:transformer" \
        --max-arxiv-results 500 \
        --min-citations 250 \
        --max-seed-papers 100 \
        --output-dir data/arxiv_citation_graph/cs_lg_transformers
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import email.utils
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, TextIO
from xml.etree import ElementTree as ET

try:
    import fcntl
except ImportError:  # pragma: no cover - non-Unix fallback
    fcntl = None  # type: ignore[assignment]

try:
    import requests
except ImportError as exc:  # pragma: no cover - import-time guard
    raise SystemExit("Missing dependency: pip install requests") from exc


ARXIV_API_URL = "https://export.arxiv.org/api/query"
ARXIV_HTML_URL_TEMPLATE = "https://arxiv.org/html/{arxiv_id}"
S2_GRAPH_API_URL = "https://api.semanticscholar.org/graph/v1"
USER_AGENT = "asmr-private-arxiv-citation-graph/1.0"
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
RATE_LIMIT_BASE_SLEEP_SECONDS = 60.0
RATE_LIMIT_MAX_SLEEP_SECONDS = 600.0
ARXIV_RATE_LIMIT_BASE_SLEEP_SECONDS = 300.0
ARXIV_RATE_LIMIT_MAX_SLEEP_SECONDS = 1800.0
TRANSIENT_MAX_SLEEP_SECONDS = 60.0
ARXIV_RETRY_JITTER_FRACTION = 0.2
ARXIV_MIN_SLEEP_SECONDS = 3.0

ATOM_NS = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"

S2_PAPER_FIELDS = ",".join(
    [
        "paperId",
        "corpusId",
        "externalIds",
        "url",
        "title",
        "abstract",
        "venue",
        "year",
        "publicationDate",
        "authors",
        "citationCount",
        "influentialCitationCount",
        "referenceCount",
        "fieldsOfStudy",
        "s2FieldsOfStudy",
        "publicationTypes",
        "references.paperId",
        "references.corpusId",
        "references.externalIds",
        "references.url",
        "references.title",
        "references.venue",
        "references.year",
        "references.publicationDate",
        "references.authors",
        "references.citationCount",
        "references.influentialCitationCount",
        "references.referenceCount",
        "references.fieldsOfStudy",
        "references.s2FieldsOfStudy",
        "references.publicationTypes",
    ]
)

S2_SEARCH_FIELDS = ",".join(
    [
        "paperId",
        "corpusId",
        "externalIds",
        "url",
        "title",
        "abstract",
        "venue",
        "year",
        "publicationDate",
        "authors",
        "citationCount",
        "influentialCitationCount",
        "referenceCount",
        "fieldsOfStudy",
        "s2FieldsOfStudy",
        "publicationTypes",
    ]
)

ARXIV_CATEGORY_TO_S2_DEFAULTS: dict[str, tuple[str, str]] = {
    "astro-ph": ("Physics", "astrophysics"),
    "astro-ph.CO": ("Physics", "cosmology"),
    "cond-mat": ("Physics", "condensed matter"),
    "cond-mat.stat-mech": ("Physics", "statistical mechanics"),
    "cs.AI": ("Computer Science", "artificial intelligence"),
    "cs.CL": ("Computer Science", "natural language processing"),
    "cs.CV": ("Computer Science", "computer vision"),
    "cs.LG": ("Computer Science", "machine learning"),
    "cs.RO": ("Computer Science", "robotics"),
    "econ.EM": ("Economics", "econometrics"),
    "eess.IV": ("Engineering", "image processing"),
    "eess.SP": ("Engineering", "signal processing"),
    "hep-th": ("Physics", "high energy physics theory"),
    "math.OC": ("Mathematics", "optimization control"),
    "math.PR": ("Mathematics", "probability"),
    "math.ST": ("Mathematics", "statistics"),
    "q-bio": ("Biology", "quantitative biology"),
    "q-bio.QM": ("Biology", "quantitative methods biology"),
    "q-fin": ("Economics", "quantitative finance"),
    "q-fin.ST": ("Economics", "statistical finance"),
    "quant-ph": ("Physics", "quantum physics"),
    "stat.ME": ("Mathematics", "statistical methodology"),
    "stat.ML": ("Computer Science", "machine learning"),
}


def default_arxiv_rate_limit_lock_path() -> Path:
    user_id = getattr(os, "getuid", lambda: "user")()
    return Path(os.environ.get("TMPDIR", "/tmp")) / f"asmr_arxiv_api_{user_id}.lock"


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
                eta = (total - completed) / rate
                parts.append(f"eta {format_duration(eta)}")
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect arXiv-backed candidates, keep well-cited papers using "
            "Semantic Scholar, and build a citation graph with cited papers "
            "pointing to later citing papers."
        )
    )
    parser.add_argument(
        "--query",
        default="cat:cs.LG",
        help=(
            "arXiv search_query value. Examples: 'cat:cs.LG', "
            "'cat:cs.CL AND all:retrieval', or 'all:diffusion model'."
        ),
    )
    parser.add_argument(
        "--candidate-source",
        choices=["arxiv", "s2-search"],
        default="arxiv",
        help=(
            "Where to collect seed-paper candidates. 's2-search' avoids the "
            "arXiv Atom API and searches Semantic Scholar for arXiv-backed papers."
        ),
    )
    parser.add_argument(
        "--max-arxiv-results",
        type=int,
        default=500,
        help="Maximum arXiv candidate records to fetch before S2 filtering.",
    )
    parser.add_argument(
        "--arxiv-batch-size",
        type=int,
        default=100,
        help="Number of arXiv records to request per Atom API call.",
    )
    parser.add_argument(
        "--arxiv-sleep",
        type=float,
        default=3.0,
        help="Seconds to sleep between arXiv API calls.",
    )
    parser.add_argument(
        "--arxiv-initial-sleep",
        type=float,
        default=0.0,
        help=(
            "Seconds to wait before the first arXiv API call. Useful when "
            "rerunning immediately after a 429."
        ),
    )
    parser.add_argument(
        "--arxiv-rate-limit-base-sleep",
        type=float,
        default=ARXIV_RATE_LIMIT_BASE_SLEEP_SECONDS,
        help="Initial sleep after an arXiv HTTP 429 before exponential backoff.",
    )
    parser.add_argument(
        "--arxiv-rate-limit-max-sleep",
        type=float,
        default=ARXIV_RATE_LIMIT_MAX_SLEEP_SECONDS,
        help="Maximum sleep after repeated arXiv HTTP 429 responses.",
    )
    parser.add_argument(
        "--arxiv-retry-jitter",
        type=float,
        default=ARXIV_RETRY_JITTER_FRACTION,
        help=(
            "Extra random retry delay as a fraction of the computed arXiv "
            "backoff. Set 0 to disable."
        ),
    )
    parser.add_argument(
        "--arxiv-rate-limit-lock-path",
        type=Path,
        default=default_arxiv_rate_limit_lock_path(),
        help=(
            "Local lock file used to pace arXiv requests across concurrent "
            "processes on this machine."
        ),
    )
    parser.add_argument(
        "--no-arxiv-rate-limit-lock",
        action="store_true",
        help="Disable the local cross-process arXiv request pacing lock.",
    )
    parser.add_argument(
        "--sort-by",
        choices=["relevance", "lastUpdatedDate", "submittedDate"],
        default="relevance",
        help="arXiv API sortBy value for candidate collection.",
    )
    parser.add_argument(
        "--sort-order",
        choices=["ascending", "descending"],
        default="descending",
        help="arXiv API sortOrder value for candidate collection.",
    )
    parser.add_argument(
        "--min-citations",
        type=int,
        default=100,
        help="Minimum Semantic Scholar citationCount for seed arXiv papers.",
    )
    parser.add_argument(
        "--max-seed-papers",
        type=int,
        default=200,
        help="Maximum number of well-cited arXiv seed papers to keep.",
    )
    parser.add_argument(
        "--graph-size-preset",
        choices=["none", "domain-10k"],
        default="none",
        help=(
            "Convenience preset for graph size knobs. 'domain-10k' sets a "
            "depth-2 crawl with a roughly 10k-node budget unless overridden "
            "by explicit CLI flags."
        ),
    )
    parser.add_argument(
        "--target-node-count",
        type=int,
        default=0,
        help=(
            "Approximate hard budget for graph nodes. Once reached, new "
            "reference nodes are skipped but edges among existing nodes can "
            "still be added. Use 0 for no node budget."
        ),
    )
    parser.add_argument(
        "--citation-depth",
        type=int,
        default=1,
        help=(
            "Number of backward reference hops to traverse from seed papers. "
            "1 keeps today's behavior: direct references of seeds only. "
            "2 also includes references of those references, and so on."
        ),
    )
    parser.add_argument(
        "--max-references-per-paper",
        type=int,
        default=0,
        help=(
            "Breadth cap for references inspected per expanded paper. "
            "References are ranked by citation/reference metadata before "
            "capping. Use 0 for no per-paper cap."
        ),
    )
    parser.add_argument(
        "--max-expansion-papers-per-depth",
        type=int,
        default=0,
        help=(
            "Optional cap on how many newly discovered papers are fetched and "
            "expanded at each depth after the seed frontier. Use 0 for no "
            "frontier cap."
        ),
    )
    parser.add_argument(
        "--min-year",
        type=int,
        default=None,
        help="Optional minimum publication year for seed papers.",
    )
    parser.add_argument(
        "--max-year",
        type=int,
        default=None,
        help="Optional maximum publication year for seed papers.",
    )
    parser.add_argument(
        "--internal-only",
        action="store_true",
        help=(
            "Only emit edges where both cited and citing papers are selected "
            "well-cited arXiv seeds. By default, referenced papers outside "
            "the seed set are included as reference nodes."
        ),
    )
    parser.add_argument(
        "--arxiv-references-only",
        action="store_true",
        help=(
            "Only include referenced papers that also have an arXiv external ID. "
            "Seed papers are always arXiv papers."
        ),
    )
    parser.add_argument(
        "--allow-nontemporal-edges",
        action="store_true",
        help=(
            "Keep reference edges even when metadata says the cited paper is "
            "newer than the citing seed paper."
        ),
    )
    parser.add_argument(
        "--fetch-arxiv-html",
        action="store_true",
        help=(
            "Download arXiv HTML pages from https://arxiv.org/html/<id> for "
            "arXiv-backed graph nodes and annotate nodes with local paths."
        ),
    )
    parser.add_argument(
        "--arxiv-html-scope",
        choices=["seed", "graph"],
        default="seed",
        help=(
            "Which papers to fetch HTML for when --fetch-arxiv-html is set. "
            "'seed' fetches selected seed papers only; 'graph' fetches every "
            "graph node with an arXiv ID."
        ),
    )
    parser.add_argument(
        "--arxiv-html-dir",
        type=Path,
        default=Path("paper_html"),
        help=(
            "Directory for downloaded arXiv HTML. Relative paths are resolved "
            "under --output-dir."
        ),
    )
    parser.add_argument(
        "--overwrite-arxiv-html",
        action="store_true",
        help="Refetch arXiv HTML even when the target HTML file already exists.",
    )
    parser.add_argument(
        "--arxiv-html-concurrency",
        type=int,
        default=4,
        help="Number of arXiv HTML downloads to run concurrently.",
    )
    parser.add_argument(
        "--s2-batch-size",
        type=int,
        default=500,
        help="Number of paper IDs per Semantic Scholar batch call.",
    )
    parser.add_argument(
        "--s2-search-query",
        default=None,
        help=(
            "Plain-text Semantic Scholar search query for --candidate-source "
            "s2-search. If omitted, a best-effort query is derived from --query."
        ),
    )
    parser.add_argument(
        "--s2-fields-of-study",
        default=None,
        help=(
            "Comma-separated Semantic Scholar fieldsOfStudy filter for "
            "--candidate-source s2-search, e.g. 'Computer Science'. If omitted, "
            "a best-effort value is derived from arXiv cat: prefixes."
        ),
    )
    parser.add_argument(
        "--s2-publication-types",
        default=None,
        help=(
            "Comma-separated Semantic Scholar publicationTypes filter for "
            "--candidate-source s2-search."
        ),
    )
    parser.add_argument(
        "--s2-search-sort",
        default="citationCount:desc",
        help=(
            "Semantic Scholar bulk-search sort, e.g. citationCount:desc or "
            "publicationDate:desc."
        ),
    )
    parser.add_argument(
        "--s2-search-batch-size",
        type=int,
        default=1000,
        help="Semantic Scholar bulk-search records to request per page, max 1000.",
    )
    parser.add_argument(
        "--s2-search-max-pages",
        type=int,
        default=20,
        help="Maximum Semantic Scholar bulk-search pages to scan for candidates.",
    )
    parser.add_argument(
        "--s2-concurrency",
        type=int,
        default=1,
        help=(
            "Number of Semantic Scholar batch requests to run concurrently. "
            "Use 1 for the most conservative rate-limit behavior."
        ),
    )
    parser.add_argument(
        "--s2-sleep",
        type=float,
        default=1.0,
        help="Seconds to sleep between Semantic Scholar API calls.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Retries for transient HTTP errors.",
    )
    parser.add_argument(
        "--env-path",
        type=Path,
        default=Path(".env"),
        help="Path to a .env file containing S2_API_KEY.",
    )
    parser.add_argument(
        "--allow-missing-s2-key",
        action="store_true",
        help="Run without S2_API_KEY. This is usually slower and more rate-limited.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/arxiv_citation_graph"),
        help="Directory for nodes.jsonl, edges.jsonl, graph.json, and manifest.json.",
    )
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=15.0,
        help=(
            "Minimum seconds between non-forced progress updates. Network "
            "batch completions and retry sleeps are always reported."
        ),
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable timestamped progress updates on stderr.",
    )
    return parser.parse_args()


def cli_flag_present(flag: str) -> bool:
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in sys.argv[1:])


def apply_graph_size_preset(args: argparse.Namespace) -> None:
    if args.graph_size_preset == "none":
        return

    if args.graph_size_preset == "domain-10k":
        preset_values = {
            "--target-node-count": ("target_node_count", 10_000),
            "--citation-depth": ("citation_depth", 2),
            "--max-references-per-paper": ("max_references_per_paper", 40),
            "--max-expansion-papers-per-depth": ("max_expansion_papers_per_depth", 2_500),
            "--s2-batch-size": ("s2_batch_size", 500),
            "--s2-concurrency": ("s2_concurrency", 4),
        }
    else:  # pragma: no cover - argparse choices should prevent this
        raise SystemExit(f"Unknown graph size preset: {args.graph_size_preset}")

    for flag, (attribute, value) in preset_values.items():
        if not cli_flag_present(flag):
            setattr(args, attribute, value)


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value


def clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split())
    return text or None


def strip_arxiv_version(arxiv_id: str | None) -> str | None:
    if not arxiv_id:
        return None
    return re.sub(r"v\d+$", "", arxiv_id.strip())


def extract_arxiv_id(entry_id: str | None) -> str | None:
    if not entry_id:
        return None
    if "/abs/" in entry_id:
        arxiv_id = entry_id.rsplit("/abs/", 1)[1]
    else:
        arxiv_id = entry_id.rstrip("/").rsplit("/", 1)[-1]
    return strip_arxiv_version(arxiv_id)


def first_arxiv_category(query: str) -> str | None:
    match = re.search(r"\bcat:([A-Za-z0-9_.-]+)", query)
    return match.group(1) if match else None


def split_csv_option(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def external_id_from_mapping(external_ids: Any, name: str) -> str | None:
    if not isinstance(external_ids, dict):
        return None
    for key, value in external_ids.items():
        if key.lower() == name.lower() and value:
            return str(value)
    return None


def derive_s2_search_query(arxiv_query: str) -> str:
    category = first_arxiv_category(arxiv_query)
    if category and category in ARXIV_CATEGORY_TO_S2_DEFAULTS:
        return ARXIV_CATEGORY_TO_S2_DEFAULTS[category][1]

    query = re.sub(r"\bcat:[A-Za-z0-9_.-]+", " ", arxiv_query)
    query = re.sub(r"\b(?:all|ti|au|abs):", " ", query)
    query = re.sub(r"\b(?:AND|OR|ANDNOT)\b", " ", query, flags=re.IGNORECASE)
    query = re.sub(r"[()\"]+", " ", query)
    query = " ".join(query.split())
    if query:
        return query
    return "machine learning"


def derive_s2_fields_of_study(arxiv_query: str) -> list[str]:
    category = first_arxiv_category(arxiv_query)
    if category and category in ARXIV_CATEGORY_TO_S2_DEFAULTS:
        return [ARXIV_CATEGORY_TO_S2_DEFAULTS[category][0]]
    return []


def child_text(element: ET.Element, tag: str) -> str | None:
    child = element.find(tag)
    if child is None:
        return None
    return clean_text(child.text)


def parse_arxiv_feed(xml_bytes: bytes) -> list[dict[str, Any]]:
    root = ET.fromstring(xml_bytes)
    parsed: list[dict[str, Any]] = []
    for entry in root.findall(f"{ATOM_NS}entry"):
        entry_id = child_text(entry, f"{ATOM_NS}id")
        arxiv_id = extract_arxiv_id(entry_id)
        if not arxiv_id:
            continue

        primary_category = None
        primary_category_el = entry.find(f"{ARXIV_NS}primary_category")
        if primary_category_el is not None:
            primary_category = primary_category_el.attrib.get("term")

        categories = [
            category.attrib["term"]
            for category in entry.findall(f"{ATOM_NS}category")
            if category.attrib.get("term")
        ]
        authors = [
            clean_text(author.findtext(f"{ATOM_NS}name"))
            for author in entry.findall(f"{ATOM_NS}author")
        ]

        parsed.append(
            {
                "arxiv_id": arxiv_id,
                "entry_id": entry_id,
                "title": child_text(entry, f"{ATOM_NS}title"),
                "summary": child_text(entry, f"{ATOM_NS}summary"),
                "published": child_text(entry, f"{ATOM_NS}published"),
                "updated": child_text(entry, f"{ATOM_NS}updated"),
                "primary_category": primary_category,
                "categories": categories,
                "authors": [author for author in authors if author],
                "doi": child_text(entry, f"{ARXIV_NS}doi"),
            }
        )
    return parsed


def with_retry_jitter(seconds: float, retry_jitter: float) -> float:
    if seconds <= 0 or retry_jitter <= 0:
        return seconds
    return seconds + random.uniform(0.0, seconds * retry_jitter)


def wait_for_local_rate_limit_slot(
    *,
    lock_path: Path | None,
    min_interval_seconds: float,
    progress: ProgressReporter | None,
    request_label: str,
) -> None:
    if lock_path is None or min_interval_seconds <= 0 or fcntl is None:
        return

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            lock_file.seek(0)
            raw_timestamp = lock_file.read().strip()
            try:
                last_request_at = float(raw_timestamp) if raw_timestamp else 0.0
            except ValueError:
                last_request_at = 0.0

            elapsed = time.time() - last_request_at
            sleep_for = min_interval_seconds - elapsed
            if sleep_for > 0:
                if progress:
                    progress.log(
                        f"{request_label}: local arXiv throttle sleeping "
                        f"{format_duration(sleep_for)}"
                    )
                time.sleep(sleep_for)

            lock_file.seek(0)
            lock_file.truncate()
            lock_file.write(str(time.time()))
            lock_file.flush()
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def retry_after_seconds(
    response: requests.Response,
    attempt: int,
    *,
    rate_limit_base_sleep: float = RATE_LIMIT_BASE_SLEEP_SECONDS,
    rate_limit_max_sleep: float = RATE_LIMIT_MAX_SLEEP_SECONDS,
    transient_max_sleep: float = TRANSIENT_MAX_SLEEP_SECONDS,
    retry_jitter: float = 0.0,
) -> float:
    retry_after = response.headers.get("Retry-After")
    if retry_after:
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            try:
                retry_after_date = email.utils.parsedate_to_datetime(retry_after)
            except (TypeError, ValueError):
                retry_after_date = None
            if retry_after_date is not None:
                if retry_after_date.tzinfo is None:
                    retry_after_date = retry_after_date.replace(tzinfo=dt.timezone.utc)
                delay = (retry_after_date - dt.datetime.now(dt.timezone.utc)).total_seconds()
                if delay > 0:
                    return delay

    if response.status_code == 429:
        sleep_for = min(rate_limit_max_sleep, rate_limit_base_sleep * (2.0**attempt))
        return with_retry_jitter(sleep_for, retry_jitter)
    sleep_for = min(transient_max_sleep, 2.0**attempt)
    return with_retry_jitter(sleep_for, retry_jitter)


def request_with_retries(
    session: requests.Session,
    method: str,
    url: str,
    *,
    max_retries: int,
    timeout: float,
    progress: ProgressReporter | None = None,
    request_label: str | None = None,
    rate_limit_base_sleep: float = RATE_LIMIT_BASE_SLEEP_SECONDS,
    rate_limit_max_sleep: float = RATE_LIMIT_MAX_SLEEP_SECONDS,
    transient_max_sleep: float = TRANSIENT_MAX_SLEEP_SECONDS,
    retry_jitter: float = 0.0,
    before_attempt: Callable[[], None] | None = None,
    **kwargs: Any,
) -> requests.Response:
    label = request_label or f"{method} {url}"
    for attempt in range(max_retries + 1):
        if before_attempt:
            before_attempt()
        try:
            response = session.request(method, url, timeout=timeout, **kwargs)
        except requests.RequestException as exc:
            if attempt >= max_retries:
                raise
            sleep_for = with_retry_jitter(
                min(transient_max_sleep, 2.0**attempt),
                retry_jitter,
            )
            if progress:
                progress.log(
                    f"{label}: attempt {attempt + 1}/{max_retries + 1} failed "
                    f"with {exc.__class__.__name__}; retrying in {format_duration(sleep_for)}"
            )
            time.sleep(sleep_for)
            continue

        if response.status_code not in RETRYABLE_STATUS_CODES:
            response.raise_for_status()
            return response

        if attempt >= max_retries:
            response.raise_for_status()

        sleep_for = retry_after_seconds(
            response,
            attempt,
            rate_limit_base_sleep=rate_limit_base_sleep,
            rate_limit_max_sleep=rate_limit_max_sleep,
            transient_max_sleep=transient_max_sleep,
            retry_jitter=retry_jitter,
        )
        if progress:
            progress.log(
                f"{label}: HTTP {response.status_code} on attempt "
                f"{attempt + 1}/{max_retries + 1}; retrying in {format_duration(sleep_for)}"
            )
        time.sleep(sleep_for)

    raise RuntimeError("unreachable retry loop")


def fetch_arxiv_candidates(
    args: argparse.Namespace,
    progress: ProgressReporter,
) -> list[dict[str, Any]]:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    rate_limit_lock_path = None if args.no_arxiv_rate_limit_lock else args.arxiv_rate_limit_lock_path

    candidates: list[dict[str, Any]] = []
    seen_arxiv_ids: set[str] = set()
    start = 0
    batch_number = 0
    progress.start_stage(
        "arXiv candidate fetch",
        total=args.max_arxiv_results,
        unit="candidates",
        detail=f"query={args.query!r}",
    )
    if args.arxiv_initial_sleep > 0:
        progress.log(
            f"arXiv candidate fetch: initial cooldown sleeping "
            f"{format_duration(args.arxiv_initial_sleep)}"
        )
        time.sleep(args.arxiv_initial_sleep)

    while len(candidates) < args.max_arxiv_results:
        batch_number += 1
        max_results = min(args.arxiv_batch_size, args.max_arxiv_results - len(candidates))
        request_label = f"arXiv batch {batch_number} start={start}"
        params = {
            "search_query": args.query,
            "start": start,
            "max_results": max_results,
            "sortBy": args.sort_by,
            "sortOrder": args.sort_order,
        }
        response = request_with_retries(
            session,
            "GET",
            ARXIV_API_URL,
            params=params,
            max_retries=args.max_retries,
            timeout=args.timeout,
            progress=progress,
            request_label=request_label,
            rate_limit_base_sleep=args.arxiv_rate_limit_base_sleep,
            rate_limit_max_sleep=args.arxiv_rate_limit_max_sleep,
            retry_jitter=args.arxiv_retry_jitter,
            before_attempt=lambda: wait_for_local_rate_limit_slot(
                lock_path=rate_limit_lock_path,
                min_interval_seconds=args.arxiv_sleep,
                progress=progress,
                request_label=request_label,
            ),
        )
        batch = parse_arxiv_feed(response.content)
        if not batch:
            progress.log(f"arXiv batch {batch_number}: API returned no records; stopping")
            break

        added = 0
        for record in batch:
            arxiv_id = record["arxiv_id"]
            if arxiv_id in seen_arxiv_ids:
                continue
            candidates.append(record)
            seen_arxiv_ids.add(arxiv_id)
            added += 1

        duplicate_count = len(batch) - added
        progress.update(
            len(candidates),
            total=args.max_arxiv_results,
            detail=(
                f"batch={batch_number}, start={start}, returned={len(batch)}, "
                f"added={added}, duplicates={duplicate_count}"
            ),
            force=True,
        )
        if added == 0 or len(batch) < max_results:
            if added == 0:
                progress.log(f"arXiv batch {batch_number}: no new candidates; stopping")
            else:
                progress.log(
                    f"arXiv batch {batch_number}: short batch "
                    f"({len(batch)}/{max_results}); stopping"
                )
            break
        start += len(batch)
        if len(candidates) < args.max_arxiv_results:
            time.sleep(args.arxiv_sleep)

    progress.finish_stage(completed=len(candidates), total=args.max_arxiv_results)
    return candidates


class SemanticScholarClient:
    def __init__(
        self,
        api_key: str | None,
        timeout: float,
        max_retries: int,
        progress: ProgressReporter,
    ) -> None:
        self.timeout = timeout
        self.max_retries = max_retries
        self.progress = progress
        self.headers = {"User-Agent": USER_AGENT}
        if api_key:
            self.headers["x-api-key"] = api_key

    def make_session(self) -> requests.Session:
        session = requests.Session()
        session.headers.update(self.headers)
        return session

    def paper_batch(
        self,
        ids: list[str],
        *,
        request_label: str | None = None,
    ) -> list[dict[str, Any] | None]:
        response = request_with_retries(
            self.make_session(),
            "POST",
            f"{S2_GRAPH_API_URL}/paper/batch",
            params={"fields": S2_PAPER_FIELDS},
            json={"ids": ids},
            max_retries=self.max_retries,
            timeout=self.timeout,
            progress=self.progress,
            request_label=request_label,
        )
        data = response.json()
        if not isinstance(data, list):
            raise RuntimeError(f"Unexpected Semantic Scholar batch response: {data!r}")
        return data

    def paper_search_bulk(
        self,
        *,
        params: dict[str, Any],
        request_label: str | None = None,
    ) -> dict[str, Any]:
        response = request_with_retries(
            self.make_session(),
            "GET",
            f"{S2_GRAPH_API_URL}/paper/search/bulk",
            params=params,
            max_retries=self.max_retries,
            timeout=self.timeout,
            progress=self.progress,
            request_label=request_label,
        )
        data = response.json()
        if not isinstance(data, dict):
            raise RuntimeError(f"Unexpected Semantic Scholar search response: {data!r}")
        return data


def s2_paper_to_arxiv_candidate(paper: dict[str, Any]) -> dict[str, Any] | None:
    arxiv_id = strip_arxiv_version(external_id_from_mapping(paper.get("externalIds"), "arxiv"))
    if not arxiv_id:
        return None

    authors = []
    for author in paper.get("authors") or []:
        if isinstance(author, dict) and author.get("name"):
            authors.append(str(author["name"]))

    publication_date = paper.get("publicationDate")
    if not publication_date and paper.get("year"):
        publication_date = f"{paper['year']}-01-01"

    s2_fields = paper.get("s2FieldsOfStudy") or []
    categories = []
    if isinstance(s2_fields, list):
        for field in s2_fields:
            if isinstance(field, dict) and field.get("category"):
                categories.append(str(field["category"]))
            elif isinstance(field, str):
                categories.append(field)
    if not categories:
        raw_fields = paper.get("fieldsOfStudy") or []
        if isinstance(raw_fields, list):
            categories = [str(field) for field in raw_fields if field]

    return {
        "arxiv_id": arxiv_id,
        "entry_id": f"https://arxiv.org/abs/{arxiv_id}",
        "title": clean_text(paper.get("title")),
        "summary": clean_text(paper.get("abstract")),
        "published": publication_date,
        "updated": None,
        "primary_category": categories[0] if categories else None,
        "categories": categories,
        "authors": authors,
        "doi": external_id_from_mapping(paper.get("externalIds"), "doi"),
        "candidate_source": "s2-search",
        "s2_paper_id": paper.get("paperId"),
        "s2_corpus_id": paper.get("corpusId"),
        "s2_citation_count": paper.get("citationCount"),
    }


def fetch_s2_search_arxiv_candidates(
    args: argparse.Namespace,
    client: SemanticScholarClient,
    progress: ProgressReporter,
) -> list[dict[str, Any]]:
    search_query = args.s2_search_query or derive_s2_search_query(args.query)
    fields_of_study = split_csv_option(args.s2_fields_of_study)
    if args.s2_fields_of_study is None:
        fields_of_study = derive_s2_fields_of_study(args.query)
    publication_types = split_csv_option(args.s2_publication_types)

    candidates: list[dict[str, Any]] = []
    seen_arxiv_ids: set[str] = set()
    token: str | None = None
    page_number = 0

    progress.start_stage(
        "Semantic Scholar candidate search",
        total=args.max_arxiv_results,
        unit="arXiv-backed candidates",
        detail=(
            f"query={search_query!r}, fields_of_study={fields_of_study or 'none'}, "
            f"sort={args.s2_search_sort}"
        ),
    )

    while len(candidates) < args.max_arxiv_results and page_number < args.s2_search_max_pages:
        page_number += 1
        params: dict[str, Any] = {
            "query": search_query,
            "fields": S2_SEARCH_FIELDS,
            "sort": args.s2_search_sort,
            "limit": args.s2_search_batch_size,
        }
        if token:
            params["token"] = token
        if fields_of_study:
            params["fieldsOfStudy"] = ",".join(fields_of_study)
        if publication_types:
            params["publicationTypes"] = ",".join(publication_types)
        if args.min_citations > 0:
            params["minCitationCount"] = str(args.min_citations)
        if args.min_year is not None or args.max_year is not None:
            start_year = args.min_year if args.min_year is not None else ""
            end_year = args.max_year if args.max_year is not None else ""
            params["year"] = f"{start_year}-{end_year}"

        if page_number > 1 and args.s2_sleep > 0:
            time.sleep(args.s2_sleep)

        data = client.paper_search_bulk(
            params=params,
            request_label=f"Semantic Scholar candidate search page {page_number}",
        )
        records = data.get("data")
        if not isinstance(records, list) or not records:
            progress.log(f"Semantic Scholar candidate search page {page_number}: no records; stopping")
            break

        added = 0
        missing_arxiv_id = 0
        duplicate_count = 0
        for paper in records:
            if not isinstance(paper, dict):
                continue
            candidate = s2_paper_to_arxiv_candidate(paper)
            if not candidate:
                missing_arxiv_id += 1
                continue
            arxiv_id = candidate["arxiv_id"]
            if arxiv_id in seen_arxiv_ids:
                duplicate_count += 1
                continue
            candidates.append(candidate)
            seen_arxiv_ids.add(arxiv_id)
            added += 1
            if len(candidates) >= args.max_arxiv_results:
                break

        progress.update(
            len(candidates),
            total=args.max_arxiv_results,
            detail=(
                f"page={page_number}, returned={len(records)}, added={added}, "
                f"missing_arxiv={missing_arxiv_id}, duplicates={duplicate_count}"
            ),
            force=True,
        )

        token_value = data.get("token")
        token = str(token_value) if token_value else None
        if not token:
            progress.log("Semantic Scholar candidate search: no continuation token; stopping")
            break
        if added == 0 and missing_arxiv_id == len(records):
            progress.log(
                "Semantic Scholar candidate search: page had no arXiv-backed records; continuing"
            )

    progress.finish_stage(completed=len(candidates), total=args.max_arxiv_results)
    return candidates


def chunks(values: list[str], size: int) -> list[list[str]]:
    return [values[start : start + size] for start in range(0, len(values), size)]


def fetch_s2_batch_map(
    client: SemanticScholarClient,
    requested_ids: list[str],
    batch_size: int,
    sleep_seconds: float,
    progress: ProgressReporter,
    *,
    stage_name: str,
    s2_concurrency: int,
) -> dict[str, dict[str, Any]]:
    unique_ids = list(dict.fromkeys(requested_id for requested_id in requested_ids if requested_id))
    papers: dict[str, dict[str, Any]] = {}
    if not unique_ids:
        return papers

    batches = chunks(unique_ids, batch_size)
    progress.start_stage(
        stage_name,
        total=len(unique_ids),
        unit="papers",
        detail=(
            f"{len(batches):,} batches of up to {batch_size:,}; "
            f"s2_concurrency={s2_concurrency}"
        ),
    )

    def process_batch(batch_index: int, batch: list[str]) -> tuple[int, list[str], list[dict[str, Any] | None]]:
        data = client.paper_batch(
            batch,
            request_label=f"{stage_name} batch {batch_index}/{len(batches)}",
        )
        return batch_index, batch, data

    completed = 0
    if s2_concurrency == 1:
        for batch_index, batch in enumerate(batches, start=1):
            _, completed_batch, data = process_batch(batch_index, batch)
            matched_in_batch = 0
            for requested_id, paper in zip(completed_batch, data):
                if paper:
                    papers[requested_id] = paper
                    matched_in_batch += 1
            completed += len(completed_batch)
            progress.update(
                completed,
                total=len(unique_ids),
                detail=(
                    f"batch={batch_index}/{len(batches)}, matched_batch={matched_in_batch}, "
                    f"matched_total={len(papers)}"
                ),
                force=True,
            )
            if completed < len(unique_ids):
                time.sleep(sleep_seconds)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=s2_concurrency) as executor:
            future_to_batch = {
                executor.submit(process_batch, batch_index, batch): (batch_index, batch)
                for batch_index, batch in enumerate(batches, start=1)
            }
            for future in concurrent.futures.as_completed(future_to_batch):
                batch_index, _ = future_to_batch[future]
                _, completed_batch, data = future.result()
                matched_in_batch = 0
                for requested_id, paper in zip(completed_batch, data):
                    if paper:
                        papers[requested_id] = paper
                        matched_in_batch += 1
                completed += len(completed_batch)
                progress.update(
                    completed,
                    total=len(unique_ids),
                    detail=(
                        f"completed_batch={batch_index}/{len(batches)}, "
                        f"matched_batch={matched_in_batch}, matched_total={len(papers)}"
                    ),
                    force=True,
                )

    progress.finish_stage(
        completed=len(unique_ids),
        total=len(unique_ids),
        detail=f"matched {len(papers):,} papers",
    )
    return papers


def fetch_s2_papers(
    client: SemanticScholarClient,
    arxiv_ids: list[str],
    batch_size: int,
    sleep_seconds: float,
    progress: ProgressReporter,
    s2_concurrency: int,
) -> dict[str, dict[str, Any]]:
    requested_ids = [f"ARXIV:{arxiv_id}" for arxiv_id in arxiv_ids]
    return fetch_s2_batch_map(
        client,
        requested_ids,
        batch_size,
        sleep_seconds,
        progress,
        stage_name="Semantic Scholar enrichment",
        s2_concurrency=s2_concurrency,
    )


def fetch_s2_papers_by_lookup_ids(
    client: SemanticScholarClient,
    lookup_ids: list[str],
    batch_size: int,
    sleep_seconds: float,
    progress: ProgressReporter,
    *,
    stage_name: str,
    s2_concurrency: int,
) -> dict[str, dict[str, Any]]:
    return fetch_s2_batch_map(
        client,
        lookup_ids,
        batch_size,
        sleep_seconds,
        progress,
        stage_name=stage_name,
        s2_concurrency=s2_concurrency,
    )


def as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def external_id(paper: dict[str, Any], name: str) -> str | None:
    external_ids = paper.get("externalIds") or {}
    for key, value in external_ids.items():
        if key.lower() == name.lower() and value:
            if name.lower() == "arxiv":
                return strip_arxiv_version(str(value))
            return str(value)
    return None


def s2_lookup_id_for_paper(paper: dict[str, Any]) -> str | None:
    paper_id = paper.get("paperId")
    if paper_id:
        return str(paper_id)

    arxiv_id = external_id(paper, "arxiv")
    if arxiv_id:
        return f"ARXIV:{arxiv_id}"

    doi = external_id(paper, "doi")
    if doi:
        return f"DOI:{doi}"

    corpus_id = paper.get("corpusId")
    if corpus_id:
        return f"CorpusId:{corpus_id}"

    return None


def publication_year(paper: dict[str, Any], arxiv_entry: dict[str, Any] | None = None) -> int | None:
    year = as_int(paper.get("year"))
    if year:
        return year
    date_text = paper.get("publicationDate")
    if isinstance(date_text, str) and len(date_text) >= 4:
        return as_int(date_text[:4])
    if arxiv_entry:
        published = arxiv_entry.get("published")
        if isinstance(published, str) and len(published) >= 4:
            return as_int(published[:4])
    return None


def parse_date(value: Any) -> dt.date | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return dt.date.fromisoformat(value[:10])
    except ValueError:
        return None


def paper_rank_key(paper: dict[str, Any]) -> tuple[int, int, int, int, str]:
    return (
        as_int(paper.get("citationCount")) or 0,
        as_int(paper.get("influentialCitationCount")) or 0,
        as_int(paper.get("referenceCount")) or 0,
        publication_year(paper) or 0,
        clean_text(paper.get("title")) or "",
    )


def reference_list_available(paper: dict[str, Any]) -> bool:
    return isinstance(paper.get("references"), list)


def safe_arxiv_html_filename(arxiv_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", arxiv_id).strip("_") or "unknown"


def arxiv_html_url(arxiv_id: str) -> str:
    return ARXIV_HTML_URL_TEMPLATE.format(arxiv_id=strip_arxiv_version(arxiv_id) or arxiv_id)


def resolve_html_dir(output_dir: Path, html_dir: Path) -> Path:
    if html_dir.is_absolute():
        return html_dir
    return output_dir / html_dir


def relative_to_output_dir(path: Path, output_dir: Path) -> str:
    try:
        return str(path.relative_to(output_dir))
    except ValueError:
        return str(path)


def fetch_arxiv_html_pages(
    nodes: list[dict[str, Any]],
    seed_node_ids: set[str],
    args: argparse.Namespace,
    progress: ProgressReporter,
) -> list[dict[str, Any]]:
    html_dir = resolve_html_dir(args.output_dir, args.arxiv_html_dir)
    html_dir.mkdir(parents=True, exist_ok=True)

    node_ids_by_arxiv_id: dict[str, list[str]] = {}
    nodes_by_id = {node["node_id"]: node for node in nodes}
    for node in nodes:
        raw_arxiv_id = node.get("arxiv_id")
        arxiv_id = strip_arxiv_version(str(raw_arxiv_id)) if raw_arxiv_id else None
        if not arxiv_id:
            continue
        if args.arxiv_html_scope == "seed" and node["node_id"] not in seed_node_ids:
            continue
        node_ids_by_arxiv_id.setdefault(arxiv_id, []).append(node["node_id"])

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    rate_limit_lock_path = None if args.no_arxiv_rate_limit_lock else args.arxiv_rate_limit_lock_path
    records: list[dict[str, Any]] = []
    progress.start_stage(
        "arXiv HTML fetch",
        total=len(node_ids_by_arxiv_id),
        unit="papers",
        detail=f"scope={args.arxiv_html_scope}, dir={html_dir}",
    )

    for index, (arxiv_id, node_ids) in enumerate(node_ids_by_arxiv_id.items(), start=1):
        url = arxiv_html_url(arxiv_id)
        output_path = html_dir / f"{safe_arxiv_html_filename(arxiv_id)}.html"
        relative_path = relative_to_output_dir(output_path, args.output_dir)
        record: dict[str, Any] = {
            "arxiv_id": arxiv_id,
            "url": url,
            "path": relative_path,
            "node_ids": node_ids,
        }

        try:
            if output_path.exists() and not args.overwrite_arxiv_html:
                record.update(
                    {
                        "status": "exists",
                        "http_status": None,
                        "bytes": output_path.stat().st_size,
                    }
                )
            else:
                request_label = f"arXiv HTML {arxiv_id}"
                response = request_with_retries(
                    session,
                    "GET",
                    url,
                    max_retries=args.max_retries,
                    timeout=args.timeout,
                    progress=progress,
                    request_label=request_label,
                    rate_limit_base_sleep=args.arxiv_rate_limit_base_sleep,
                    rate_limit_max_sleep=args.arxiv_rate_limit_max_sleep,
                    retry_jitter=args.arxiv_retry_jitter,
                    before_attempt=lambda: wait_for_local_rate_limit_slot(
                        lock_path=rate_limit_lock_path,
                        min_interval_seconds=args.arxiv_sleep,
                        progress=progress,
                        request_label=request_label,
                    ),
                )
                output_path.write_bytes(response.content)
                record.update(
                    {
                        "status": "saved",
                        "http_status": response.status_code,
                        "bytes": len(response.content),
                    }
                )
        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            record.update(
                {
                    "status": "error",
                    "http_status": status_code,
                    "bytes": 0,
                    "error": str(exc),
                }
            )
        except requests.RequestException as exc:
            record.update(
                {
                    "status": "error",
                    "http_status": None,
                    "bytes": 0,
                    "error": f"{exc.__class__.__name__}: {exc}",
                }
            )

        for node_id in node_ids:
            node = nodes_by_id[node_id]
            node["arxiv_html_url"] = url
            node["arxiv_html_path"] = relative_path
            node["arxiv_html_status"] = record["status"]

        records.append(record)
        progress.update(
            index,
            total=len(node_ids_by_arxiv_id),
            detail=f"status={record['status']}, bytes={record.get('bytes', 0):,}",
            force=True,
        )

    status_counts: dict[str, int] = {}
    for record in records:
        status = str(record.get("status"))
        status_counts[status] = status_counts.get(status, 0) + 1
    progress.finish_stage(
        completed=len(records),
        total=len(node_ids_by_arxiv_id),
        detail=f"statuses={json.dumps(status_counts, sort_keys=True)}",
    )
    return records


def temporal_edge_allowed(
    cited_paper: dict[str, Any],
    citing_paper: dict[str, Any],
    allow_nontemporal_edges: bool,
) -> bool:
    if allow_nontemporal_edges:
        return True

    cited_year = publication_year(cited_paper)
    citing_year = publication_year(citing_paper)
    if cited_year is not None and citing_year is not None and cited_year > citing_year:
        return False

    cited_date = parse_date(cited_paper.get("publicationDate"))
    citing_date = parse_date(citing_paper.get("publicationDate"))
    if cited_date is not None and citing_date is not None and cited_date > citing_date:
        return False

    return True


def node_id_for_paper(
    paper: dict[str, Any],
    arxiv_entry: dict[str, Any] | None = None,
) -> str | None:
    paper_id = paper.get("paperId")
    if paper_id:
        return f"s2:{paper_id}"

    arxiv_id = external_id(paper, "arxiv")
    if not arxiv_id and arxiv_entry:
        arxiv_id = arxiv_entry.get("arxiv_id")
    if arxiv_id:
        return f"arxiv:{arxiv_id}"

    doi = external_id(paper, "doi")
    if doi:
        return f"doi:{doi.lower()}"

    corpus_id = paper.get("corpusId")
    if corpus_id:
        return f"s2-corpus:{corpus_id}"

    title = clean_text(paper.get("title"))
    year = publication_year(paper, arxiv_entry)
    if not title:
        return None
    digest = hashlib.sha1(f"{title}|{year}".encode("utf-8")).hexdigest()[:16]
    return f"title:{digest}"


def paper_authors(paper: dict[str, Any], arxiv_entry: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    authors = paper.get("authors")
    if isinstance(authors, list) and authors:
        parsed = []
        for author in authors:
            if not isinstance(author, dict):
                continue
            parsed.append(
                {
                    "author_id": author.get("authorId"),
                    "name": clean_text(author.get("name")),
                }
            )
        return parsed
    if arxiv_entry:
        return [{"author_id": None, "name": name} for name in arxiv_entry.get("authors", [])]
    return []


def make_node(
    paper: dict[str, Any],
    role: str,
    arxiv_entry: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    node_id = node_id_for_paper(paper, arxiv_entry)
    if not node_id:
        return None

    arxiv_id = external_id(paper, "arxiv")
    if not arxiv_id and arxiv_entry:
        arxiv_id = arxiv_entry.get("arxiv_id")

    return {
        "node_id": node_id,
        "roles": [role],
        "paper_id": paper.get("paperId"),
        "corpus_id": paper.get("corpusId"),
        "arxiv_id": arxiv_id,
        "doi": external_id(paper, "doi"),
        "title": clean_text(paper.get("title")) or (arxiv_entry or {}).get("title"),
        "abstract": clean_text(paper.get("abstract")) or (arxiv_entry or {}).get("summary"),
        "year": publication_year(paper, arxiv_entry),
        "publication_date": paper.get("publicationDate") or (arxiv_entry or {}).get("published"),
        "venue": clean_text(paper.get("venue")),
        "url": paper.get("url") or (arxiv_entry or {}).get("entry_id"),
        "citation_count": as_int(paper.get("citationCount")),
        "influential_citation_count": as_int(paper.get("influentialCitationCount")),
        "reference_count": as_int(paper.get("referenceCount")),
        "authors": paper_authors(paper, arxiv_entry),
        "external_ids": paper.get("externalIds") or {},
        "fields_of_study": paper.get("fieldsOfStudy") or [],
        "s2_fields_of_study": paper.get("s2FieldsOfStudy") or [],
        "publication_types": paper.get("publicationTypes") or [],
        "arxiv_primary_category": (arxiv_entry or {}).get("primary_category"),
        "arxiv_categories": (arxiv_entry or {}).get("categories", []),
    }


def upsert_node(
    nodes: dict[str, dict[str, Any]],
    paper: dict[str, Any],
    role: str,
    arxiv_entry: dict[str, Any] | None = None,
) -> str | None:
    node = make_node(paper, role, arxiv_entry)
    if node is None:
        return None
    node_id = node["node_id"]
    existing = nodes.get(node_id)
    if existing is None:
        nodes[node_id] = node
        return node_id

    if role not in existing["roles"]:
        existing["roles"].append(role)
        existing["roles"].sort()
    for key in ["abstract", "arxiv_id", "doi", "publication_date", "url", "venue"]:
        if not existing.get(key) and node.get(key):
            existing[key] = node[key]
    if not existing.get("arxiv_categories") and node.get("arxiv_categories"):
        existing["arxiv_categories"] = node["arxiv_categories"]
        existing["arxiv_primary_category"] = node["arxiv_primary_category"]
    return node_id


def add_node_role(nodes: dict[str, dict[str, Any]], node_id: str, role: str) -> None:
    node = nodes.get(node_id)
    if node is None:
        return
    if role not in node["roles"]:
        node["roles"].append(role)
        node["roles"].sort()


def set_node_depth(nodes: dict[str, dict[str, Any]], node_id: str, depth: int) -> None:
    node = nodes.get(node_id)
    if node is None:
        return
    existing_depth = node.get("seed_reference_depth")
    if existing_depth is None or depth < existing_depth:
        node["seed_reference_depth"] = depth


def value_is_missing(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def merge_paper_metadata_into_node(
    nodes: dict[str, dict[str, Any]],
    node_id: str,
    paper: dict[str, Any],
    role: str,
    arxiv_entry: dict[str, Any] | None = None,
) -> None:
    existing = nodes.get(node_id)
    if existing is None:
        node = make_node(paper, role, arxiv_entry)
        if node is not None:
            node["node_id"] = node_id
            nodes[node_id] = node
        return

    add_node_role(nodes, node_id, role)
    node = make_node(paper, role, arxiv_entry)
    if node is None:
        return
    for key, value in node.items():
        if key in {"node_id", "roles"}:
            continue
        if value_is_missing(existing.get(key)) and not value_is_missing(value):
            existing[key] = value


def unwrap_reference(reference: Any) -> dict[str, Any] | None:
    if not isinstance(reference, dict):
        return None
    nested = reference.get("citedPaper")
    if isinstance(nested, dict):
        return nested
    return reference


def limited_reference_papers(
    references: list[Any],
    max_references_per_paper: int,
    skipped: dict[str, int],
) -> list[dict[str, Any]]:
    papers: list[dict[str, Any]] = []
    for reference_record in references:
        cited_paper = unwrap_reference(reference_record)
        if not cited_paper:
            skipped["references_without_id"] += 1
            continue
        papers.append(cited_paper)

    if max_references_per_paper > 0 and len(papers) > max_references_per_paper:
        papers.sort(key=paper_rank_key, reverse=True)
        skipped["references_breadth_cap"] += len(papers) - max_references_per_paper
        papers = papers[:max_references_per_paper]
    return papers


def select_frontier_node_ids(
    node_ids: set[str],
    papers_by_node_id: dict[str, dict[str, Any]],
    max_expansion_papers_per_depth: int | None,
    root_seed_by_node_id: dict[str, str],
    skipped: dict[str, int],
) -> list[str]:
    if max_expansion_papers_per_depth is None:
        return sorted(
            node_ids,
            key=lambda node_id: paper_rank_key(papers_by_node_id.get(node_id, {})),
            reverse=True,
        )

    if max_expansion_papers_per_depth <= 0:
        skipped["expansion_breadth_cap"] += len(node_ids)
        return []

    if len(node_ids) <= max_expansion_papers_per_depth:
        return sorted(
            node_ids,
            key=lambda node_id: paper_rank_key(papers_by_node_id.get(node_id, {})),
            reverse=True,
        )

    by_root: dict[str, list[str]] = {}
    for node_id in node_ids:
        root_id = root_seed_by_node_id.get(node_id, node_id)
        by_root.setdefault(root_id, []).append(node_id)

    for root_ids in by_root.values():
        root_ids.sort(key=lambda node_id: paper_rank_key(papers_by_node_id.get(node_id, {})), reverse=True)

    selected: list[str] = []
    root_order = sorted(
        by_root,
        key=lambda root_id: paper_rank_key(papers_by_node_id.get(root_id, {})),
        reverse=True,
    )
    root_offsets = {root_id: 0 for root_id in root_order}
    while len(selected) < max_expansion_papers_per_depth:
        added_this_round = 0
        for root_id in root_order:
            offset = root_offsets[root_id]
            candidates = by_root[root_id]
            if offset >= len(candidates):
                continue
            selected.append(candidates[offset])
            root_offsets[root_id] = offset + 1
            added_this_round += 1
            if len(selected) >= max_expansion_papers_per_depth:
                break
        if added_this_round == 0:
            break

    skipped["expansion_breadth_cap"] += len(node_ids) - len(selected)
    return selected


def effective_frontier_cap(args: argparse.Namespace, current_node_count: int) -> int | None:
    explicit_cap = (
        args.max_expansion_papers_per_depth
        if args.max_expansion_papers_per_depth > 0
        else None
    )
    if args.target_node_count <= 0 or args.max_references_per_paper <= 0:
        return explicit_cap

    remaining_nodes = args.target_node_count - current_node_count
    if remaining_nodes <= 0:
        return 0

    budget_cap = max(1, math.ceil(remaining_nodes / args.max_references_per_paper) * 3)
    if explicit_cap is None:
        return budget_cap
    return min(explicit_cap, budget_cap)


def add_seed_component_root(
    nodes: dict[str, dict[str, Any]],
    node_id: str,
    root_seed_id: str,
) -> None:
    node = nodes.get(node_id)
    if node is None:
        return
    roots = node.setdefault("seed_component_roots", [])
    if root_seed_id not in roots:
        roots.append(root_seed_id)
        roots.sort()


def annotate_connected_components(
    nodes: dict[str, dict[str, Any]],
    edges: dict[tuple[str, str, str], dict[str, Any]],
) -> dict[str, Any]:
    adjacency: dict[str, set[str]] = {node_id: set() for node_id in nodes}
    for edge in edges.values():
        source = edge["source"]
        target = edge["target"]
        if source in adjacency and target in adjacency:
            adjacency[source].add(target)
            adjacency[target].add(source)

    components: list[list[str]] = []
    seen: set[str] = set()
    for node_id in nodes:
        if node_id in seen:
            continue
        stack = [node_id]
        seen.add(node_id)
        component: list[str] = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in adjacency[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(component)

    components.sort(key=len, reverse=True)
    for component_id, component in enumerate(components):
        component_size = len(component)
        for node_id in component:
            nodes[node_id]["connected_component_id"] = component_id
            nodes[node_id]["connected_component_size"] = component_size

    return {
        "connected_component_count": len(components),
        "largest_connected_component_size": len(components[0]) if components else 0,
        "top_connected_component_sizes": [len(component) for component in components[:20]],
    }


def select_seed_papers(
    candidates: list[dict[str, Any]],
    s2_papers: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    progress: ProgressReporter,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    missing_s2 = 0
    below_min_citations = 0
    below_min_year = 0
    above_max_year = 0
    for entry in candidates:
        requested_id = f"ARXIV:{entry['arxiv_id']}"
        paper = s2_papers.get(requested_id)
        if not paper:
            missing_s2 += 1
            continue

        citation_count = as_int(paper.get("citationCount")) or 0
        if citation_count < args.min_citations:
            below_min_citations += 1
            continue

        year = publication_year(paper, entry)
        if args.min_year is not None and year is not None and year < args.min_year:
            below_min_year += 1
            continue
        if args.max_year is not None and year is not None and year > args.max_year:
            above_max_year += 1
            continue

        selected.append((entry, paper))

    eligible_count = len(selected)
    selected.sort(
        key=lambda item: (
            as_int(item[1].get("citationCount")) or 0,
            publication_year(item[1], item[0]) or 0,
            item[0]["arxiv_id"],
        ),
        reverse=True,
    )
    if args.max_seed_papers > 0:
        selected = selected[: args.max_seed_papers]
    progress.log(
        "Seed filtering: "
        f"eligible={eligible_count:,}, selected={len(selected):,}, "
        f"missing_s2={missing_s2:,}, "
        f"below_min_citations={below_min_citations:,}, "
        f"below_min_year={below_min_year:,}, above_max_year={above_max_year:,}"
    )
    return selected


def build_graph(
    seed_papers: list[tuple[dict[str, Any], dict[str, Any]]],
    args: argparse.Namespace,
    progress: ProgressReporter,
    client: SemanticScholarClient | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int], dict[str, Any]]:
    nodes: dict[str, dict[str, Any]] = {}
    edges: dict[tuple[str, str, str], dict[str, Any]] = {}
    skipped = {
        "references_without_id": 0,
        "non_arxiv_references": 0,
        "external_references": 0,
        "nontemporal_edges": 0,
        "duplicate_edges": 0,
        "self_edges": 0,
        "references_breadth_cap": 0,
        "expansion_breadth_cap": 0,
        "expansion_lookup_missing": 0,
        "expansion_fetch_missing": 0,
        "target_node_budget": 0,
    }
    graph_stats: dict[str, Any] = {
        "target_node_count": args.target_node_count,
        "citation_depth": args.citation_depth,
        "max_references_per_paper": args.max_references_per_paper,
        "max_expansion_papers_per_depth": args.max_expansion_papers_per_depth,
        "raw_references_seen": 0,
        "references_inspected": 0,
        "expanded_paper_count": 0,
        "expanded_depth_counts": {},
        "expansion_fetch_count": 0,
        "target_node_budget_reached": False,
    }

    seed_node_ids: set[str] = set()
    seed_node_id_by_arxiv_id: dict[str, str] = {}
    seed_entry_by_node_id: dict[str, dict[str, Any]] = {}
    papers_by_node_id: dict[str, dict[str, Any]] = {}
    lookup_id_by_node_id: dict[str, str] = {}
    node_depth_by_id: dict[str, int] = {}
    root_seed_by_node_id: dict[str, str] = {}

    progress.start_stage("Seed node indexing", total=len(seed_papers), unit="seeds")
    for arxiv_entry, paper in seed_papers:
        node_id = upsert_node(nodes, paper, "seed_arxiv", arxiv_entry)
        if node_id:
            seed_node_ids.add(node_id)
            seed_node_id_by_arxiv_id[arxiv_entry["arxiv_id"]] = node_id
            seed_entry_by_node_id[node_id] = arxiv_entry
            papers_by_node_id[node_id] = paper
            lookup_id = s2_lookup_id_for_paper(paper) or f"ARXIV:{arxiv_entry['arxiv_id']}"
            lookup_id_by_node_id[node_id] = lookup_id
            node_depth_by_id[node_id] = 0
            root_seed_by_node_id[node_id] = node_id
            set_node_depth(nodes, node_id, 0)
            add_seed_component_root(nodes, node_id, node_id)
    progress.finish_stage(
        completed=len(seed_papers),
        total=len(seed_papers),
        detail=f"seed_nodes={len(seed_node_ids):,}, nodes={len(nodes):,}",
    )

    expanded_node_ids: set[str] = set()
    current_frontier: dict[str, dict[str, Any]] = {
        node_id: papers_by_node_id[node_id] for node_id in seed_node_ids if node_id in papers_by_node_id
    }

    for depth_index in range(args.citation_depth):
        current_frontier = {
            node_id: paper
            for node_id, paper in current_frontier.items()
            if node_id not in expanded_node_ids
        }
        if not current_frontier:
            progress.log(f"Reference depth {depth_index + 1}: empty frontier; stopping expansion")
            break

        frontier_items = sorted(
            current_frontier.items(),
            key=lambda item: paper_rank_key(item[1]),
            reverse=True,
        )
        progress.start_stage(
            f"Reference depth {depth_index + 1}/{args.citation_depth}",
            total=len(frontier_items),
            unit="papers",
        )
        next_candidate_ids: set[str] = set()

        for frontier_index, (target_id, citing_paper) in enumerate(frontier_items, start=1):
            expanded_node_ids.add(target_id)
            target_depth = node_depth_by_id.get(target_id, depth_index)
            graph_stats["expanded_paper_count"] += 1
            depth_counts = graph_stats["expanded_depth_counts"]
            depth_key = str(target_depth)
            depth_counts[depth_key] = depth_counts.get(depth_key, 0) + 1

            references = citing_paper.get("references") or []
            if not isinstance(references, list):
                progress.update(
                    frontier_index,
                    total=len(frontier_items),
                    detail=(
                        f"raw_refs={graph_stats['raw_references_seen']:,}, "
                        f"inspected_refs={graph_stats['references_inspected']:,}, "
                        f"edges={len(edges):,}, nodes={len(nodes):,}, "
                        f"skipped={sum(skipped.values()):,}"
                    ),
                )
                continue

            graph_stats["raw_references_seen"] += len(references)
            for cited_paper in limited_reference_papers(
                references,
                args.max_references_per_paper,
                skipped,
            ):
                graph_stats["references_inspected"] += 1
                source_id = node_id_for_paper(cited_paper)
                if not source_id:
                    skipped["references_without_id"] += 1
                    continue

                if source_id == target_id:
                    skipped["self_edges"] += 1
                    continue

                if args.internal_only and source_id not in seed_node_ids:
                    skipped["external_references"] += 1
                    continue

                if (
                    args.arxiv_references_only
                    and source_id not in seed_node_ids
                    and not external_id(cited_paper, "arxiv")
                ):
                    skipped["non_arxiv_references"] += 1
                    continue

                if not temporal_edge_allowed(
                    cited_paper,
                    citing_paper,
                    args.allow_nontemporal_edges,
                ):
                    skipped["nontemporal_edges"] += 1
                    continue

                if (
                    source_id not in nodes
                    and args.target_node_count > 0
                    and len(nodes) >= args.target_node_count
                ):
                    skipped["target_node_budget"] += 1
                    graph_stats["target_node_budget_reached"] = True
                    continue

                if source_id not in nodes:
                    upsert_node(nodes, cited_paper, "reference")
                else:
                    add_node_role(nodes, source_id, "reference")

                source_depth = target_depth + 1
                previous_depth = node_depth_by_id.get(source_id)
                if previous_depth is None or source_depth < previous_depth:
                    node_depth_by_id[source_id] = source_depth
                    target_roots = nodes.get(target_id, {}).get("seed_component_roots") or []
                    if not target_roots and root_seed_by_node_id.get(target_id):
                        target_roots = [root_seed_by_node_id[target_id]]
                    if target_roots:
                        root_seed_by_node_id[source_id] = target_roots[0]
                actual_source_depth = node_depth_by_id.get(source_id, source_depth)
                set_node_depth(nodes, source_id, actual_source_depth)

                for root_seed_id in nodes.get(target_id, {}).get("seed_component_roots") or []:
                    add_seed_component_root(nodes, source_id, root_seed_id)

                papers_by_node_id.setdefault(source_id, cited_paper)
                lookup_id = s2_lookup_id_for_paper(cited_paper)
                if lookup_id:
                    lookup_id_by_node_id.setdefault(source_id, lookup_id)

                if depth_index + 1 < args.citation_depth and source_id not in expanded_node_ids:
                    next_candidate_ids.add(source_id)

                edge_key = (source_id, target_id, "cited_by")
                if edge_key in edges:
                    skipped["duplicate_edges"] += 1
                    continue

                edges[edge_key] = {
                    "source": source_id,
                    "target": target_id,
                    "relation": "cited_by",
                    "orientation": "cited_reference_to_citing_paper",
                    "source_year": publication_year(cited_paper),
                    "target_year": publication_year(citing_paper),
                    "source_arxiv_id": external_id(cited_paper, "arxiv"),
                    "target_arxiv_id": (
                        seed_entry_by_node_id.get(target_id, {}).get("arxiv_id")
                        or external_id(citing_paper, "arxiv")
                    ),
                    "source_seed_reference_depth": actual_source_depth,
                    "target_seed_reference_depth": target_depth,
                }

            progress.update(
                frontier_index,
                total=len(frontier_items),
                detail=(
                    f"raw_refs={graph_stats['raw_references_seen']:,}, "
                    f"inspected_refs={graph_stats['references_inspected']:,}, "
                    f"edges={len(edges):,}, nodes={len(nodes):,}, "
                    f"next_candidates={len(next_candidate_ids):,}, "
                    f"skipped={sum(skipped.values()):,}"
                ),
            )

        progress.finish_stage(
            completed=len(frontier_items),
            total=len(frontier_items),
            detail=(
                f"raw_refs={graph_stats['raw_references_seen']:,}, "
                f"inspected_refs={graph_stats['references_inspected']:,}, "
                f"edges={len(edges):,}, nodes={len(nodes):,}, "
                f"next_candidates={len(next_candidate_ids):,}, "
                f"skipped={sum(skipped.values()):,}"
            ),
        )

        if depth_index + 1 >= args.citation_depth:
            break

        next_candidate_ids -= expanded_node_ids
        frontier_cap = effective_frontier_cap(args, len(nodes))
        graph_stats.setdefault("effective_frontier_caps", {})[
            str(depth_index + 2)
        ] = frontier_cap
        selected_next_ids = select_frontier_node_ids(
            next_candidate_ids,
            papers_by_node_id,
            frontier_cap,
            root_seed_by_node_id,
            skipped,
        )
        lookup_to_node_ids: dict[str, list[str]] = {}
        next_frontier: dict[str, dict[str, Any]] = {}
        for node_id in selected_next_ids:
            paper = papers_by_node_id.get(node_id)
            if paper and reference_list_available(paper):
                next_frontier[node_id] = paper
                continue

            lookup_id = lookup_id_by_node_id.get(node_id)
            if not lookup_id and paper:
                lookup_id = s2_lookup_id_for_paper(paper)
            if not lookup_id:
                skipped["expansion_lookup_missing"] += 1
                continue
            lookup_to_node_ids.setdefault(lookup_id, []).append(node_id)

        if lookup_to_node_ids:
            if client is None:
                raise RuntimeError("A SemanticScholarClient is required for citation depth > 1")
            fetched_papers = fetch_s2_papers_by_lookup_ids(
                client,
                list(lookup_to_node_ids),
                args.s2_batch_size,
                args.s2_sleep,
                progress,
                stage_name=f"Semantic Scholar expansion depth {depth_index + 2}",
                s2_concurrency=args.s2_concurrency,
            )
            graph_stats["expansion_fetch_count"] += len(fetched_papers)
            for lookup_id, node_ids in lookup_to_node_ids.items():
                fetched_paper = fetched_papers.get(lookup_id)
                if not fetched_paper:
                    skipped["expansion_fetch_missing"] += len(node_ids)
                    continue
                for node_id in node_ids:
                    merge_paper_metadata_into_node(nodes, node_id, fetched_paper, "reference")
                    set_node_depth(nodes, node_id, node_depth_by_id.get(node_id, depth_index + 1))
                    papers_by_node_id[node_id] = fetched_paper
                    fetched_lookup_id = s2_lookup_id_for_paper(fetched_paper)
                    if fetched_lookup_id:
                        lookup_id_by_node_id[node_id] = fetched_lookup_id
                    if reference_list_available(fetched_paper):
                        next_frontier[node_id] = fetched_paper
                    else:
                        skipped["expansion_fetch_missing"] += 1

        progress.log(
            f"Reference depth {depth_index + 2}: prepared frontier "
            f"{len(next_frontier):,}/{len(selected_next_ids):,} papers"
        )
        current_frontier = next_frontier

    node_list = sorted(
        nodes.values(),
        key=lambda node: (
            "seed_arxiv" not in node.get("roles", []),
            -(node.get("citation_count") or 0),
            node.get("title") or "",
        ),
    )
    edge_list = sorted(edges.values(), key=lambda edge: (edge["source"], edge["target"]))
    return node_list, edge_list, skipped, graph_stats


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            json.dump(record, file, ensure_ascii=False)
            file.write("\n")


def main() -> int:
    args = parse_args()
    apply_graph_size_preset(args)
    if args.max_arxiv_results <= 0:
        raise SystemExit("--max-arxiv-results must be positive")
    if args.arxiv_batch_size <= 0:
        raise SystemExit("--arxiv-batch-size must be positive")
    if (
        (args.candidate_source == "arxiv" or args.fetch_arxiv_html)
        and args.arxiv_sleep < ARXIV_MIN_SLEEP_SECONDS
    ):
        raise SystemExit(
            f"--arxiv-sleep must be >= {ARXIV_MIN_SLEEP_SECONDS:g} seconds for arXiv API limits"
        )
    if args.arxiv_initial_sleep < 0:
        raise SystemExit("--arxiv-initial-sleep must be non-negative")
    if args.arxiv_rate_limit_base_sleep <= 0:
        raise SystemExit("--arxiv-rate-limit-base-sleep must be positive")
    if args.arxiv_rate_limit_max_sleep < args.arxiv_rate_limit_base_sleep:
        raise SystemExit(
            "--arxiv-rate-limit-max-sleep must be >= --arxiv-rate-limit-base-sleep"
        )
    if args.arxiv_retry_jitter < 0:
        raise SystemExit("--arxiv-retry-jitter must be non-negative")
    if args.s2_batch_size <= 0:
        raise SystemExit("--s2-batch-size must be positive")
    if args.s2_batch_size > 500:
        raise SystemExit("--s2-batch-size must be <= 500 for the S2 batch endpoint")
    if args.s2_search_batch_size <= 0:
        raise SystemExit("--s2-search-batch-size must be positive")
    if args.s2_search_batch_size > 1000:
        raise SystemExit("--s2-search-batch-size must be <= 1000 for the S2 bulk search endpoint")
    if args.s2_search_max_pages <= 0:
        raise SystemExit("--s2-search-max-pages must be positive")
    if args.s2_concurrency <= 0:
        raise SystemExit("--s2-concurrency must be positive")
    if args.citation_depth <= 0:
        raise SystemExit("--citation-depth must be positive")
    if args.target_node_count < 0:
        raise SystemExit("--target-node-count must be non-negative")
    if args.max_references_per_paper < 0:
        raise SystemExit("--max-references-per-paper must be non-negative")
    if args.max_expansion_papers_per_depth < 0:
        raise SystemExit("--max-expansion-papers-per-depth must be non-negative")
    if args.arxiv_html_concurrency <= 0:
        raise SystemExit("--arxiv-html-concurrency must be positive")
    if args.progress_interval < 0:
        raise SystemExit("--progress-interval must be non-negative")

    load_env_file(args.env_path)
    s2_api_key = os.environ.get("S2_API_KEY")
    if not s2_api_key and not args.allow_missing_s2_key:
        raise SystemExit(
            "Missing S2_API_KEY. Add it to .env or rerun with --allow-missing-s2-key."
        )

    progress = ProgressReporter(
        enabled=not args.no_progress,
        interval_seconds=args.progress_interval,
    )
    if (
        args.citation_depth > 1
        and args.max_references_per_paper == 0
        and args.max_expansion_papers_per_depth == 0
    ):
        progress.log(
            "Recursive citation expansion is uncapped. Consider setting "
            "--max-references-per-paper or --max-expansion-papers-per-depth "
            "to avoid very large graphs."
        )

    client = SemanticScholarClient(s2_api_key, args.timeout, args.max_retries, progress)
    if args.candidate_source == "arxiv":
        candidates = fetch_arxiv_candidates(args, progress)
    elif args.candidate_source == "s2-search":
        candidates = fetch_s2_search_arxiv_candidates(args, client, progress)
    else:
        raise RuntimeError(f"Unsupported candidate source: {args.candidate_source}")
    if not candidates:
        raise SystemExit(f"No candidates found from {args.candidate_source}.")

    s2_papers = fetch_s2_papers(
        client,
        [candidate["arxiv_id"] for candidate in candidates],
        args.s2_batch_size,
        args.s2_sleep,
        progress,
        args.s2_concurrency,
    )

    seed_papers = select_seed_papers(candidates, s2_papers, args, progress)
    if not seed_papers:
        raise SystemExit(
            "No seed papers passed the filters. Lower --min-citations or increase --max-arxiv-results."
        )

    nodes, edges, skipped, graph_stats = build_graph(seed_papers, args, progress, client)

    progress.log(f"Writing graph outputs to {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_node_ids = {
        node_id_for_paper(paper, arxiv_entry)
        for arxiv_entry, paper in seed_papers
        if node_id_for_paper(paper, arxiv_entry)
    }
    seed_nodes = [node for node in nodes if node["node_id"] in seed_node_ids]

    arxiv_html_records: list[dict[str, Any]] = []
    if args.fetch_arxiv_html:
        arxiv_html_records = fetch_arxiv_html_pages(nodes, seed_node_ids, args, progress)

    arxiv_html_status_counts: dict[str, int] = {}
    for record in arxiv_html_records:
        status = str(record.get("status"))
        arxiv_html_status_counts[status] = arxiv_html_status_counts.get(status, 0) + 1

    manifest = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "candidate_source": args.candidate_source,
        "query": args.query,
        "sort_by": args.sort_by,
        "sort_order": args.sort_order,
        "max_arxiv_results": args.max_arxiv_results,
        "arxiv_batch_size": args.arxiv_batch_size,
        "arxiv_sleep": args.arxiv_sleep,
        "arxiv_initial_sleep": args.arxiv_initial_sleep,
        "arxiv_rate_limit_base_sleep": args.arxiv_rate_limit_base_sleep,
        "arxiv_rate_limit_max_sleep": args.arxiv_rate_limit_max_sleep,
        "arxiv_retry_jitter": args.arxiv_retry_jitter,
        "arxiv_rate_limit_lock_path": (
            None
            if args.no_arxiv_rate_limit_lock
            else str(args.arxiv_rate_limit_lock_path)
        ),
        "s2_search_query": args.s2_search_query,
        "s2_search_effective_query": (
            args.s2_search_query or derive_s2_search_query(args.query)
            if args.candidate_source == "s2-search"
            else None
        ),
        "s2_fields_of_study": args.s2_fields_of_study,
        "s2_fields_of_study_effective": (
            split_csv_option(args.s2_fields_of_study)
            if args.s2_fields_of_study is not None
            else derive_s2_fields_of_study(args.query)
        )
        if args.candidate_source == "s2-search"
        else None,
        "s2_publication_types": args.s2_publication_types,
        "s2_search_sort": args.s2_search_sort,
        "s2_search_batch_size": args.s2_search_batch_size,
        "s2_search_max_pages": args.s2_search_max_pages,
        "min_citations": args.min_citations,
        "max_seed_papers": args.max_seed_papers,
        "citation_depth": args.citation_depth,
        "max_references_per_paper": args.max_references_per_paper,
        "max_expansion_papers_per_depth": args.max_expansion_papers_per_depth,
        "min_year": args.min_year,
        "max_year": args.max_year,
        "internal_only": args.internal_only,
        "arxiv_references_only": args.arxiv_references_only,
        "allow_nontemporal_edges": args.allow_nontemporal_edges,
        "fetch_arxiv_html": args.fetch_arxiv_html,
        "arxiv_html_scope": args.arxiv_html_scope,
        "arxiv_html_dir": str(args.arxiv_html_dir),
        "overwrite_arxiv_html": args.overwrite_arxiv_html,
        "arxiv_candidate_count": len(candidates),
        "s2_match_count": len(s2_papers),
        "seed_paper_count": len(seed_papers),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "graph_stats": graph_stats,
        "arxiv_html": {
            "enabled": args.fetch_arxiv_html,
            "record_count": len(arxiv_html_records),
            "status_counts": arxiv_html_status_counts,
        },
        "skipped": skipped,
    }

    write_jsonl(args.output_dir / "seed_papers.jsonl", seed_nodes)
    write_jsonl(args.output_dir / "nodes.jsonl", nodes)
    write_jsonl(args.output_dir / "edges.jsonl", edges)
    if args.fetch_arxiv_html:
        write_jsonl(args.output_dir / "arxiv_html.jsonl", arxiv_html_records)
    write_json(args.output_dir / "manifest.json", manifest)
    write_json(args.output_dir / "graph.json", {"metadata": manifest, "nodes": nodes, "edges": edges})

    print(f"Wrote {len(nodes)} nodes and {len(edges)} edges to {args.output_dir}")
    print(f"Skipped references: {json.dumps(skipped, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
