#!/usr/bin/env python3
"""Build a citation graph from Hugging Face Paper Pages markdown.

This is a Hugging Face Paper Pages-oriented companion to
``build_arxiv_citation_graph.py``. It uses the public HF paper endpoints from
the Hugging Face papers skill:

    https://huggingface.co/api/papers/search?q=...
    https://huggingface.co/api/papers/{ARXIV_ID}
    https://huggingface.co/papers/{ARXIV_ID}.md

The graph is intentionally arXiv-ID centric. Seed papers come from HF paper
search, explicit IDs, an ID file, or the arXiv Atom API. Edges are extracted
from arXiv/HF-paper links found in the seed paper markdown and point from the
referenced paper to the later paper that cites it.

Example:
    python arxiv_paper/build_hf_paper_pages_citation_graph.py \
        --query "retrieval augmented generation" \
        --max-papers 50 \
        --output-dir data/hf_paper_page_graph/rag

The script writes graph JSON/JSONL outputs and raw markdown files under
``paper_markdown/`` so the paper pages are easy to inspect or reuse.
"""

from __future__ import annotations

import argparse
import datetime as dt
import email.utils
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable, TextIO
from urllib.parse import quote_plus
from xml.etree import ElementTree as ET

try:
    import requests
except ImportError as exc:  # pragma: no cover - import-time guard
    raise SystemExit("Missing dependency: pip install requests") from exc


HF_BASE_URL = "https://huggingface.co"
ARXIV_API_URL = "https://export.arxiv.org/api/query"
USER_AGENT = "asmr-private-hf-paper-pages-citation-graph/1.0"
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
RATE_LIMIT_BASE_SLEEP_SECONDS = 60.0
RATE_LIMIT_MAX_SLEEP_SECONDS = 600.0
TRANSIENT_MAX_SLEEP_SECONDS = 60.0

ATOM_NS = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"

MODERN_ARXIV_ID_RE = re.compile(r"^\d{4}\.\d{4,5}(?:v\d+)?$")
LEGACY_ARXIV_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*/\d{7}(?:v\d+)?$")
ARXIV_VERSION_RE = re.compile(r"v\d+$")

ARXIV_URL_ID_RE = re.compile(
    r"(?:arxiv\.org/(?:abs|pdf|html)/|huggingface\.co/papers/)"
    r"([A-Za-z][A-Za-z0-9_.-]*/\d{7}|\d{4}\.\d{4,5})(?:v\d+)?"
    r"(?:\.pdf|\.md)?",
    flags=re.IGNORECASE,
)
ARXIV_LABEL_ID_RE = re.compile(
    r"\barxiv\s*:\s*"
    r"([A-Za-z][A-Za-z0-9_.-]*/\d{7}|\d{4}\.\d{4,5})(?:v\d+)?",
    flags=re.IGNORECASE,
)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an arXiv-ID citation graph from Hugging Face Paper Pages "
            "markdown, while saving the markdown files locally."
        )
    )
    parser.add_argument(
        "--source",
        choices=["hf-search", "arxiv"],
        default="hf-search",
        help=(
            "Candidate source. 'hf-search' treats --query as a Hugging Face "
            "paper search query. 'arxiv' treats --query as an arXiv Atom "
            "search_query value such as 'cat:cs.LG'. Explicit --paper-id "
            "values are always included."
        ),
    )
    parser.add_argument(
        "--query",
        default=None,
        help=(
            "Candidate query. For --source hf-search, this is passed to "
            "/api/papers/search. For --source arxiv, this is passed to the "
            "arXiv Atom API."
        ),
    )
    parser.add_argument(
        "--paper-id",
        action="append",
        default=[],
        help=(
            "Explicit arXiv ID or HF/arXiv paper URL to include. May be "
            "provided multiple times."
        ),
    )
    parser.add_argument(
        "--paper-ids-file",
        type=Path,
        default=None,
        help=(
            "Optional text file containing arXiv IDs or paper URLs. Commas, "
            "whitespace, and comments starting with # are supported."
        ),
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=50,
        help="Maximum seed papers to process after de-duplication.",
    )
    parser.add_argument(
        "--hf-search-limit",
        type=int,
        default=120,
        help="Maximum HF search results to request in one API call.",
    )
    parser.add_argument(
        "--min-year",
        type=int,
        default=None,
        help="Optional minimum seed publication year.",
    )
    parser.add_argument(
        "--max-year",
        type=int,
        default=None,
        help="Optional maximum seed publication year.",
    )
    parser.add_argument(
        "--max-references-per-paper",
        type=int,
        default=250,
        help=(
            "Maximum unique arXiv/HF paper links to keep from each seed "
            "markdown file. Use 0 for no limit."
        ),
    )
    parser.add_argument(
        "--fetch-reference-metadata",
        action="store_true",
        help=(
            "Also call /api/papers/{id} for referenced papers. By default, "
            "reference nodes only include IDs and URLs."
        ),
    )
    parser.add_argument(
        "--fetch-reference-markdown",
        action="store_true",
        help=(
            "Also save markdown for referenced papers. This can be much "
            "slower and is not needed for seed markdown access."
        ),
    )
    parser.add_argument(
        "--allow-nontemporal-edges",
        action="store_true",
        help=(
            "Keep edges even when inferred years suggest the referenced paper "
            "is newer than the citing seed paper."
        ),
    )
    parser.add_argument(
        "--sort-by",
        choices=["relevance", "lastUpdatedDate", "submittedDate"],
        default="relevance",
        help="arXiv API sortBy value when --source arxiv is used.",
    )
    parser.add_argument(
        "--sort-order",
        choices=["ascending", "descending"],
        default="descending",
        help="arXiv API sortOrder value when --source arxiv is used.",
    )
    parser.add_argument(
        "--arxiv-batch-size",
        type=int,
        default=100,
        help="arXiv records per Atom API request when --source arxiv is used.",
    )
    parser.add_argument(
        "--arxiv-sleep",
        type=float,
        default=3.0,
        help="Seconds to sleep between arXiv API calls.",
    )
    parser.add_argument(
        "--request-sleep",
        type=float,
        default=0.2,
        help="Seconds to sleep between HF paper metadata/markdown requests.",
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
        help="Path to a .env file containing optional HF_TOKEN.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/hf_paper_page_citation_graph"),
        help="Directory for graph outputs and paper_markdown/.",
    )
    parser.add_argument(
        "--markdown-dir",
        type=Path,
        default=None,
        help=(
            "Directory for raw HF paper markdown. Defaults to "
            "<output-dir>/paper_markdown."
        ),
    )
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
    return parser.parse_args()


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


def normalize_arxiv_id(value: str | None) -> str | None:
    if not value:
        return None
    text = value.strip()
    text = re.sub(r"^arxiv\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^https?://arxiv\.org/(abs|pdf|html)/", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^https?://huggingface\.co/papers/", "", text, flags=re.IGNORECASE)
    text = text.split("#", 1)[0].split("?", 1)[0].strip()
    for suffix in [".pdf", ".md"]:
        if text.lower().endswith(suffix):
            text = text[: -len(suffix)]
    text = ARXIV_VERSION_RE.sub("", text)
    if MODERN_ARXIV_ID_RE.match(text) or LEGACY_ARXIV_ID_RE.match(text):
        return text
    return None


def arxiv_id_sort_key(arxiv_id: str) -> tuple[int, str]:
    if MODERN_ARXIV_ID_RE.match(arxiv_id):
        return (0, arxiv_id)
    return (1, arxiv_id)


def safe_filename(arxiv_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", arxiv_id)


def markdown_path_for(markdown_dir: Path, arxiv_id: str) -> Path:
    return markdown_dir / f"{safe_filename(arxiv_id)}.md"


def unique_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        normalized = normalize_arxiv_id(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(normalized)
    return output


def read_paper_ids_file(path: Path) -> list[str]:
    values: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        values.extend(part.strip() for part in re.split(r"[\s,]+", line) if part.strip())
    return values


def retry_after_seconds(response: requests.Response, attempt: int) -> float:
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
        return min(RATE_LIMIT_MAX_SLEEP_SECONDS, RATE_LIMIT_BASE_SLEEP_SECONDS * (2.0**attempt))
    return min(TRANSIENT_MAX_SLEEP_SECONDS, 2.0**attempt)


def request_with_retries(
    session: requests.Session,
    method: str,
    url: str,
    *,
    max_retries: int,
    timeout: float,
    progress: ProgressReporter | None = None,
    request_label: str | None = None,
    **kwargs: Any,
) -> requests.Response:
    label = request_label or f"{method} {url}"
    for attempt in range(max_retries + 1):
        try:
            response = session.request(method, url, timeout=timeout, **kwargs)
        except requests.RequestException as exc:
            if attempt >= max_retries:
                raise
            sleep_for = min(60.0, 2.0**attempt)
            if progress:
                progress.log(
                    f"{label}: attempt {attempt + 1}/{max_retries + 1} failed "
                    f"with {exc.__class__.__name__}; retrying in {format_duration(sleep_for)}"
                )
            time.sleep(sleep_for)
            continue

        if response.status_code not in RETRYABLE_STATUS_CODES:
            return response

        if attempt >= max_retries:
            return response

        sleep_for = retry_after_seconds(response, attempt)
        if progress:
            progress.log(
                f"{label}: HTTP {response.status_code} on attempt "
                f"{attempt + 1}/{max_retries + 1}; retrying in {format_duration(sleep_for)}"
            )
        time.sleep(sleep_for)

    raise RuntimeError("unreachable retry loop")


class HuggingFacePaperPagesClient:
    def __init__(
        self,
        *,
        token: str | None,
        timeout: float,
        max_retries: int,
        progress: ProgressReporter,
    ) -> None:
        self.timeout = timeout
        self.max_retries = max_retries
        self.progress = progress
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        if token:
            self.session.headers.update({"Authorization": f"Bearer {token}"})

    def search(self, query: str, limit: int) -> list[dict[str, Any]]:
        url = f"{HF_BASE_URL}/api/papers/search"
        response = request_with_retries(
            self.session,
            "GET",
            url,
            params={"q": query, "limit": limit},
            max_retries=self.max_retries,
            timeout=self.timeout,
            progress=self.progress,
            request_label=f"HF paper search query={query!r}",
        )
        response.raise_for_status()
        return extract_records_from_response(response.json())

    def paper_metadata(self, arxiv_id: str) -> dict[str, Any]:
        url = f"{HF_BASE_URL}/api/papers/{quote_plus(arxiv_id)}"
        response = request_with_retries(
            self.session,
            "GET",
            url,
            max_retries=self.max_retries,
            timeout=self.timeout,
            progress=self.progress,
            request_label=f"HF paper metadata {arxiv_id}",
        )
        if response.status_code == 404:
            return {
                "arxiv_id": arxiv_id,
                "hf_indexed": False,
                "metadata_error": "not_indexed_or_not_found",
            }
        response.raise_for_status()
        metadata = unwrap_paper_record(response.json())
        metadata["hf_indexed"] = True
        return metadata

    def paper_markdown(self, arxiv_id: str) -> tuple[str | None, str | None]:
        md_url = f"{HF_BASE_URL}/papers/{quote_plus(arxiv_id)}.md"
        response = request_with_retries(
            self.session,
            "GET",
            md_url,
            max_retries=self.max_retries,
            timeout=self.timeout,
            progress=self.progress,
            request_label=f"HF paper markdown {arxiv_id}",
        )
        if response.status_code == 200 and response.text.strip():
            return response.text, md_url

        # The papers skill documents this as an equivalent markdown path.
        page_url = f"{HF_BASE_URL}/papers/{quote_plus(arxiv_id)}"
        response = request_with_retries(
            self.session,
            "GET",
            page_url,
            headers={"Accept": "text/markdown"},
            max_retries=self.max_retries,
            timeout=self.timeout,
            progress=self.progress,
            request_label=f"HF paper markdown fallback {arxiv_id}",
        )
        if response.status_code == 200 and response.text.strip():
            return response.text, page_url
        return None, f"HTTP {response.status_code}"


def extract_records_from_response(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [record for record in data if isinstance(record, dict)]
    if not isinstance(data, dict):
        return []
    for key in ["papers", "results", "items", "data"]:
        value = data.get(key)
        if isinstance(value, list):
            return [record for record in value if isinstance(record, dict)]
    nested = data.get("paper")
    if isinstance(nested, dict):
        return [nested]
    return [data]


def unwrap_paper_record(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    for key in ["paper", "paperData", "data"]:
        nested = data.get(key)
        if isinstance(nested, dict):
            return dict(nested)
    return dict(data)


def arxiv_id_from_record(record: dict[str, Any]) -> str | None:
    for key in [
        "arxiv_id",
        "arxivId",
        "arxiv",
        "paperId",
        "paper_id",
        "id",
        "url",
        "paperUrl",
        "htmlUrl",
        "pdfUrl",
    ]:
        value = record.get(key)
        if isinstance(value, str):
            arxiv_id = normalize_arxiv_id(value)
            if arxiv_id:
                return arxiv_id

    external_ids = record.get("externalIds") or record.get("external_ids")
    if isinstance(external_ids, dict):
        for key, value in external_ids.items():
            if key.lower() == "arxiv" and isinstance(value, str):
                arxiv_id = normalize_arxiv_id(value)
                if arxiv_id:
                    return arxiv_id

    tags = record.get("tags")
    if isinstance(tags, list):
        for tag in tags:
            if isinstance(tag, str):
                arxiv_id = normalize_arxiv_id(tag)
                if arxiv_id:
                    return arxiv_id
    return None


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
        arxiv_id = normalize_arxiv_id(entry_id)
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


def fetch_arxiv_candidates(
    args: argparse.Namespace,
    progress: ProgressReporter,
) -> list[dict[str, Any]]:
    if not args.query:
        return []
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    start = 0
    batch_number = 0
    progress.start_stage(
        "arXiv candidate fetch",
        total=args.max_papers,
        unit="candidates",
        detail=f"query={args.query!r}",
    )
    while len(candidates) < args.max_papers:
        batch_number += 1
        max_results = min(args.arxiv_batch_size, args.max_papers - len(candidates))
        response = request_with_retries(
            session,
            "GET",
            ARXIV_API_URL,
            params={
                "search_query": args.query,
                "start": start,
                "max_results": max_results,
                "sortBy": args.sort_by,
                "sortOrder": args.sort_order,
            },
            max_retries=args.max_retries,
            timeout=args.timeout,
            progress=progress,
            request_label=f"arXiv batch {batch_number} start={start}",
        )
        response.raise_for_status()
        batch = parse_arxiv_feed(response.content)
        if not batch:
            progress.log(f"arXiv batch {batch_number}: no records; stopping")
            break
        added = 0
        for record in batch:
            arxiv_id = record["arxiv_id"]
            if arxiv_id in seen:
                continue
            seen.add(arxiv_id)
            candidates.append(record)
            added += 1
        progress.update(
            len(candidates),
            total=args.max_papers,
            detail=f"batch={batch_number}, returned={len(batch)}, added={added}",
            force=True,
        )
        if added == 0 or len(batch) < max_results:
            break
        start += len(batch)
        if len(candidates) < args.max_papers:
            time.sleep(args.arxiv_sleep)
    progress.finish_stage(completed=len(candidates), total=args.max_papers)
    return candidates


def collect_seed_ids(
    args: argparse.Namespace,
    hf_client: HuggingFacePaperPagesClient,
    progress: ProgressReporter,
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    records_by_id: dict[str, dict[str, Any]] = {}
    candidate_ids: list[str] = []

    explicit_values = list(args.paper_id)
    if args.paper_ids_file:
        explicit_values.extend(read_paper_ids_file(args.paper_ids_file))
    explicit_ids = unique_preserve_order(explicit_values)
    for arxiv_id in explicit_ids:
        candidate_ids.append(arxiv_id)
        records_by_id.setdefault(arxiv_id, {"arxiv_id": arxiv_id, "source": "explicit"})

    if args.query and args.source == "hf-search":
        progress.start_stage(
            "HF paper search",
            total=min(args.hf_search_limit, args.max_papers),
            unit="results",
            detail=f"query={args.query!r}",
        )
        records = hf_client.search(args.query, min(args.hf_search_limit, max(args.max_papers, 1)))
        for record in records:
            arxiv_id = arxiv_id_from_record(record)
            if not arxiv_id:
                continue
            candidate_ids.append(arxiv_id)
            existing = records_by_id.setdefault(arxiv_id, {})
            existing.update(record)
            existing.setdefault("source", "hf-search")
        progress.finish_stage(completed=len(records), total=min(args.hf_search_limit, args.max_papers))

    if args.query and args.source == "arxiv":
        for record in fetch_arxiv_candidates(args, progress):
            arxiv_id = record["arxiv_id"]
            candidate_ids.append(arxiv_id)
            existing = records_by_id.setdefault(arxiv_id, {})
            existing.update(record)
            existing.setdefault("source", "arxiv")

    seed_ids = unique_preserve_order(candidate_ids)
    if args.max_papers > 0:
        seed_ids = seed_ids[: args.max_papers]
    return seed_ids, records_by_id


def extract_reference_arxiv_ids(markdown: str, self_arxiv_id: str) -> list[str]:
    matches: list[tuple[int, str]] = []
    for regex in [ARXIV_URL_ID_RE, ARXIV_LABEL_ID_RE]:
        for match in regex.finditer(markdown):
            arxiv_id = normalize_arxiv_id(match.group(1))
            if arxiv_id:
                matches.append((match.start(), arxiv_id))
    candidates = [arxiv_id for _, arxiv_id in sorted(matches)]
    return [value for value in unique_preserve_order(candidates) if value != self_arxiv_id]


def first_value(record: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        value = record.get(key)
        if value not in [None, "", []]:
            return value
    return None


def parse_year_from_date(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and len(value) >= 4:
        try:
            return int(value[:4])
        except ValueError:
            return None
    return None


def year_from_arxiv_id(arxiv_id: str) -> int | None:
    if not MODERN_ARXIV_ID_RE.match(arxiv_id):
        return None
    yy = int(arxiv_id[:2])
    return 2000 + yy if yy < 90 else 1900 + yy


def publication_year(record: dict[str, Any], arxiv_id: str) -> int | None:
    for key in ["year", "publishedAt", "published_at", "publishedDate", "published", "submittedOn"]:
        year = parse_year_from_date(record.get(key))
        if year:
            return year
    return year_from_arxiv_id(arxiv_id)


def parse_authors(record: dict[str, Any]) -> list[dict[str, Any]]:
    authors = first_value(record, ["authors", "authorNames", "author_names"])
    parsed: list[dict[str, Any]] = []
    if isinstance(authors, list):
        for author in authors:
            if isinstance(author, str):
                name = clean_text(author)
                if name:
                    parsed.append({"name": name, "hf_username": None, "author_id": None})
            elif isinstance(author, dict):
                name = clean_text(
                    first_value(author, ["name", "fullName", "displayName", "user"])
                )
                if not name:
                    user = author.get("user")
                    if isinstance(user, dict):
                        name = clean_text(first_value(user, ["fullname", "name", "username"]))
                parsed.append(
                    {
                        "name": name,
                        "hf_username": clean_text(
                            first_value(author, ["username", "userName", "hfUsername"])
                        ),
                        "author_id": clean_text(first_value(author, ["_id", "id", "authorId"])),
                    }
                )
    return [author for author in parsed if author.get("name")]


def normalize_hf_metadata(
    arxiv_id: str,
    metadata: dict[str, Any] | None,
    candidate_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    if candidate_record:
        merged.update(candidate_record)
    if metadata:
        merged.update(metadata)
    merged["arxiv_id"] = arxiv_id
    merged.setdefault("hf_indexed", bool(metadata and metadata.get("hf_indexed", True)))
    return merged


def node_from_metadata(
    arxiv_id: str,
    metadata: dict[str, Any],
    role: str,
    markdown_relpath: str | None = None,
) -> dict[str, Any]:
    title = clean_text(first_value(metadata, ["title", "name"]))
    abstract = clean_text(first_value(metadata, ["summary", "abstract", "description"]))
    project_page = clean_text(first_value(metadata, ["projectPage", "project_page"]))
    github_repo = clean_text(first_value(metadata, ["githubRepo", "github_repo"]))
    upvotes = first_value(metadata, ["upvotes", "upvoteCount", "numLikes"])
    try:
        upvotes = int(upvotes) if upvotes is not None else None
    except (TypeError, ValueError):
        upvotes = None

    return {
        "node_id": f"arxiv:{arxiv_id}",
        "roles": [role],
        "arxiv_id": arxiv_id,
        "title": title,
        "abstract": abstract,
        "year": publication_year(metadata, arxiv_id),
        "publication_date": clean_text(
            first_value(metadata, ["publishedAt", "published_at", "publishedDate", "published"])
        ),
        "authors": parse_authors(metadata),
        "hf_paper_url": f"{HF_BASE_URL}/papers/{arxiv_id}",
        "hf_markdown_url": f"{HF_BASE_URL}/papers/{arxiv_id}.md",
        "arxiv_url": f"https://arxiv.org/abs/{arxiv_id}",
        "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
        "local_markdown_path": markdown_relpath,
        "hf_indexed": metadata.get("hf_indexed"),
        "project_page": project_page,
        "github_repo": github_repo,
        "organization": first_value(metadata, ["organization", "org"]),
        "upvotes": upvotes,
        "source_metadata": metadata,
    }


def merge_node(existing: dict[str, Any], incoming: dict[str, Any]) -> None:
    for role in incoming.get("roles", []):
        if role not in existing["roles"]:
            existing["roles"].append(role)
    existing["roles"].sort()
    for key in [
        "title",
        "abstract",
        "year",
        "publication_date",
        "authors",
        "local_markdown_path",
        "hf_indexed",
        "project_page",
        "github_repo",
        "organization",
        "upvotes",
    ]:
        if not existing.get(key) and incoming.get(key):
            existing[key] = incoming[key]
    if not existing.get("source_metadata") and incoming.get("source_metadata"):
        existing["source_metadata"] = incoming["source_metadata"]


def upsert_node(nodes: dict[str, dict[str, Any]], node: dict[str, Any]) -> None:
    node_id = node["node_id"]
    existing = nodes.get(node_id)
    if existing is None:
        nodes[node_id] = node
        return
    merge_node(existing, node)


def temporal_edge_allowed(
    source_node: dict[str, Any],
    target_node: dict[str, Any],
    allow_nontemporal_edges: bool,
) -> bool:
    if allow_nontemporal_edges:
        return True
    source_year = source_node.get("year")
    target_year = target_node.get("year")
    if isinstance(source_year, int) and isinstance(target_year, int):
        return source_year <= target_year
    return True


def relative_path(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()


def maybe_sleep(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)


def build_graph(
    args: argparse.Namespace,
    hf_client: HuggingFacePaperPagesClient,
    seed_ids: list[str],
    candidate_records: dict[str, dict[str, Any]],
    markdown_dir: Path,
    progress: ProgressReporter,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    nodes: dict[str, dict[str, Any]] = {}
    edges: dict[tuple[str, str, str], dict[str, Any]] = {}
    page_records: list[dict[str, Any]] = []
    skipped = {
        "seed_year_filtered": 0,
        "seed_markdown_missing": 0,
        "duplicate_edges": 0,
        "self_edges": 0,
        "nontemporal_edges": 0,
        "references_limited": 0,
    }

    progress.start_stage("Seed paper markdown fetch", total=len(seed_ids), unit="papers")
    for index, arxiv_id in enumerate(seed_ids, start=1):
        metadata = normalize_hf_metadata(
            arxiv_id,
            hf_client.paper_metadata(arxiv_id),
            candidate_records.get(arxiv_id),
        )
        seed_year = publication_year(metadata, arxiv_id)
        if args.min_year is not None and seed_year is not None and seed_year < args.min_year:
            skipped["seed_year_filtered"] += 1
            progress.update(index, total=len(seed_ids), detail=f"skipped={sum(skipped.values())}")
            maybe_sleep(args.request_sleep)
            continue
        if args.max_year is not None and seed_year is not None and seed_year > args.max_year:
            skipped["seed_year_filtered"] += 1
            progress.update(index, total=len(seed_ids), detail=f"skipped={sum(skipped.values())}")
            maybe_sleep(args.request_sleep)
            continue

        markdown, markdown_source = hf_client.paper_markdown(arxiv_id)
        markdown_relpath = None
        reference_ids: list[str] = []
        if markdown:
            markdown_path = markdown_path_for(markdown_dir, arxiv_id)
            markdown_path.write_text(markdown, encoding="utf-8")
            markdown_relpath = relative_path(markdown_path, args.output_dir)
            reference_ids = extract_reference_arxiv_ids(markdown, arxiv_id)
            if args.max_references_per_paper > 0 and len(reference_ids) > args.max_references_per_paper:
                skipped["references_limited"] += len(reference_ids) - args.max_references_per_paper
                reference_ids = reference_ids[: args.max_references_per_paper]
        else:
            skipped["seed_markdown_missing"] += 1

        seed_node = node_from_metadata(arxiv_id, metadata, "seed_hf_paper", markdown_relpath)
        upsert_node(nodes, seed_node)
        page_records.append(
            {
                "arxiv_id": arxiv_id,
                "title": seed_node.get("title"),
                "year": seed_node.get("year"),
                "hf_paper_url": seed_node["hf_paper_url"],
                "hf_markdown_url": seed_node["hf_markdown_url"],
                "local_markdown_path": markdown_relpath,
                "markdown_source": markdown_source if markdown else None,
                "markdown_error": None if markdown else markdown_source,
                "reference_count_extracted": len(reference_ids),
                "reference_arxiv_ids": reference_ids,
                "metadata": metadata,
            }
        )

        for reference_id in reference_ids:
            if reference_id == arxiv_id:
                skipped["self_edges"] += 1
                continue

            reference_metadata: dict[str, Any] = {"arxiv_id": reference_id, "hf_indexed": None}
            reference_markdown_relpath = None
            if args.fetch_reference_metadata:
                reference_metadata = normalize_hf_metadata(
                    reference_id,
                    hf_client.paper_metadata(reference_id),
                    None,
                )
                maybe_sleep(args.request_sleep)
            if args.fetch_reference_markdown:
                reference_markdown, _ = hf_client.paper_markdown(reference_id)
                if reference_markdown:
                    path = markdown_path_for(markdown_dir, reference_id)
                    path.write_text(reference_markdown, encoding="utf-8")
                    reference_markdown_relpath = relative_path(path, args.output_dir)
                maybe_sleep(args.request_sleep)

            reference_node = node_from_metadata(
                reference_id,
                reference_metadata,
                "reference",
                reference_markdown_relpath,
            )
            upsert_node(nodes, reference_node)
            target_node = nodes[f"arxiv:{arxiv_id}"]
            source_node = nodes[f"arxiv:{reference_id}"]
            if not temporal_edge_allowed(source_node, target_node, args.allow_nontemporal_edges):
                skipped["nontemporal_edges"] += 1
                continue

            edge_key = (source_node["node_id"], target_node["node_id"], "cited_by")
            if edge_key in edges:
                skipped["duplicate_edges"] += 1
                continue
            edges[edge_key] = {
                "source": source_node["node_id"],
                "target": target_node["node_id"],
                "relation": "cited_by",
                "orientation": "referenced_paper_to_citing_hf_paper",
                "source_arxiv_id": reference_id,
                "target_arxiv_id": arxiv_id,
                "source_year": source_node.get("year"),
                "target_year": target_node.get("year"),
                "evidence": {
                    "kind": "hf_paper_markdown_arxiv_link",
                    "seed_markdown_path": markdown_relpath,
                    "seed_hf_markdown_url": seed_node["hf_markdown_url"],
                },
            }

        progress.update(
            index,
            total=len(seed_ids),
            detail=f"nodes={len(nodes):,}, edges={len(edges):,}, skipped={sum(skipped.values()):,}",
            force=True,
        )
        maybe_sleep(args.request_sleep)

    progress.finish_stage(
        completed=len(seed_ids),
        total=len(seed_ids),
        detail=f"nodes={len(nodes):,}, edges={len(edges):,}, skipped={sum(skipped.values()):,}",
    )

    node_list = sorted(
        nodes.values(),
        key=lambda node: (
            "seed_hf_paper" not in node.get("roles", []),
            -(node.get("year") or 0),
            node.get("title") or node.get("arxiv_id") or "",
        ),
    )
    edge_list = sorted(edges.values(), key=lambda edge: (edge["source"], edge["target"]))
    return node_list, edge_list, page_records, skipped


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            json.dump(record, file, ensure_ascii=False)
            file.write("\n")


def markdown_escape_cell(value: Any) -> str:
    text = clean_text(value) or ""
    return text.replace("|", "\\|")


def write_markdown_index(
    path: Path,
    *,
    args: argparse.Namespace,
    manifest: dict[str, Any],
    page_records: list[dict[str, Any]],
) -> None:
    lines = [
        "# Hugging Face Paper Page Markdown Index",
        "",
        f"Generated: {manifest['created_at']}",
        "",
        f"- Source: `{args.source}`",
        f"- Query: `{args.query or ''}`",
        f"- Seed papers: {manifest['seed_paper_count']}",
        f"- Nodes: {manifest['node_count']}",
        f"- Edges: {manifest['edge_count']}",
        "",
        "| arXiv ID | Title | Year | References | Local Markdown | Hugging Face |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    for record in page_records:
        arxiv_id = record["arxiv_id"]
        title = markdown_escape_cell(record.get("title") or arxiv_id)
        year = record.get("year") or ""
        ref_count = record.get("reference_count_extracted") or 0
        local = record.get("local_markdown_path")
        if local:
            local_link = f"[md]({local})"
        else:
            local_link = "missing"
        hf_link = f"[HF]({HF_BASE_URL}/papers/{arxiv_id})"
        lines.append(
            f"| `{arxiv_id}` | {title} | {year} | {ref_count} | {local_link} | {hf_link} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_args(args: argparse.Namespace) -> None:
    if args.max_papers <= 0:
        raise SystemExit("--max-papers must be positive")
    if args.hf_search_limit <= 0:
        raise SystemExit("--hf-search-limit must be positive")
    if args.arxiv_batch_size <= 0:
        raise SystemExit("--arxiv-batch-size must be positive")
    if args.max_references_per_paper < 0:
        raise SystemExit("--max-references-per-paper must be non-negative")
    if args.request_sleep < 0:
        raise SystemExit("--request-sleep must be non-negative")
    if args.progress_interval < 0:
        raise SystemExit("--progress-interval must be non-negative")
    if not args.query and not args.paper_id and not args.paper_ids_file:
        raise SystemExit("Provide --query, --paper-id, or --paper-ids-file.")


def main() -> int:
    args = parse_args()
    validate_args(args)
    load_env_file(args.env_path)

    progress = ProgressReporter(
        enabled=not args.no_progress,
        interval_seconds=args.progress_interval,
    )
    hf_client = HuggingFacePaperPagesClient(
        token=os.environ.get("HF_TOKEN"),
        timeout=args.timeout,
        max_retries=args.max_retries,
        progress=progress,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    markdown_dir = args.markdown_dir or (args.output_dir / "paper_markdown")
    markdown_dir.mkdir(parents=True, exist_ok=True)

    seed_ids, candidate_records = collect_seed_ids(args, hf_client, progress)
    if not seed_ids:
        raise SystemExit("No seed paper IDs found.")

    progress.log(f"Collected {len(seed_ids):,} seed paper IDs")
    nodes, edges, page_records, skipped = build_graph(
        args,
        hf_client,
        seed_ids,
        candidate_records,
        markdown_dir,
        progress,
    )

    seed_node_ids = {f"arxiv:{record['arxiv_id']}" for record in page_records}
    seed_nodes = [node for node in nodes if node["node_id"] in seed_node_ids]
    manifest = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "builder": "build_hf_paper_pages_citation_graph",
        "source": args.source,
        "query": args.query,
        "max_papers": args.max_papers,
        "hf_search_limit": args.hf_search_limit,
        "min_year": args.min_year,
        "max_year": args.max_year,
        "max_references_per_paper": args.max_references_per_paper,
        "fetch_reference_metadata": args.fetch_reference_metadata,
        "fetch_reference_markdown": args.fetch_reference_markdown,
        "allow_nontemporal_edges": args.allow_nontemporal_edges,
        "seed_id_count": len(seed_ids),
        "seed_paper_count": len(seed_nodes),
        "markdown_file_count": sum(1 for record in page_records if record.get("local_markdown_path")),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "skipped": skipped,
        "outputs": {
            "seed_papers": "seed_papers.jsonl",
            "nodes": "nodes.jsonl",
            "edges": "edges.jsonl",
            "paper_pages": "paper_pages.jsonl",
            "graph": "graph.json",
            "markdown_index": "markdown_index.md",
            "markdown_dir": relative_path(markdown_dir, args.output_dir),
        },
    }

    progress.log(f"Writing graph outputs to {args.output_dir}")
    write_jsonl(args.output_dir / "seed_papers.jsonl", seed_nodes)
    write_jsonl(args.output_dir / "nodes.jsonl", nodes)
    write_jsonl(args.output_dir / "edges.jsonl", edges)
    write_jsonl(args.output_dir / "paper_pages.jsonl", page_records)
    write_json(args.output_dir / "manifest.json", manifest)
    write_json(args.output_dir / "graph.json", {"metadata": manifest, "nodes": nodes, "edges": edges})
    write_markdown_index(
        args.output_dir / "markdown_index.md",
        args=args,
        manifest=manifest,
        page_records=page_records,
    )

    print(f"Wrote {len(nodes)} nodes and {len(edges)} edges to {args.output_dir}")
    print(f"Wrote {manifest['markdown_file_count']} markdown files to {markdown_dir}")
    print(f"Markdown index: {args.output_dir / 'markdown_index.md'}")
    print(f"Skipped: {json.dumps(skipped, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
