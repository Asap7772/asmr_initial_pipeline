from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
GRAPH_NODES_FILENAME = "nodes.jsonl"
GRAPH_EDGES_FILENAME = "edges.jsonl"
DEFAULT_DATA_DIRNAMES = (
    "arxiv_citation_graph_premier_tcs_depth5_s2",
    # "arxiv_citation_graph_deep_dense_tcs_theory_s2",
    # "arxiv_citation_graph_deep5_dense_tcs_s2",
    # "arxiv_citation_graph_dense_tcs_s2",
    # "arxiv_citation_graph_theory_s2",
)


def is_graph_dir(path: Path) -> bool:
    return (path / GRAPH_NODES_FILENAME).is_file() and (path / GRAPH_EDGES_FILENAME).is_file()


def iter_graph_dirs(data_dir: Path) -> list[Path]:
    if not data_dir.exists():
        return []

    graph_dirs: set[Path] = set()
    if is_graph_dir(data_dir):
        graph_dirs.add(data_dir)

    for nodes_path in data_dir.rglob(GRAPH_NODES_FILENAME):
        graph_dir = nodes_path.parent
        if graph_dir != data_dir and is_graph_dir(graph_dir):
            graph_dirs.add(graph_dir)

    return sorted(graph_dirs)


def resolve_default_data_dir() -> Path | None:
    if os.environ.get("ARXIV_GRAPH_DATA_DIR"):
        return Path(os.environ["ARXIV_GRAPH_DATA_DIR"])

    for dirname in DEFAULT_DATA_DIRNAMES:
        candidate = REPO_ROOT / "data" / dirname
        if iter_graph_dirs(candidate):
            return candidate
    return None


DEFAULT_DATA_DIR = resolve_default_data_dir()

DEFAULT_MAX_NODES = 50
EDGE_CAP = 5000
CATEGORY_LANE_LIMIT = 9
TOP_LABEL_LIMIT = 14
TOP_TABLE_LIMIT = 120
SIMPLE_DEFAULT_MAX_NODES = 150
SIMPLE_MAX_NODES = 500
SIMPLE_TABLE_LIMIT = 50
SIMPLE_GRAPH_NODE_LIMIT = 120
SIMPLE_GRAPH_EDGE_LIMIT = 220

ROLE_STYLES = {
    "reference": {"color": "#64748b", "symbol": "circle-open", "label": "References"},
    "paper": {"color": "#f97316", "symbol": "circle", "label": "Other papers"},
    "seed paper": {"color": "#2563eb", "symbol": "circle", "label": "Seed papers"},
    "seed + reference": {"color": "#059669", "symbol": "diamond", "label": "Seed references"},
}

PLOT_FONT = "Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"
PLOT_BG = "#fbfcff"
PAPER_BG = "#ffffff"
GRID_COLOR = "#e5e7eb"
TEXT_COLOR = "#111827"
MUTED_TEXT_COLOR = "#64748b"

APP_CSS = """
html,
body,
:root,
.dark,
.gradio-container,
.dark .gradio-container {
    color-scheme: light !important;
    --body-background-fill: #f6f8fb !important;
    --body-text-color: #111827 !important;
    --body-text-color-subdued: #475569 !important;
    --background-fill-primary: #ffffff !important;
    --background-fill-secondary: #f8fafc !important;
    --block-background-fill: #ffffff !important;
    --block-border-color: #dbe3ef !important;
    --block-label-background-fill: #ffffff !important;
    --block-label-text-color: #475569 !important;
    --block-title-text-color: #111827 !important;
    --panel-background-fill: #ffffff !important;
    --panel-border-color: #dbe3ef !important;
    --input-background-fill: #ffffff !important;
    --input-background-fill-focus: #ffffff !important;
    --input-background-fill-hover: #ffffff !important;
    --input-border-color: #cbd5e1 !important;
    --input-border-color-focus: #2563eb !important;
    --input-border-color-hover: #94a3b8 !important;
    --input-placeholder-color: #94a3b8 !important;
    --checkbox-label-background-fill: #ffffff !important;
    --checkbox-label-background-fill-hover: #f8fafc !important;
    --checkbox-label-background-fill-selected: #eff6ff !important;
    --checkbox-label-border-color: #cbd5e1 !important;
    --checkbox-label-border-color-selected: #2563eb !important;
    --checkbox-label-text-color: #334155 !important;
    --checkbox-label-text-color-selected: #1d4ed8 !important;
    --table-border-color: #e2e8f0 !important;
    --table-even-background-fill: #ffffff !important;
    --table-odd-background-fill: #f8fafc !important;
    --table-row-focus: #eff6ff !important;
    --button-primary-background-fill: #2563eb !important;
    --button-primary-background-fill-hover: #1d4ed8 !important;
    --button-primary-border-color: #2563eb !important;
    --button-primary-text-color: #ffffff !important;
    --button-secondary-background-fill: #ffffff !important;
    --button-secondary-background-fill-hover: #f8fafc !important;
    --button-secondary-border-color: #cbd5e1 !important;
    --button-secondary-text-color: #334155 !important;
}
html,
body {
    background: #f6f8fb !important;
}
.gradio-container {
    max-width: 1500px !important;
    margin: 0 auto !important;
    background: #f6f8fb !important;
    color: #111827 !important;
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
}
.gradio-container .contain {
    gap: 16px !important;
}
.gradio-container label,
.gradio-container .block-label,
.gradio-container .label-wrap,
.gradio-container .wrap,
.gradio-container .prose,
.gradio-container .markdown,
.gradio-container table,
.gradio-container input,
.gradio-container textarea,
.gradio-container select {
    color: #111827 !important;
}
.gradio-container input,
.gradio-container textarea,
.gradio-container select,
.gradio-container [role="combobox"],
.gradio-container .wrap > div {
    background: #ffffff !important;
}
.gradio-container input::placeholder,
.gradio-container textarea::placeholder {
    color: #94a3b8 !important;
}
.gradio-container button {
    border-radius: 7px !important;
    font-weight: 700 !important;
}
.gradio-container button.primary {
    background: #2563eb !important;
    border-color: #2563eb !important;
    color: #ffffff !important;
}
.gradio-container button.primary:hover {
    background: #1d4ed8 !important;
    border-color: #1d4ed8 !important;
}
.gradio-container .tabs {
    background: transparent !important;
    border: 0 !important;
}
.gradio-container .tab-nav {
    background: #eef2f7 !important;
    border: 1px solid #dbe3ef !important;
    border-radius: 8px !important;
    padding: 4px !important;
    gap: 4px !important;
}
.gradio-container .tab-nav button {
    background: transparent !important;
    border: 0 !important;
    color: #475569 !important;
}
.gradio-container .tab-nav button.selected {
    background: #ffffff !important;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.10) !important;
    color: #111827 !important;
}
.gradio-container table {
    background: #ffffff !important;
}
.gradio-container th {
    background: #f1f5f9 !important;
    color: #334155 !important;
    font-weight: 800 !important;
}
.gradio-container td {
    background: #ffffff !important;
    color: #111827 !important;
}
.gradio-container tr:nth-child(even) td {
    background: #f8fafc !important;
}
.app-header {
    border: 1px solid #dbe3ef;
    background: linear-gradient(135deg, #ffffff 0%, #f8fbff 100%);
    border-radius: 8px;
    padding: 20px 24px;
    margin: 10px 0 14px;
    box-shadow: 0 10px 28px rgba(15, 23, 42, 0.07);
}
.app-header-row {
    align-items: flex-start;
    display: flex;
    gap: 18px;
    justify-content: space-between;
}
.app-title-group {
    min-width: 0;
}
.app-eyebrow,
.section-eyebrow {
    color: #2563eb;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0;
    margin: 0 0 6px;
    text-transform: uppercase;
}
.app-header h1 {
    color: #0f172a;
    font-size: 26px;
    line-height: 1.15;
    margin: 0;
}
.app-subtitle {
    color: #475569;
    font-size: 15px;
    line-height: 1.5;
    max-width: 960px;
    margin: 8px 0 0;
}
.dataset-badge {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 8px;
    color: #1e40af;
    font-size: 12px;
    font-weight: 800;
    line-height: 1.25;
    padding: 10px 12px;
    white-space: nowrap;
}
.dashboard-grid {
    align-items: stretch !important;
    gap: 18px !important;
}
.control-panel {
    background: transparent;
    border: 0;
    padding: 0;
    box-shadow: none;
}
.control-panel > .wrap,
.panel-surface,
.reader-panel {
    background: #ffffff !important;
    border: 1px solid #dbe3ef !important;
    border-radius: 8px !important;
    box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05) !important;
}
.control-panel .tabs,
.workspace-tabs {
    margin: 0 !important;
}
.control-panel .tabitem {
    padding: 12px 0 0 !important;
}
.control-group {
    gap: 10px !important;
}
.section-title {
    color: #0f172a;
    font-size: 16px;
    font-weight: 800;
    margin: 0 0 8px;
}
.metric-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 10px;
    margin: 0;
}
.metric-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 11px 12px;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
}
.metric-label {
    color: #64748b;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0;
    margin-bottom: 4px;
    text-transform: uppercase;
}
.metric-value {
    color: #0f172a;
    font-size: 22px;
    font-weight: 800;
    line-height: 1.1;
}
.summary-block {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    color: #334155;
    font-size: 13px;
    line-height: 1.45;
    margin-top: 12px;
    padding: 12px;
}
.summary-block b {
    color: #0f172a;
}
.paper-details {
    background: #ffffff;
    border: 1px solid #dbe3ef;
    border-radius: 8px;
    padding: 16px 18px;
    color: #111827;
}
.plot-wrap {
    background: #ffffff;
    border: 1px solid #dbe3ef;
    border-radius: 8px;
    padding: 8px;
    box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
}
.reader-grid {
    display: grid;
    gap: 12px;
    grid-template-columns: repeat(3, minmax(0, 1fr));
}
.reader-section {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 13px 14px;
}
.reader-section h3 {
    color: #0f172a;
    font-size: 14px;
    margin: 0 0 10px;
}
.paper-list {
    display: grid;
    gap: 8px;
}
.paper-list-item {
    border-left: 3px solid #2563eb;
    padding-left: 9px;
}
.paper-list-title {
    color: #0f172a;
    font-size: 13px;
    font-weight: 800;
    line-height: 1.3;
}
.paper-list-meta {
    color: #64748b;
    font-size: 12px;
    line-height: 1.35;
    margin-top: 2px;
}
.diagnostic-grid {
    display: grid;
    gap: 12px;
    grid-template-columns: repeat(2, minmax(0, 1fr));
}
.diagnostic-table {
    border-collapse: collapse;
    width: 100%;
}
.diagnostic-table td {
    border-bottom: 1px solid #eef2f7;
    font-size: 13px;
    padding: 7px 4px;
    vertical-align: top;
}
.diagnostic-table td:first-child {
    color: #64748b;
    font-weight: 700;
    width: 48%;
}
.table-wrap {
    border: 1px solid #dbe3ef;
    border-radius: 8px;
    overflow: hidden;
}
.table-wrap .wrap {
    border: 0 !important;
}
.hidden-signal {
    display: none !important;
}
@media (max-width: 980px) {
    .app-header-row,
    .reader-grid,
    .diagnostic-grid {
        display: block;
    }
    .dataset-badge {
        display: inline-block;
        margin-top: 12px;
        white-space: normal;
    }
    .reader-section {
        margin-top: 12px;
    }
}
footer {
    display: none !important;
}
@media (prefers-color-scheme: dark) {
    html,
    body,
    .gradio-container {
        background: #f6f8fb !important;
        color: #111827 !important;
    }
}
"""

SIMPLE_APP_CSS = """
html,
body,
:root,
.dark,
.gradio-container,
.dark .gradio-container {
    color-scheme: light !important;
    --body-background-fill: #f8fafc !important;
    --body-text-color: #0f172a !important;
    --background-fill-primary: #ffffff !important;
    --background-fill-secondary: #f8fafc !important;
    --block-background-fill: #ffffff !important;
    --block-border-color: #dbe3ef !important;
    --input-background-fill: #ffffff !important;
    --input-border-color: #cbd5e1 !important;
    --button-primary-background-fill: #2563eb !important;
    --button-primary-background-fill-hover: #1d4ed8 !important;
    --button-primary-text-color: #ffffff !important;
}
.gradio-container {
    max-width: 1180px !important;
    margin: 0 auto !important;
    background: #f8fafc !important;
    color: #0f172a !important;
}
.simple-header {
    background: #ffffff;
    border: 1px solid #dbe3ef;
    border-radius: 8px;
    margin: 10px 0 14px;
    padding: 16px 18px;
}
.simple-header h1 {
    font-size: 24px;
    line-height: 1.2;
    margin: 0 0 4px;
}
.simple-header p {
    color: #475569;
    margin: 0;
}
.simple-stats {
    display: grid;
    gap: 10px;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    margin: 8px 0 12px;
}
.simple-stat {
    background: #ffffff;
    border: 1px solid #dbe3ef;
    border-radius: 8px;
    padding: 10px 12px;
}
.simple-stat-label {
    color: #64748b;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
}
.simple-stat-value {
    color: #0f172a;
    font-size: 20px;
    font-weight: 800;
    line-height: 1.25;
}
.simple-note {
    background: #ffffff;
    border: 1px solid #dbe3ef;
    border-radius: 8px;
    color: #334155;
    line-height: 1.45;
    margin-bottom: 12px;
    padding: 10px 12px;
}
footer {
    display: none !important;
}
@media (max-width: 760px) {
    .simple-stats {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}
"""

MAP_CLICK_JS = """
() => {
    if (window.__citationGraphClickInterval) {
        clearInterval(window.__citationGraphClickInterval);
    }
    const signalSelector = "#clicked-node-id textarea, #clicked-node-id input";
    const bindCitationMap = () => {
        const root = document.querySelector("#citation-map-plot");
        const plot = root ? root.querySelector(".js-plotly-plot") : null;
        if (!plot || plot.dataset.citationClickBound === "1" || typeof plot.on !== "function") {
            return;
        }
        plot.dataset.citationClickBound = "1";
        plot.on("plotly_click", (event) => {
            const point = event && event.points && event.points[0];
            const customdata = point ? point.customdata : null;
            const nodeId = Array.isArray(customdata) ? customdata[0] : null;
            if (!nodeId) {
                return;
            }
            const signal = document.querySelector(signalSelector);
            if (!signal) {
                return;
            }
            signal.value = nodeId;
            signal.dispatchEvent(new Event("input", { bubbles: true }));
            signal.dispatchEvent(new Event("change", { bubbles: true }));
        });
    };
    bindCitationMap();
    window.__citationGraphClickInterval = setInterval(bindCitationMap, 750);
}
"""


@dataclass
class CitationGraph:
    name: str
    metadata: dict[str, Any]
    nodes: dict[str, dict[str, Any]]
    edges: list[dict[str, Any]]
    in_neighbors: dict[str, set[str]]
    out_neighbors: dict[str, set[str]]
    in_degree: Counter[str]
    out_degree: Counter[str]
    category_counts: Counter[str]
    year_min: int
    year_max: int
    default_year_min: int
    default_year_max: int


GRAPH_CACHE: dict[tuple[str, str], CitationGraph] = {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def discover_subdomains(data_dir: Path) -> list[str]:
    data_dir = Path(data_dir)
    graph_dirs = iter_graph_dirs(data_dir)
    if not graph_dirs:
        return []

    subdomains: list[str] = []
    for graph_dir in graph_dirs:
        if graph_dir == data_dir:
            subdomains.append(data_dir.name)
        else:
            subdomains.append(graph_dir.relative_to(data_dir).as_posix())
    return sorted(subdomains)


def graph_dir_for_subdomain(data_dir: Path, subdomain: str) -> Path:
    candidate = data_dir / subdomain
    if is_graph_dir(candidate):
        return candidate
    if subdomain in {"", ".", data_dir.name} and is_graph_dir(data_dir):
        return data_dir
    return candidate


def load_graph(data_dir: Path | str, subdomain: str) -> CitationGraph:
    data_dir = Path(data_dir)
    subdomain_dir = graph_dir_for_subdomain(data_dir, subdomain)
    graph_name = subdomain if subdomain not in {"", "."} else subdomain_dir.name
    cache_key = (str(subdomain_dir.resolve()), graph_name)
    if cache_key in GRAPH_CACHE:
        return GRAPH_CACHE[cache_key]

    if not is_graph_dir(subdomain_dir):
        raise FileNotFoundError(f"Graph directory must contain nodes.jsonl and edges.jsonl: {subdomain_dir}")

    manifest_path = subdomain_dir / "manifest.json"
    metadata = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    nodes = {node["node_id"]: node for node in read_jsonl(subdomain_dir / "nodes.jsonl")}
    edges = read_jsonl(subdomain_dir / "edges.jsonl")

    in_neighbors: dict[str, set[str]] = defaultdict(set)
    out_neighbors: dict[str, set[str]] = defaultdict(set)
    in_degree: Counter[str] = Counter()
    out_degree: Counter[str] = Counter()

    for edge in edges:
        source = edge.get("source")
        target = edge.get("target")
        if source not in nodes or target not in nodes:
            continue
        out_neighbors[source].add(target)
        in_neighbors[target].add(source)
        out_degree[source] += 1
        in_degree[target] += 1

    category_counts: Counter[str] = Counter()
    years: list[int] = []
    for node_id, node in nodes.items():
        node["_role_label"] = role_label(node)
        node["_authors_text"] = author_names(node)
        node["_category"] = paper_category(node, fallback=graph_name.replace("_", ".").replace("/", "."))
        node["_graph_refs"] = in_degree[node_id]
        node["_graph_cited_by"] = out_degree[node_id]
        node["_search_blob"] = searchable_text(node)
        node["_score"] = paper_score(node, in_degree[node_id], out_degree[node_id])
        category_counts[node["_category"]] += 1
        year = safe_int(node.get("year"))
        if year:
            years.append(year)

    year_min = min(years) if years else 1900
    year_max = max(years) if years else 2026
    default_year_min = percentile_year(years, 0.02) if years else year_min
    default_year_max = percentile_year(years, 1.0) if years else year_max

    graph = CitationGraph(
        name=graph_name,
        metadata=metadata,
        nodes=nodes,
        edges=edges,
        in_neighbors=in_neighbors,
        out_neighbors=out_neighbors,
        in_degree=in_degree,
        out_degree=out_degree,
        category_counts=category_counts,
        year_min=year_min,
        year_max=year_max,
        default_year_min=default_year_min,
        default_year_max=default_year_max,
    )
    GRAPH_CACHE[cache_key] = graph
    return graph


def role_label(node: dict[str, Any]) -> str:
    roles = set(node.get("roles") or [])
    is_seed = any(role.startswith("seed") for role in roles)
    is_reference = "reference" in roles
    if is_seed and is_reference:
        return "seed + reference"
    if is_seed:
        return "seed paper"
    if is_reference:
        return "reference"
    return "paper"


def author_names(node: dict[str, Any], limit: int = 5) -> str:
    authors = node.get("authors") or []
    names = [author.get("name") for author in authors if isinstance(author, dict) and author.get("name")]
    if not names:
        return ""
    suffix = "" if len(names) <= limit else f" +{len(names) - limit}"
    return ", ".join(names[:limit]) + suffix


def paper_category(node: dict[str, Any], fallback: str) -> str:
    if node.get("arxiv_primary_category"):
        return str(node["arxiv_primary_category"])
    categories = node.get("arxiv_categories") or []
    if categories:
        return str(categories[0])
    fields = node.get("fields_of_study") or []
    if fields:
        return str(fields[0])
    return fallback


def searchable_text(node: dict[str, Any]) -> str:
    pieces = [
        node.get("node_id"),
        node.get("title"),
        node.get("arxiv_id"),
        node.get("doi"),
        node.get("venue"),
        node.get("abstract"),
        node.get("_authors_text"),
        node.get("arxiv_primary_category"),
        " ".join(node.get("arxiv_categories") or []),
    ]
    return " ".join(str(piece) for piece in pieces if piece).lower()


def paper_score(node: dict[str, Any], graph_refs: int, graph_cited_by: int) -> float:
    role_bonus = 18 if role_label(node).startswith("seed") else 0
    role_bonus += 8 if role_label(node) == "seed + reference" else 0
    citation_count = safe_int(node.get("citation_count")) or 0
    influential = safe_int(node.get("influential_citation_count")) or 0
    return (
        role_bonus
        + graph_cited_by * 5
        + graph_refs * 1.5
        + math.log1p(citation_count) * 2
        + math.log1p(influential) * 1.2
    )


def safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def percentile_year(years: list[int], quantile: float) -> int:
    if not years:
        return 1900
    ordered = sorted(years)
    index = round((len(ordered) - 1) * max(0.0, min(1.0, quantile)))
    return ordered[index]


def selected_node_ids(
    graph: CitationGraph,
    mode: str,
    max_nodes: int,
    min_year: int | None,
    max_year: int | None,
    search_text: str,
    selected_node: str | None,
) -> set[str]:
    eligible = {
        node_id
        for node_id, node in graph.nodes.items()
        if year_allowed(node, min_year=min_year, max_year=max_year)
    }
    max_nodes = max(25, int(max_nodes or DEFAULT_MAX_NODES))
    pinned: set[str] = set()
    protected: set[str] = set()
    if selected_node in eligible:
        pinned.add(selected_node)
        protected.add(selected_node)

    query_tokens = [token for token in re.split(r"\s+", (search_text or "").strip().lower()) if token]
    if query_tokens:
        matches = {
            node_id
            for node_id, node in graph.nodes.items()
            if node_id in eligible and all(token in node.get("_search_blob", "") for token in query_tokens)
        }
        protected |= matches
        ids = expand_neighbors(graph, matches, depth=1) & eligible
        return cap_nodes(graph, ids, max_nodes=max_nodes, protected=protected, pinned=pinned)

    if mode == "Selected paper neighborhood" and selected_node in graph.nodes:
        protected.add(selected_node)
        ids = expand_neighbors(graph, {selected_node}, depth=2) & eligible
        return cap_nodes(graph, ids, max_nodes=max_nodes, protected=protected, pinned=pinned)

    seed_ids = {
        node_id for node_id, node in graph.nodes.items() if node_id in eligible and role_label(node).startswith("seed")
    }

    if mode == "Seed papers and references":
        ids = set(seed_ids)
        for seed_id in seed_ids:
            ids.update(graph.in_neighbors.get(seed_id, set()))
            ids.update(graph.out_neighbors.get(seed_id, set()))
        ids |= protected
        protected |= seed_ids
        return cap_nodes(graph, ids & eligible, max_nodes=max_nodes, protected=protected, pinned=pinned)

    if mode == "Most cited within graph":
        ids = top_nodes(graph, eligible, key=lambda node_id: graph.out_degree[node_id], limit=max_nodes)
        return cap_nodes(graph, set(ids) | protected, max_nodes=max_nodes, protected=protected, pinned=pinned)

    if mode == "Most connected papers":
        ids = top_nodes(
            graph,
            eligible,
            key=lambda node_id: graph.in_degree[node_id] + graph.out_degree[node_id],
            limit=max_nodes,
        )
        return cap_nodes(graph, set(ids) | protected, max_nodes=max_nodes, protected=protected, pinned=pinned)

    ids = set(seed_ids)
    ids |= protected
    ids.update(
        top_nodes(
            graph,
            eligible - ids,
            key=lambda node_id: graph.nodes[node_id].get("_score", 0),
            limit=max_nodes - len(ids),
        )
    )
    protected |= seed_ids
    return cap_nodes(graph, ids, max_nodes=max_nodes, protected=protected, pinned=pinned)


def year_allowed(node: dict[str, Any], min_year: int | None, max_year: int | None) -> bool:
    year = safe_int(node.get("year"))
    if year is None:
        return True
    if min_year is not None and year < min_year:
        return False
    if max_year is not None and year > max_year:
        return False
    return True


def expand_neighbors(graph: CitationGraph, roots: set[str], depth: int) -> set[str]:
    seen = set(roots)
    frontier = set(roots)
    for _ in range(depth):
        next_frontier: set[str] = set()
        for node_id in frontier:
            next_frontier.update(graph.in_neighbors.get(node_id, set()))
            next_frontier.update(graph.out_neighbors.get(node_id, set()))
        next_frontier -= seen
        seen.update(next_frontier)
        frontier = next_frontier
    return seen


def top_nodes(
    graph: CitationGraph,
    ids: set[str],
    key: Any,
    limit: int,
) -> list[str]:
    if limit <= 0:
        return []
    return sorted(ids, key=lambda node_id: (key(node_id), graph.nodes[node_id].get("_score", 0)), reverse=True)[:limit]


def cap_nodes(
    graph: CitationGraph,
    ids: set[str],
    max_nodes: int,
    protected: set[str],
    pinned: set[str] | None = None,
) -> set[str]:
    if len(ids) <= max_nodes:
        return ids

    def score(node_id: str) -> float:
        return graph.nodes[node_id].get("_score", 0)

    pinned_ids = {node_id for node_id in (pinned or set()) if node_id in ids}
    protected_ids = {node_id for node_id in protected if node_id in ids} - pinned_ids
    remaining = ids - pinned_ids - protected_ids
    kept: list[str] = []

    for candidates in (pinned_ids, protected_ids, remaining):
        if len(kept) >= max_nodes:
            break
        keep_count = max_nodes - len(kept)
        kept.extend(sorted(candidates, key=score, reverse=True)[:keep_count])

    return set(kept)


def layout_positions(graph: CitationGraph, node_ids: set[str]) -> tuple[dict[str, tuple[float, float]], list[str]]:
    category_counts = Counter(graph.nodes[node_id].get("_category") or "unknown" for node_id in node_ids)
    categories = [category for category, _ in category_counts.most_common(CATEGORY_LANE_LIMIT)]
    if not categories:
        categories = ["unknown"]
    if len(category_counts) > len(categories):
        categories.append("other categories")
    category_index = {category: index for index, category in enumerate(categories)}

    positions: dict[str, tuple[float, float]] = {}
    for node_id in node_ids:
        node = graph.nodes[node_id]
        year = safe_int(node.get("year")) or graph.year_min
        category = node.get("_category") or "unknown"
        lane = category if category in category_index else "other categories"
        x = year + stable_jitter(f"{node_id}:x", scale=0.18)
        y = category_index.get(lane, 0) + stable_jitter(f"{node_id}:y", scale=0.56)
        positions[node_id] = (x, y)
    return positions, categories


def stable_jitter(text: str, scale: float) -> float:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    return (value - 0.5) * scale


def filtered_edges(graph: CitationGraph, node_ids: set[str]) -> list[dict[str, Any]]:
    edges = [
        edge
        for edge in graph.edges
        if edge.get("source") in node_ids and edge.get("target") in node_ids
    ]
    if len(edges) <= EDGE_CAP:
        return edges
    return sorted(
        edges,
        key=lambda edge: graph.nodes[edge["source"]].get("_score", 0)
        + graph.nodes[edge["target"]].get("_score", 0),
        reverse=True,
    )[:EDGE_CAP]


def edge_importance(graph: CitationGraph, edge: dict[str, Any]) -> float:
    source = edge.get("source")
    target = edge.get("target")
    if source not in graph.nodes or target not in graph.nodes:
        return 0
    return graph.nodes[source].get("_score", 0) + graph.nodes[target].get("_score", 0)


def edge_curve_points(
    edge: dict[str, Any],
    positions: dict[str, tuple[float, float]],
    point_count: int = 6,
) -> tuple[list[float], list[float]]:
    source = edge["source"]
    target = edge["target"]
    x0, y0 = positions[source]
    x1, y1 = positions[target]
    mid_x = (x0 + x1) / 2
    mid_y = (y0 + y1) / 2 + stable_jitter(f"{source}->{target}:curve", scale=0.74)

    xs: list[float] = []
    ys: list[float] = []
    for index in range(point_count):
        t = index / (point_count - 1)
        omt = 1 - t
        xs.append(omt * omt * x0 + 2 * omt * t * mid_x + t * t * x1)
        ys.append(omt * omt * y0 + 2 * omt * t * mid_y + t * t * y1)
    return xs, ys


def add_edge_trace(
    figure: Any,
    edges: list[dict[str, Any]],
    positions: dict[str, tuple[float, float]],
    name: str,
    color: str,
    width: float,
    opacity: float,
) -> None:
    import plotly.graph_objects as go

    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for edge in edges:
        if edge.get("source") not in positions or edge.get("target") not in positions:
            continue
        xs, ys = edge_curve_points(edge, positions)
        edge_x.extend(xs + [None])
        edge_y.extend(ys + [None])

    if not edge_x:
        return

    figure.add_trace(
        go.Scattergl(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line={"width": width, "color": color},
            opacity=opacity,
            hoverinfo="skip",
            name=name,
            showlegend=False,
        )
    )


def make_figure(
    graph: CitationGraph,
    node_ids: set[str],
    show_edges: bool,
    size_by: str,
    selected_node: str | None = None,
    edges: list[dict[str, Any]] | None = None,
) -> Any:
    import plotly.graph_objects as go

    positions, categories = layout_positions(graph, node_ids)
    edges = filtered_edges(graph, node_ids) if edges is None else edges

    figure = go.Figure()
    for index in range(len(categories)):
        fill = "#f8fafc" if index % 2 == 0 else "#ffffff"
        figure.add_hrect(y0=index - 0.5, y1=index + 0.5, fillcolor=fill, opacity=0.72, line_width=0, layer="below")

    if show_edges and edges:
        ranked_edges = sorted(edges, key=lambda edge: edge_importance(graph, edge), reverse=True)
        strong_count = min(180, len(ranked_edges))
        add_edge_trace(
            figure,
            ranked_edges[strong_count:],
            positions,
            name="citation links",
            color="rgba(100,116,139,0.18)",
            width=0.65,
            opacity=0.62,
        )
        add_edge_trace(
            figure,
            ranked_edges[:strong_count],
            positions,
            name="prominent citation links",
            color="rgba(15,23,42,0.42)",
            width=1.0,
            opacity=0.82,
        )

    for role, style in ROLE_STYLES.items():
        role_ids = [node_id for node_id in node_ids if graph.nodes[node_id].get("_role_label") == role]
        if not role_ids:
            continue
        figure.add_trace(
            go.Scattergl(
                x=[positions[node_id][0] for node_id in role_ids],
                y=[positions[node_id][1] for node_id in role_ids],
                mode="markers",
                marker={
                    "size": [marker_size(graph.nodes[node_id], size_by) for node_id in role_ids],
                    "color": style["color"],
                    "symbol": style["symbol"],
                    "line": {"width": 1.2, "color": "#ffffff"},
                    "opacity": 0.9,
                },
                hovertemplate="%{customdata[1]}<extra></extra>",
                customdata=[[node_id, hover_html(graph.nodes[node_id])] for node_id in role_ids],
                name=style["label"],
            )
        )

    if selected_node in node_ids:
        selected = graph.nodes[selected_node]
        x, y = positions[selected_node]
        figure.add_trace(
            go.Scattergl(
                x=[x],
                y=[y],
                mode="markers",
                marker={
                    "size": marker_size(selected, size_by) + 12,
                    "color": "#db2777",
                    "symbol": "circle-open",
                    "line": {"width": 3.2, "color": "#db2777"},
                },
                hovertemplate="%{customdata[1]}<extra></extra>",
                customdata=[[selected_node, hover_html(selected)]],
                name="Selected paper",
                showlegend=True,
            )
        )

    label_ids: list[str] = []
    if selected_node in node_ids:
        label_ids.append(selected_node)
    for node_id in sorted(node_ids, key=lambda item: graph.nodes[item].get("_score", 0), reverse=True):
        if node_id not in label_ids:
            label_ids.append(node_id)
        if len(label_ids) >= TOP_LABEL_LIMIT:
            break

    for node_id in label_ids:
        if node_id not in positions:
            continue
        x, y = positions[node_id]
        node = graph.nodes[node_id]
        role = node.get("_role_label")
        border = ROLE_STYLES.get(role, {}).get("color", "#64748b")
        figure.add_annotation(
            x=x,
            y=y + 0.30,
            text=html.escape(short_title(node, 38)),
            showarrow=False,
            xanchor="center",
            yanchor="bottom",
            font={"size": 10, "color": TEXT_COLOR, "family": PLOT_FONT},
            bgcolor="rgba(255,255,255,0.90)",
            bordercolor=border,
            borderwidth=1,
            borderpad=3,
        )

    title = f"{graph.name.replace('_', '.')} citation graph"
    edge_phrase = f"{len(edges):,} citation links" if show_edges else f"{len(edges):,} links hidden on map"
    subtitle = (
        f"{len(node_ids):,} papers shown · {edge_phrase} · "
        "left-to-right flow is referenced paper to citing paper"
    )
    if positions:
        shown_years = [position[0] for position in positions.values()]
        x_min = min(shown_years)
        x_max = max(shown_years)
        x_pad = max(1, (x_max - x_min) * 0.035)
    else:
        x_min = graph.year_min
        x_max = graph.year_max
        x_pad = 1

    figure.update_layout(
        title={"text": f"{title}<br><sup>{subtitle}</sup>", "x": 0.01},
        height=780,
        margin={"l": 76, "r": 26, "t": 84, "b": 56},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "font": {"size": 12},
        },
        font={"family": PLOT_FONT, "color": TEXT_COLOR},
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        hovermode="closest",
        clickmode="event+select",
        template="plotly_white",
        uirevision=graph.name,
        xaxis={
            "title": "Publication year",
            "gridcolor": GRID_COLOR,
            "linecolor": "#cbd5e1",
            "showline": True,
            "zeroline": False,
            "range": [x_min - x_pad, x_max + x_pad],
        },
        yaxis={
            "title": "Primary category lane",
            "tickmode": "array",
            "tickvals": list(range(len(categories))),
            "ticktext": categories,
            "gridcolor": "rgba(226,232,240,0.72)",
            "linecolor": "#cbd5e1",
            "range": [-0.65, len(categories) - 0.35],
            "showline": True,
            "automargin": True,
            "zeroline": False,
        },
    )
    return figure


def marker_size(node: dict[str, Any], size_by: str) -> float:
    if size_by == "Semantic Scholar citations":
        value = safe_int(node.get("citation_count")) or 0
        return min(34, 8 + math.log1p(value) * 2.05)
    if size_by == "In-graph cited by count":
        value = safe_int(node.get("_graph_cited_by")) or 0
        return min(34, 8 + math.sqrt(value) * 4.0)
    value = (safe_int(node.get("_graph_refs")) or 0) + (safe_int(node.get("_graph_cited_by")) or 0)
    return min(34, 8 + math.sqrt(value) * 3.2)


def format_count(value: Any) -> str:
    number = safe_int(value)
    return "0" if number is None else f"{number:,}"


def format_percent(value: float) -> str:
    if value < 0.01:
        return "0.00%"
    return f"{value:.2f}%"


def role_color(role: str | None) -> str:
    return ROLE_STYLES.get(role or "", {}).get("color", "#64748b")


def style_companion_figure(figure: Any, height: int) -> Any:
    figure.update_layout(
        height=height,
        margin={"l": 64, "r": 28, "t": 72, "b": 52},
        font={"family": PLOT_FONT, "color": TEXT_COLOR},
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        template="plotly_white",
        hoverlabel={"font": {"family": PLOT_FONT}},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    figure.update_xaxes(gridcolor=GRID_COLOR, zeroline=False, linecolor="#cbd5e1", showline=True)
    figure.update_yaxes(gridcolor=GRID_COLOR, zeroline=False, linecolor="#cbd5e1", showline=True)
    return figure


def make_empty_figure(title: str, message: str, height: int = 430) -> Any:
    import plotly.graph_objects as go

    figure = go.Figure()
    figure.add_annotation(
        text=html.escape(message),
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 15, "color": MUTED_TEXT_COLOR, "family": PLOT_FONT},
    )
    figure.update_layout(title={"text": title, "x": 0.01}, xaxis={"visible": False}, yaxis={"visible": False})
    return style_companion_figure(figure, height=height)


def binned_counts(values: list[int]) -> tuple[list[str], list[int]]:
    bins = [
        ("0", 0, 0),
        ("1", 1, 1),
        ("2-4", 2, 4),
        ("5-9", 5, 9),
        ("10-19", 10, 19),
        ("20+", 20, None),
    ]
    counts = []
    for _, lower, upper in bins:
        if upper is None:
            counts.append(sum(value >= lower for value in values))
        else:
            counts.append(sum(lower <= value <= upper for value in values))
    return [label for label, _, _ in bins], counts


def make_summary_figure(graph: CitationGraph, node_ids: set[str], edges: list[dict[str, Any]]) -> Any:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if not node_ids:
        return make_empty_figure("Selection overview", "No papers match the current filters.")

    nodes = [graph.nodes[node_id] for node_id in node_ids]
    years = sorted({safe_int(node.get("year")) for node in nodes if safe_int(node.get("year")) is not None})
    categories = Counter(node.get("_category") or "unknown" for node in nodes)
    roles = Counter(node.get("_role_label") or "paper" for node in nodes)
    cited_by_values = [safe_int(node.get("_graph_cited_by")) or 0 for node in nodes]
    degree_labels, degree_counts = binned_counts(cited_by_values)

    figure = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "xy"}, {"type": "xy"}], [{"type": "domain"}, {"type": "xy"}]],
        subplot_titles=(
            "Publication years",
            "Primary categories",
            "Role composition",
            "In-graph cited-by distribution",
        ),
        horizontal_spacing=0.15,
        vertical_spacing=0.20,
    )

    for role, style in ROLE_STYLES.items():
        year_counts = Counter(
            safe_int(node.get("year"))
            for node in nodes
            if node.get("_role_label") == role and safe_int(node.get("year")) is not None
        )
        if not year_counts:
            continue
        figure.add_trace(
            go.Bar(
                x=years,
                y=[year_counts.get(year, 0) for year in years],
                marker_color=style["color"],
                name=style["label"],
                hovertemplate="Year %{x}<br>Papers %{y}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    category_items = list(reversed(categories.most_common(12)))
    figure.add_trace(
        go.Bar(
            x=[count for _, count in category_items],
            y=[category for category, _ in category_items],
            orientation="h",
            marker_color="#2563eb",
            showlegend=False,
            hovertemplate="Category %{y}<br>Papers %{x}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    role_items = [(role, roles[role]) for role in ROLE_STYLES if roles.get(role)]
    figure.add_trace(
        go.Pie(
            labels=[ROLE_STYLES[role]["label"] for role, _ in role_items],
            values=[count for _, count in role_items],
            hole=0.58,
            marker={"colors": [ROLE_STYLES[role]["color"] for role, _ in role_items]},
            textinfo="percent",
            hovertemplate="%{label}<br>Papers %{value}<extra></extra>",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    figure.add_trace(
        go.Bar(
            x=degree_labels,
            y=degree_counts,
            marker_color="#059669",
            showlegend=False,
            hovertemplate="In-graph cited by %{x}<br>Papers %{y}<extra></extra>",
        ),
        row=2,
        col=2,
    )

    edge_density = 0.0 if len(node_ids) <= 1 else len(edges) / (len(node_ids) * (len(node_ids) - 1)) * 100
    figure.update_layout(
        title={
            "text": (
                f"Selection overview<br><sup>{len(node_ids):,} papers · {len(edges):,} links · "
                f"{format_percent(edge_density)} directed density</sup>"
            ),
            "x": 0.01,
        },
        barmode="stack",
    )
    figure.update_xaxes(title_text="Year", row=1, col=1)
    figure.update_yaxes(title_text="Papers", row=1, col=1)
    figure.update_xaxes(title_text="Papers", row=1, col=2)
    figure.update_xaxes(title_text="In-graph cited by", row=2, col=2)
    figure.update_yaxes(title_text="Papers", row=2, col=2)
    return style_companion_figure(figure, height=680)


def make_temporal_flow_figure(graph: CitationGraph, edges: list[dict[str, Any]]) -> Any:
    import plotly.graph_objects as go

    year_pairs: Counter[tuple[int, int]] = Counter()
    for edge in edges:
        source = edge.get("source")
        target = edge.get("target")
        if source not in graph.nodes or target not in graph.nodes:
            continue
        source_year = safe_int(edge.get("source_year")) or safe_int(graph.nodes[source].get("year"))
        target_year = safe_int(edge.get("target_year")) or safe_int(graph.nodes[target].get("year"))
        if source_year is None or target_year is None:
            continue
        year_pairs[(source_year, target_year)] += 1

    if not year_pairs:
        return make_empty_figure("Temporal citation flow", "No dated citation links match the current filters.")

    min_year = min(min(source_year, target_year) for source_year, target_year in year_pairs)
    max_year = max(max(source_year, target_year) for source_year, target_year in year_pairs)
    years = list(range(min_year, max_year + 1))
    year_index = {year: index for index, year in enumerate(years)}
    z = [[0 for _ in years] for _ in years]
    for (source_year, target_year), count in year_pairs.items():
        z[year_index[source_year]][year_index[target_year]] = count

    figure = go.Figure()
    figure.add_trace(
        go.Heatmap(
            x=years,
            y=years,
            z=z,
            colorscale=[
                [0.0, "#f8fafc"],
                [0.25, "#dbeafe"],
                [0.60, "#60a5fa"],
                [0.82, "#2563eb"],
                [1.0, "#ea580c"],
            ],
            colorbar={"title": "Links", "len": 0.82},
            hovertemplate="Referenced year %{y}<br>Citing year %{x}<br>Links %{z}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[min_year, max_year],
            y=[min_year, max_year],
            mode="lines",
            line={"color": "rgba(15,23,42,0.52)", "dash": "dot", "width": 1.2},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    figure.update_layout(
        title={
            "text": (
                "Temporal citation flow<br><sup>Cell intensity counts references from row year "
                "to citing papers in column year</sup>"
            ),
            "x": 0.01,
        }
    )
    figure.update_xaxes(title_text="Citing paper year", constrain="domain")
    figure.update_yaxes(title_text="Referenced paper year", scaleanchor="x", scaleratio=1)
    return style_companion_figure(figure, height=680)


def make_influence_figure(graph: CitationGraph, node_ids: set[str]) -> Any:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if not node_ids:
        return make_empty_figure("Influential papers", "No papers match the current filters.")

    top_graph = list(
        reversed(
            sorted(
                node_ids,
                key=lambda node_id: (
                    safe_int(graph.nodes[node_id].get("_graph_cited_by")) or 0,
                    graph.nodes[node_id].get("_score", 0),
                ),
                reverse=True,
            )[:18]
        )
    )
    top_s2 = list(
        reversed(
            sorted(
                node_ids,
                key=lambda node_id: (
                    safe_int(graph.nodes[node_id].get("citation_count")) or 0,
                    graph.nodes[node_id].get("_score", 0),
                ),
                reverse=True,
            )[:18]
        )
    )

    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Cited by papers in this graph", "Semantic Scholar citations"),
        horizontal_spacing=0.22,
    )

    figure.add_trace(
        go.Bar(
            x=[safe_int(graph.nodes[node_id].get("_graph_cited_by")) or 0 for node_id in top_graph],
            y=[short_title(graph.nodes[node_id], 44) for node_id in top_graph],
            orientation="h",
            marker_color=[role_color(graph.nodes[node_id].get("_role_label")) for node_id in top_graph],
            customdata=[hover_html(graph.nodes[node_id]) for node_id in top_graph],
            hovertemplate="%{customdata}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Bar(
            x=[safe_int(graph.nodes[node_id].get("citation_count")) or 0 for node_id in top_s2],
            y=[short_title(graph.nodes[node_id], 44) for node_id in top_s2],
            orientation="h",
            marker_color=[role_color(graph.nodes[node_id].get("_role_label")) for node_id in top_s2],
            customdata=[hover_html(graph.nodes[node_id]) for node_id in top_s2],
            hovertemplate="%{customdata}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    figure.update_layout(
        title={
            "text": "Influential papers<br><sup>Within-graph centrality and broad citation reach can disagree</sup>",
            "x": 0.01,
        }
    )
    figure.update_xaxes(title_text="In-graph cited by", row=1, col=1)
    figure.update_xaxes(title_text="S2 citations", row=1, col=2)
    figure.update_yaxes(automargin=True)
    return style_companion_figure(figure, height=720)


def hover_html(node: dict[str, Any]) -> str:
    title = html.escape(str(node.get("title") or "Untitled"))
    authors = html.escape(node.get("_authors_text") or "Unknown authors")
    venue = html.escape(str(node.get("venue") or ""))
    arxiv_id = html.escape(str(node.get("arxiv_id") or ""))
    category = html.escape(str(node.get("_category") or "unknown"))
    return (
        f"<b>{title}</b><br>"
        f"{authors}<br>"
        f"Year: {node.get('year') or 'unknown'} | Role: {node.get('_role_label')} | Category: {category}<br>"
        f"arXiv: {arxiv_id or 'n/a'} | Venue: {venue or 'n/a'}<br>"
        f"S2 citations: {format_count(node.get('citation_count'))} | "
        f"Influential: {format_count(node.get('influential_citation_count'))}<br>"
        f"In-graph cited by: {node.get('_graph_cited_by') or 0} | "
        f"In-graph refs: {node.get('_graph_refs') or 0}"
    )


def short_title(node: dict[str, Any], max_len: int = 42) -> str:
    title = str(node.get("title") or "Untitled")
    return title if len(title) <= max_len else title[: max_len - 1] + "..."


def compact_label(value: Any, max_len: int = 32) -> str:
    text = str(value or "unknown")
    return text if len(text) <= max_len else text[: max_len - 1] + "..."


def ranked_display_node_ids(graph: CitationGraph, node_ids: set[str], limit: int = TOP_TABLE_LIMIT) -> list[str]:
    return sorted(node_ids, key=lambda item: graph.nodes[item].get("_score", 0), reverse=True)[:limit]


def table_rows(graph: CitationGraph, node_ids: set[str], limit: int = TOP_TABLE_LIMIT) -> list[list[Any]]:
    rows = []
    for node_id in ranked_display_node_ids(graph, node_ids, limit=limit):
        node = graph.nodes[node_id]
        rows.append(
            [
                node.get("title") or "Untitled",
                node.get("year") or "",
                node.get("_role_label"),
                node.get("_category") or "",
                node.get("_graph_cited_by") or 0,
                node.get("_graph_refs") or 0,
                node.get("citation_count") or 0,
                node.get("influential_citation_count") or 0,
                node.get("arxiv_id") or "",
                node.get("venue") or "",
            ]
        )
    return rows


def table_node_ids(graph: CitationGraph, node_ids: set[str], limit: int = TOP_TABLE_LIMIT) -> list[str]:
    return ranked_display_node_ids(graph, node_ids, limit=limit)


def paper_choice(graph: CitationGraph, node_id: str) -> tuple[str, str]:
    node = graph.nodes[node_id]
    label = f"[{node.get('year') or '?'}] {short_title(node, 86)}"
    arxiv_id = node.get("arxiv_id")
    if arxiv_id:
        label += f" ({arxiv_id})"
    return label, node_id


def paper_choices(graph: CitationGraph, limit: int = 1200) -> list[tuple[str, str]]:
    ranked_ids = sorted(graph.nodes, key=lambda node_id: graph.nodes[node_id].get("_score", 0), reverse=True)
    return [paper_choice(graph, node_id) for node_id in ranked_ids[:limit]]


def choices_with_selected(graph: CitationGraph, selected_node: str | None) -> list[tuple[str, str]]:
    choices = paper_choices(graph)
    if selected_node and selected_node in graph.nodes and selected_node not in {value for _, value in choices}:
        choices.append(paper_choice(graph, selected_node))
    return choices


def neighbor_rows(graph: CitationGraph, selected_node: str | None, limit: int = 80) -> list[list[Any]]:
    if not selected_node or selected_node not in graph.nodes:
        return []

    related: list[tuple[str, str]] = []
    references = sorted(
        graph.in_neighbors.get(selected_node, set()),
        key=lambda node_id: graph.nodes[node_id].get("_score", 0),
        reverse=True,
    )
    citing = sorted(
        graph.out_neighbors.get(selected_node, set()),
        key=lambda node_id: graph.nodes[node_id].get("_score", 0),
        reverse=True,
    )
    related.extend(("Reference", node_id) for node_id in references[: limit // 2])
    related.extend(("Citing paper", node_id) for node_id in citing[: limit // 2])

    rows: list[list[Any]] = []
    for relation, node_id in related[:limit]:
        node = graph.nodes[node_id]
        rows.append(
            [
                relation,
                node.get("title") or "Untitled",
                node.get("year") or "",
                node.get("_role_label"),
                node.get("_category") or "",
                node.get("_graph_cited_by") or 0,
                node.get("citation_count") or 0,
                node.get("arxiv_id") or "",
            ]
        )
    return rows


def paper_list_html(graph: CitationGraph, node_ids: list[str], empty_text: str) -> str:
    if not node_ids:
        return f'<div class="paper-list-meta">{html.escape(empty_text)}</div>'
    items = []
    for node_id in node_ids:
        node = graph.nodes[node_id]
        role = node.get("_role_label") or "paper"
        color = role_color(role)
        title = html.escape(short_title(node, 82))
        meta = (
            f"{node.get('year') or 'unknown'} · {html.escape(role)} · "
            f"{format_count(node.get('_graph_cited_by'))} in-graph cites · "
            f"{format_count(node.get('citation_count'))} S2 cites"
        )
        items.append(
            f"""
            <div class="paper-list-item" style="border-left-color: {color}">
              <div class="paper-list-title">{title}</div>
              <div class="paper-list-meta">{meta}</div>
            </div>
            """
        )
    return '<div class="paper-list">' + "\n".join(items) + "</div>"


def reader_path_html(graph: CitationGraph, node_ids: set[str], selected_node: str | None) -> str:
    selected = {selected_node} if selected_node in graph.nodes else set()
    seed_ids = [
        node_id
        for node_id in ranked_display_node_ids(graph, node_ids, limit=80)
        if graph.nodes[node_id].get("_role_label", "").startswith("seed")
    ][:5]
    references = sorted(
        node_ids,
        key=lambda node_id: (
            graph.nodes[node_id].get("_role_label") == "reference",
            safe_int(graph.nodes[node_id].get("_graph_cited_by")) or 0,
            graph.nodes[node_id].get("_score", 0),
        ),
        reverse=True,
    )[:5]
    recent = sorted(
        node_ids - selected,
        key=lambda node_id: (
            safe_int(graph.nodes[node_id].get("year")) or 0,
            safe_int(graph.nodes[node_id].get("_graph_cited_by")) or 0,
            graph.nodes[node_id].get("_score", 0),
        ),
        reverse=True,
    )[:5]

    return f"""
<div class="reader-grid">
  <div class="reader-section">
    <h3>Seed Anchors</h3>
    {paper_list_html(graph, seed_ids, "No seed papers in the current selection.")}
  </div>
  <div class="reader-section">
    <h3>Core References</h3>
    {paper_list_html(graph, references, "No references in the current selection.")}
  </div>
  <div class="reader-section">
    <h3>Recent Connectors</h3>
    {paper_list_html(graph, recent, "No recent papers in the current selection.")}
  </div>
</div>
"""


def paper_details(graph: CitationGraph, selected_node: str | None) -> str:
    if not selected_node or selected_node not in graph.nodes:
        return "No paper selected."

    node = graph.nodes[selected_node]
    title = node.get("title") or "Untitled"
    authors = node.get("_authors_text") or "Unknown authors"
    abstract = node.get("abstract") or ""
    abstract = abstract[:1400] + "..." if len(abstract) > 1400 else abstract

    links = []
    if node.get("arxiv_id"):
        links.append(f"[arXiv](https://arxiv.org/abs/{node['arxiv_id']})")
    if node.get("url"):
        links.append(f"[Semantic Scholar]({node['url']})")
    if node.get("doi"):
        links.append(f"[DOI](https://doi.org/{node['doi']})")
    links_text = " | ".join(links) if links else "No external links available"

    references = ranked_neighbor_list(graph, graph.in_neighbors.get(selected_node, set()), limit=8)
    citing = ranked_neighbor_list(graph, graph.out_neighbors.get(selected_node, set()), limit=8)

    return f"""### {escape_markdown(title)}

{escape_markdown(authors)}

{links_text}

| Metric | Value |
| --- | ---: |
| Year | {node.get("year") or "unknown"} |
| Publication date | {node.get("publication_date") or "unknown"} |
| Role | {node.get("_role_label")} |
| Primary category | {node.get("_category") or ""} |
| Venue | {escape_markdown(node.get("venue") or "unknown")} |
| Semantic Scholar citations | {format_count(node.get("citation_count"))} |
| Influential citations | {format_count(node.get("influential_citation_count"))} |
| Semantic Scholar reference count | {format_count(node.get("reference_count"))} |
| In-graph papers citing this | {node.get("_graph_cited_by") or 0} |
| In-graph references cited by this | {node.get("_graph_refs") or 0} |

**Papers this cites in the source graph**

{references}

**Papers citing this in the source graph**

{citing}

**Abstract**

{escape_markdown(abstract) if abstract else "No abstract available."}
"""


def ranked_neighbor_list(graph: CitationGraph, node_ids: set[str], limit: int) -> str:
    if not node_ids:
        return "None in the source graph data."
    lines = []
    for node_id in sorted(node_ids, key=lambda item: graph.nodes[item].get("_score", 0), reverse=True)[:limit]:
        node = graph.nodes[node_id]
        year = node.get("year") or "?"
        cited_by = safe_int(node.get("_graph_cited_by")) or 0
        lines.append(f"- [{year}] {escape_markdown(short_title(node, 96))} ({cited_by} in-graph citing papers)")
    return "\n".join(lines)


def escape_markdown(text: Any) -> str:
    return str(text).replace("|", "\\|")


def diagnostic_table(rows: list[tuple[str, Any]]) -> str:
    table_rows_html = "\n".join(
        f"<tr><td>{html.escape(label)}</td><td>{html.escape(str(value if value not in (None, '') else 'n/a'))}</td></tr>"
        for label, value in rows
    )
    return f'<table class="diagnostic-table">{table_rows_html}</table>'


def build_diagnostics_html(graph: CitationGraph) -> str:
    metadata = graph.metadata or {}
    skipped = metadata.get("skipped") or {}
    graph_stats = metadata.get("graph_stats") or {}
    source_rows = [
        ("Graph", graph.name),
        ("Query", metadata.get("query")),
        ("Candidate source", metadata.get("candidate_source")),
        ("S2 query", metadata.get("s2_search_effective_query") or metadata.get("s2_search_query")),
        ("Fields of study", ", ".join(metadata.get("s2_fields_of_study_effective") or []) or metadata.get("s2_fields_of_study")),
        ("Created", metadata.get("created_at")),
    ]
    build_rows = [
        ("Nodes", format_count(len(graph.nodes))),
        ("Edges", format_count(len(graph.edges))),
        ("Seed papers", format_count(metadata.get("seed_paper_count"))),
        ("Citation depth", metadata.get("citation_depth")),
        ("Reference cap", metadata.get("max_references_per_paper")),
        ("Target node count", graph_stats.get("target_node_count")),
    ]
    skipped_rows = [
        ("Nontemporal edges", format_count(skipped.get("nontemporal_edges"))),
        ("Duplicate edges", format_count(skipped.get("duplicate_edges"))),
        ("Self edges", format_count(skipped.get("self_edges"))),
        ("Reference breadth cap", format_count(skipped.get("references_breadth_cap"))),
        ("Expansion breadth cap", format_count(skipped.get("expansion_breadth_cap"))),
        ("Target node budget", format_count(skipped.get("target_node_budget"))),
    ]
    crawl_rows = [
        ("arXiv candidates", format_count(metadata.get("arxiv_candidate_count"))),
        ("S2 matches", format_count(metadata.get("s2_match_count"))),
        ("Expanded papers", format_count(graph_stats.get("expanded_paper_count"))),
        ("Raw references seen", format_count(graph_stats.get("raw_references_seen"))),
        ("References inspected", format_count(graph_stats.get("references_inspected"))),
        ("Budget reached", graph_stats.get("target_node_budget_reached")),
    ]

    return f"""
<div class="diagnostic-grid">
  <div class="reader-section"><h3>Source</h3>{diagnostic_table(source_rows)}</div>
  <div class="reader-section"><h3>Build</h3>{diagnostic_table(build_rows)}</div>
  <div class="reader-section"><h3>Skipped</h3>{diagnostic_table(skipped_rows)}</div>
  <div class="reader-section"><h3>Crawl</h3>{diagnostic_table(crawl_rows)}</div>
</div>
"""


def stats_html(graph: CitationGraph, node_ids: set[str], edges: list[dict[str, Any]]) -> str:
    shown_roles = Counter(graph.nodes[node_id].get("_role_label") for node_id in node_ids)
    shown_categories = Counter(graph.nodes[node_id].get("_category") or "unknown" for node_id in node_ids)
    shown_years = [safe_int(graph.nodes[node_id].get("year")) for node_id in node_ids]
    shown_years = [year for year in shown_years if year is not None]
    metadata = graph.metadata or {}
    skipped = metadata.get("skipped") or {}
    role_text = ", ".join(
        f"{ROLE_STYLES.get(role or '', {}).get('label', role or 'unknown')}: {count:,}"
        for role, count in shown_roles.most_common()
    )
    top_category, top_category_count = shown_categories.most_common(1)[0] if shown_categories else ("none", 0)
    edge_density = 0.0 if len(node_ids) <= 1 else len(edges) / (len(node_ids) * (len(node_ids) - 1)) * 100
    shown_span = f"{min(shown_years)}-{max(shown_years)}" if shown_years else "unknown"

    if node_ids:
        top_node_id = max(node_ids, key=lambda item: (safe_int(graph.nodes[item].get("_graph_cited_by")) or 0, graph.nodes[item].get("_score", 0)))
        top_node = graph.nodes[top_node_id]
        top_paper = html.escape(short_title(top_node, 74))
        top_paper_metric = safe_int(top_node.get("_graph_cited_by")) or 0
    else:
        top_paper = "None"
        top_paper_metric = 0

    return f"""
<div class="metric-grid">
  <div class="metric-card"><div class="metric-label">Shown Papers</div><div class="metric-value">{len(node_ids):,}</div></div>
  <div class="metric-card"><div class="metric-label">Shown Links</div><div class="metric-value">{len(edges):,}</div></div>
  <div class="metric-card"><div class="metric-label">Year Window</div><div class="metric-value">{shown_span}</div></div>
  <div class="metric-card"><div class="metric-label">Density</div><div class="metric-value">{format_percent(edge_density)}</div></div>
</div>
<div class="summary-block">
  <b>{html.escape(graph.name.replace("_", "."))}</b><br>
  Source graph: {len(graph.nodes):,} papers and {len(graph.edges):,} citation links.<br>
  Seed papers: {format_count(metadata.get("seed_paper_count"))}. Dataset span: {graph.year_min}-{graph.year_max}.
</div>
<div class="summary-block">
  <b>Shown selection</b><br>
  Top category: {html.escape(str(top_category))} ({top_category_count:,}).<br>
  Role mix: {html.escape(role_text or "none")}.<br>
  Most cited inside graph: {top_paper} ({top_paper_metric:,} citing papers).
</div>
<div class="summary-block">
  <b>Build diagnostics</b><br>
  Nontemporal edges skipped: {format_count(skipped.get("nontemporal_edges"))}.<br>
  Duplicate edges skipped: {format_count(skipped.get("duplicate_edges"))}. Self edges skipped: {format_count(skipped.get("self_edges"))}.
</div>
"""


def make_simple_trends_figure(graph: CitationGraph, node_ids: set[str]) -> Any:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if not node_ids:
        return make_empty_figure("Selection trends", "No papers match the current filters.", height=560)

    nodes = [graph.nodes[node_id] for node_id in node_ids]
    year_counts = Counter(safe_int(node.get("year")) for node in nodes if safe_int(node.get("year")) is not None)
    category_items = list(reversed(Counter(node.get("_category") or "unknown" for node in nodes).most_common(8)))
    cited_values = [safe_int(node.get("_graph_cited_by")) or 0 for node in nodes]
    degree_labels, degree_counts = binned_counts(cited_values)
    top_cited = list(
        reversed(
            sorted(
                node_ids,
                key=lambda node_id: (
                    safe_int(graph.nodes[node_id].get("_graph_cited_by")) or 0,
                    safe_int(graph.nodes[node_id].get("citation_count")) or 0,
                    graph.nodes[node_id].get("_score", 0),
                ),
                reverse=True,
            )[:12]
        )
    )

    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Publication years",
            "Primary categories",
            "In-graph cited-by distribution",
            "Top cited papers",
        ),
        horizontal_spacing=0.25,
        vertical_spacing=0.34,
    )

    years = sorted(year_counts)
    figure.add_trace(
        go.Bar(
            x=years,
            y=[year_counts[year] for year in years],
            marker_color="#2563eb",
            hovertemplate="Year %{x}<br>Papers %{y}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Bar(
            x=[count for _, count in category_items],
            y=[compact_label(category, 30) for category, _ in category_items],
            orientation="h",
            marker_color="#0f766e",
            hovertemplate="Category %{y}<br>Papers %{x}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    figure.add_trace(
        go.Bar(
            x=degree_labels,
            y=degree_counts,
            marker_color="#f97316",
            hovertemplate="In-graph cited by %{x}<br>Papers %{y}<extra></extra>",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    figure.add_trace(
        go.Bar(
            x=[safe_int(graph.nodes[node_id].get("_graph_cited_by")) or 0 for node_id in top_cited],
            y=[short_title(graph.nodes[node_id], 34) for node_id in top_cited],
            orientation="h",
            marker_color=[role_color(graph.nodes[node_id].get("_role_label")) for node_id in top_cited],
            hovertemplate="%{y}<br>In-graph cited by %{x}<extra></extra>",
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    figure.update_layout(
        title={"text": f"Selection trends<br><sup>{len(node_ids):,} papers shown</sup>", "x": 0.01},
    )
    figure.update_xaxes(title_text="Year", row=1, col=1)
    figure.update_yaxes(title_text="Papers", row=1, col=1)
    figure.update_xaxes(title_text="Papers", row=1, col=2)
    figure.update_xaxes(title_text="Cited by papers in graph", row=2, col=2)
    figure.update_yaxes(automargin=True, tickfont={"size": 10})
    figure = style_companion_figure(figure, height=760)
    figure.update_layout(margin={"l": 92, "r": 34, "t": 94, "b": 76})
    return figure


def simple_node_hover(node: dict[str, Any]) -> str:
    return (
        f"<b>{html.escape(short_title(node, 90))}</b><br>"
        f"Year: {node.get('year') or 'unknown'} | Role: {html.escape(str(node.get('_role_label') or 'paper'))}<br>"
        f"Category: {html.escape(str(node.get('_category') or 'unknown'))}<br>"
        f"In-graph cited by: {node.get('_graph_cited_by') or 0} | "
        f"S2 citations: {format_count(node.get('citation_count'))}"
    )


def make_simple_graph_figure(graph: CitationGraph, node_ids: set[str]) -> Any:
    import plotly.graph_objects as go

    if not node_ids:
        return make_empty_figure("Citation map", "No papers match the current filters.", height=520)

    display_ids = set(ranked_display_node_ids(graph, node_ids, limit=SIMPLE_GRAPH_NODE_LIMIT))
    positions, categories = layout_positions(graph, display_ids)
    edges = [
        edge
        for edge in filtered_edges(graph, display_ids)
        if edge.get("source") in positions and edge.get("target") in positions
    ]
    edges = sorted(edges, key=lambda edge: edge_importance(graph, edge), reverse=True)[:SIMPLE_GRAPH_EDGE_LIMIT]

    figure = go.Figure()
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for edge in edges:
        source = edge["source"]
        target = edge["target"]
        x0, y0 = positions[source]
        x1, y1 = positions[target]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    if edge_x:
        figure.add_trace(
            go.Scattergl(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line={"width": 0.75, "color": "rgba(100,116,139,0.32)"},
                hoverinfo="skip",
                showlegend=False,
            )
        )

    for role, style in ROLE_STYLES.items():
        role_ids = [node_id for node_id in display_ids if graph.nodes[node_id].get("_role_label") == role]
        if not role_ids:
            continue
        figure.add_trace(
            go.Scattergl(
                x=[positions[node_id][0] for node_id in role_ids],
                y=[positions[node_id][1] for node_id in role_ids],
                mode="markers",
                marker={
                    "size": [min(22, 7 + math.sqrt(safe_int(graph.nodes[node_id].get("_graph_cited_by")) or 0) * 2.5) for node_id in role_ids],
                    "color": style["color"],
                    "symbol": style["symbol"],
                    "line": {"width": 0.8, "color": "#ffffff"},
                    "opacity": 0.86,
                },
                hovertemplate="%{customdata}<extra></extra>",
                customdata=[simple_node_hover(graph.nodes[node_id]) for node_id in role_ids],
                name=style["label"],
            )
        )

    if positions:
        shown_years = [position[0] for position in positions.values()]
        x_min = min(shown_years)
        x_max = max(shown_years)
        x_pad = max(1, (x_max - x_min) * 0.04)
    else:
        x_min = graph.year_min
        x_max = graph.year_max
        x_pad = 1

    figure.update_layout(
        title={
            "text": (
                f"Citation map<br><sup>{len(display_ids):,} papers and {len(edges):,} shown links. "
                "Edges point from referenced paper to citing paper.</sup>"
            ),
            "x": 0.01,
        },
        height=560,
        margin={"l": 96, "r": 28, "t": 92, "b": 64},
        font={"family": PLOT_FONT, "color": TEXT_COLOR},
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        hovermode="closest",
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1, "font": {"size": 11}},
        xaxis={
            "title": "Publication year",
            "gridcolor": GRID_COLOR,
            "linecolor": "#cbd5e1",
            "showline": True,
            "zeroline": False,
            "range": [x_min - x_pad, x_max + x_pad],
        },
        yaxis={
            "title": "Category lane",
            "tickmode": "array",
            "tickvals": list(range(len(categories))),
            "ticktext": [compact_label(category, 28) for category in categories],
            "tickfont": {"size": 10},
            "gridcolor": "rgba(226,232,240,0.72)",
            "linecolor": "#cbd5e1",
            "range": [-0.65, len(categories) - 0.35],
            "showline": True,
            "automargin": True,
            "zeroline": False,
        },
    )
    return figure


def simple_stats_html(graph: CitationGraph, node_ids: set[str]) -> str:
    shown_categories = Counter(graph.nodes[node_id].get("_category") or "unknown" for node_id in node_ids)
    shown_roles = Counter(graph.nodes[node_id].get("_role_label") or "paper" for node_id in node_ids)
    years = [safe_int(graph.nodes[node_id].get("year")) for node_id in node_ids]
    years = [year for year in years if year is not None]
    year_window = f"{min(years)}-{max(years)}" if years else "unknown"
    top_category, top_category_count = shown_categories.most_common(1)[0] if shown_categories else ("none", 0)
    role_text = ", ".join(
        f"{ROLE_STYLES.get(role, {}).get('label', role)}: {count:,}"
        for role, count in shown_roles.most_common()
    )

    if node_ids:
        top_node_id = max(
            node_ids,
            key=lambda node_id: (
                safe_int(graph.nodes[node_id].get("_graph_cited_by")) or 0,
                safe_int(graph.nodes[node_id].get("citation_count")) or 0,
                graph.nodes[node_id].get("_score", 0),
            ),
        )
        top_node = graph.nodes[top_node_id]
        top_paper = html.escape(short_title(top_node, 80))
        top_cited_by = safe_int(top_node.get("_graph_cited_by")) or 0
    else:
        top_paper = "None"
        top_cited_by = 0

    return f"""
<div class="simple-stats">
  <div class="simple-stat"><div class="simple-stat-label">Shown Papers</div><div class="simple-stat-value">{len(node_ids):,}</div></div>
  <div class="simple-stat"><div class="simple-stat-label">Source Papers</div><div class="simple-stat-value">{len(graph.nodes):,}</div></div>
  <div class="simple-stat"><div class="simple-stat-label">Year Window</div><div class="simple-stat-value">{year_window}</div></div>
  <div class="simple-stat"><div class="simple-stat-label">Top Category</div><div class="simple-stat-value">{html.escape(str(top_category))}</div></div>
</div>
<div class="simple-note">
  <b>{html.escape(graph.name.replace("_", "."))}</b> has {len(graph.edges):,} citation links.
  Current selection top category: {html.escape(str(top_category))} ({top_category_count:,} papers).
  Role mix: {html.escape(role_text or "none")}.
  Most cited shown paper: {top_paper} ({top_cited_by:,} in-graph citing papers).
</div>
"""


def render_simple_view(
    data_dir: str,
    subdomain: str,
    mode: str,
    max_nodes: int,
    min_year: float | int | None,
    max_year: float | int | None,
    search_text: str,
) -> tuple[str, Any, Any, list[list[Any]]]:
    graph = load_graph(data_dir, subdomain)
    min_year_int = int(min_year) if min_year not in (None, "") else None
    max_year_int = int(max_year) if max_year not in (None, "") else None
    node_ids = selected_node_ids(
        graph,
        mode=mode,
        max_nodes=min(SIMPLE_MAX_NODES, int(max_nodes or SIMPLE_DEFAULT_MAX_NODES)),
        min_year=min_year_int,
        max_year=max_year_int,
        search_text=search_text or "",
        selected_node=None,
    )
    return (
        simple_stats_html(graph, node_ids),
        make_simple_graph_figure(graph, node_ids),
        make_simple_trends_figure(graph, node_ids),
        table_rows(graph, node_ids, limit=SIMPLE_TABLE_LIMIT),
    )


def render_view(
    data_dir: str,
    subdomain: str,
    mode: str,
    max_nodes: int,
    min_year: float | int | None,
    max_year: float | int | None,
    search_text: str,
    selected_node: str | None,
    show_edges: bool,
    size_by: str,
) -> tuple[Any, Any, Any, Any, list[list[Any]], list[str], str, list[list[Any]], str, str, str]:
    graph = load_graph(data_dir, subdomain)
    min_year_int = int(min_year) if min_year not in (None, "") else None
    max_year_int = int(max_year) if max_year not in (None, "") else None
    node_ids = selected_node_ids(
        graph,
        mode=mode,
        max_nodes=max_nodes,
        min_year=min_year_int,
        max_year=max_year_int,
        search_text=search_text or "",
        selected_node=selected_node,
    )
    edges = filtered_edges(graph, node_ids)
    figure = make_figure(
        graph,
        node_ids,
        show_edges=show_edges,
        size_by=size_by,
        selected_node=selected_node,
        edges=edges,
    )
    return (
        figure,
        make_summary_figure(graph, node_ids, edges),
        make_temporal_flow_figure(graph, edges),
        make_influence_figure(graph, node_ids),
        table_rows(graph, node_ids),
        table_node_ids(graph, node_ids),
        paper_details(graph, selected_node),
        neighbor_rows(graph, selected_node),
        stats_html(graph, node_ids, edges),
        reader_path_html(graph, node_ids, selected_node),
        build_diagnostics_html(graph),
    )


def initial_values(data_dir: Path) -> tuple[str, int, int, str | None]:
    subdomains = discover_subdomains(data_dir)
    if not subdomains:
        raise FileNotFoundError(f"No graph subdomains found under {data_dir}")
    graph = load_graph(data_dir, subdomains[0])
    choices = paper_choices(graph)
    selected = choices[0][1] if choices else None
    return subdomains[0], graph.default_year_min, graph.default_year_max, selected


def make_light_theme(gr: Any) -> Any:
    return gr.themes.Base(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.sky,
        neutral_hue=gr.themes.colors.slate,
        text_size=gr.themes.sizes.text_sm,
        spacing_size=gr.themes.sizes.spacing_sm,
        radius_size=gr.themes.sizes.radius_sm,
    ).set(
        body_background_fill="#f6f8fb",
        body_background_fill_dark="#f6f8fb",
        body_text_color="#111827",
        body_text_color_dark="#111827",
        body_text_color_subdued="#475569",
        body_text_color_subdued_dark="#475569",
        background_fill_primary="#ffffff",
        background_fill_primary_dark="#ffffff",
        background_fill_secondary="#f8fafc",
        background_fill_secondary_dark="#f8fafc",
        block_background_fill="#ffffff",
        block_background_fill_dark="#ffffff",
        block_border_color="#dbe3ef",
        block_border_color_dark="#dbe3ef",
        block_label_background_fill="#ffffff",
        block_label_background_fill_dark="#ffffff",
        block_label_text_color="#475569",
        block_label_text_color_dark="#475569",
        block_shadow="0 1px 2px rgba(15, 23, 42, 0.04)",
        block_shadow_dark="0 1px 2px rgba(15, 23, 42, 0.04)",
        panel_background_fill="#ffffff",
        panel_background_fill_dark="#ffffff",
        panel_border_color="#dbe3ef",
        panel_border_color_dark="#dbe3ef",
        input_background_fill="#ffffff",
        input_background_fill_dark="#ffffff",
        input_background_fill_focus="#ffffff",
        input_background_fill_focus_dark="#ffffff",
        input_border_color="#cbd5e1",
        input_border_color_dark="#cbd5e1",
        input_border_color_focus="#2563eb",
        input_border_color_focus_dark="#2563eb",
        input_placeholder_color="#94a3b8",
        input_placeholder_color_dark="#94a3b8",
        table_text_color="#111827",
        table_text_color_dark="#111827",
        table_border_color="#e2e8f0",
        table_border_color_dark="#e2e8f0",
        table_even_background_fill="#ffffff",
        table_even_background_fill_dark="#ffffff",
        table_odd_background_fill="#f8fafc",
        table_odd_background_fill_dark="#f8fafc",
        table_row_focus="#eff6ff",
        table_row_focus_dark="#eff6ff",
        button_primary_background_fill="#2563eb",
        button_primary_background_fill_dark="#2563eb",
        button_primary_background_fill_hover="#1d4ed8",
        button_primary_background_fill_hover_dark="#1d4ed8",
        button_primary_border_color="#2563eb",
        button_primary_border_color_dark="#2563eb",
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark="#ffffff",
        button_secondary_background_fill="#ffffff",
        button_secondary_background_fill_dark="#ffffff",
        button_secondary_background_fill_hover="#f8fafc",
        button_secondary_background_fill_hover_dark="#f8fafc",
        button_secondary_border_color="#cbd5e1",
        button_secondary_border_color_dark="#cbd5e1",
        button_secondary_text_color="#334155",
        button_secondary_text_color_dark="#334155",
    )


def install_launch_defaults(demo: Any, theme: Any, css: str = APP_CSS) -> None:
    import inspect

    launch_parameters = inspect.signature(demo.launch).parameters
    if "css" not in launch_parameters and "theme" not in launch_parameters:
        return

    original_launch = demo.launch

    def launch_with_light_defaults(*args: Any, **kwargs: Any) -> Any:
        if "css" in launch_parameters:
            kwargs.setdefault("css", css)
        if "theme" in launch_parameters:
            kwargs.setdefault("theme", theme)
        return original_launch(*args, **kwargs)

    demo.launch = launch_with_light_defaults


def build_app(data_dir: str | Path | None = DEFAULT_DATA_DIR) -> Any:
    import inspect

    os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib"))

    import gradio as gr

    if data_dir is None:
        data_dir = resolve_default_data_dir()
    if data_dir is None:
        raise FileNotFoundError(
            "No default citation graph data directory found. Pass --data-dir or set ARXIV_GRAPH_DATA_DIR."
        )

    data_dir = Path(data_dir).resolve()
    subdomains = discover_subdomains(data_dir)
    if not subdomains:
        raise FileNotFoundError(f"No graph subdomains found under {data_dir}")

    initial_subdomain, initial_min_year, initial_max_year, _ = initial_values(data_dir)
    initial_graph = load_graph(data_dir, initial_subdomain)
    theme = make_light_theme(gr)

    block_parameters = inspect.signature(gr.Blocks).parameters
    blocks_kwargs: dict[str, Any] = {"title": "arXiv Citation Trends"}
    css_at_launch = "css" not in block_parameters
    theme_at_launch = "theme" not in block_parameters
    if not css_at_launch:
        blocks_kwargs["css"] = SIMPLE_APP_CSS
    if not theme_at_launch:
        blocks_kwargs["theme"] = theme

    with gr.Blocks(**blocks_kwargs) as demo:
        data_dir_state = gr.State(str(data_dir))

        gr.HTML(
            f"""
            <div class="simple-header">
              <h1>{html.escape(initial_graph.name.replace("_", "."))} citation trends</h1>
              <p>{len(initial_graph.nodes):,} papers · {len(initial_graph.edges):,} citation links</p>
            </div>
            """
        )

        with gr.Row():
            subdomain_input = gr.Dropdown(label="Graph", choices=subdomains, value=initial_subdomain, scale=2)
            mode_input = gr.Radio(
                label="Selection",
                choices=[
                    "Balanced overview",
                    "Seed papers and references",
                    "Most cited within graph",
                    "Most connected papers",
                ],
                value="Balanced overview",
                scale=3,
            )

        with gr.Row():
            max_nodes_input = gr.Slider(
                label="Max papers in trend sample",
                minimum=50,
                maximum=SIMPLE_MAX_NODES,
                step=25,
                value=SIMPLE_DEFAULT_MAX_NODES,
                scale=2,
            )
            min_year_input = gr.Number(label="Min year", value=initial_min_year, precision=0)
            max_year_input = gr.Number(label="Max year", value=initial_max_year, precision=0)

        with gr.Row():
            search_input = gr.Textbox(label="Search title, author, venue, abstract, or arXiv ID", value="", scale=4)
            update_button = gr.Button("Refresh", variant="primary", scale=1)

        stats_output = gr.HTML()
        graph_output = gr.Plot(label="Citation map", show_label=False)
        trends_output = gr.Plot(label="Trends", show_label=False)
        table_output = gr.Dataframe(
            label="Top papers in selection",
            headers=[
                "Title",
                "Year",
                "Role",
                "Category",
                "In-graph cited by",
                "In-graph refs",
                "S2 citations",
                "Influential cites",
                "arXiv",
                "Venue",
            ],
            datatype=["str", "number", "str", "str", "number", "number", "number", "number", "str", "str"],
            interactive=False,
            wrap=True,
            max_height=560,
        )

        controls = [
            data_dir_state,
            subdomain_input,
            mode_input,
            max_nodes_input,
            min_year_input,
            max_year_input,
            search_input,
        ]
        outputs = [stats_output, graph_output, trends_output, table_output]

        def update_subdomain(
            data_dir_value: str,
            subdomain_value: str,
            mode_value: str,
            max_nodes_value: int,
            search_value: str,
        ) -> tuple[int, int, str, Any, Any, list[list[Any]]]:
            graph = load_graph(data_dir_value, subdomain_value)
            stats, graph_figure, trends_figure, rows = render_simple_view(
                data_dir_value,
                subdomain_value,
                mode_value,
                max_nodes_value,
                graph.default_year_min,
                graph.default_year_max,
                search_value,
            )
            return (
                graph.default_year_min,
                graph.default_year_max,
                stats,
                graph_figure,
                trends_figure,
                rows,
            )

        subdomain_input.change(
            update_subdomain,
            inputs=[
                data_dir_state,
                subdomain_input,
                mode_input,
                max_nodes_input,
                search_input,
            ],
            outputs=[
                min_year_input,
                max_year_input,
                stats_output,
                graph_output,
                trends_output,
                table_output,
            ],
        )
        update_button.click(render_simple_view, inputs=controls, outputs=outputs, show_progress="minimal")
        search_input.submit(render_simple_view, inputs=controls, outputs=outputs, show_progress="minimal")
        demo.load(render_simple_view, inputs=controls, outputs=outputs)

    if css_at_launch or theme_at_launch:
        install_launch_defaults(demo, theme, SIMPLE_APP_CSS)

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the arXiv citation graph Gradio viewer.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Directory containing subdomain graph folders.")
    parser.add_argument("--server-name", default="0.0.0.0", help="Gradio server host.")
    parser.add_argument("--server-port", type=int, default=7860, help="Gradio server port.")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    demo = build_app(args.data_dir)
    demo.launch(server_name=args.server_name, server_port=args.server_port, share=args.share)


if __name__ == "__main__":
    main()
