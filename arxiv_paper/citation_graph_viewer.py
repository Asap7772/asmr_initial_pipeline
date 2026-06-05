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


DEFAULT_DATA_DIR = (
    Path(os.environ["ARXIV_GRAPH_DATA_DIR"])
    if os.environ.get("ARXIV_GRAPH_DATA_DIR")
    else Path(__file__).resolve().parents[1]
    / "data"
    / "arxiv_citation_graph_balanced_1000"
)

DEFAULT_MAX_NODES = 350
EDGE_CAP = 5000

ROLE_STYLES = {
    "seed + reference": {"color": "#059669", "symbol": "diamond"},
    "seed paper": {"color": "#2563eb", "symbol": "circle"},
    "reference": {"color": "#64748b", "symbol": "circle-open"},
    "paper": {"color": "#d97706", "symbol": "circle"},
}


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
    if not data_dir.exists():
        return []
    return sorted(path.name for path in data_dir.iterdir() if (path / "nodes.jsonl").exists())


def load_graph(data_dir: Path | str, subdomain: str) -> CitationGraph:
    data_dir = Path(data_dir)
    cache_key = (str(data_dir.resolve()), subdomain)
    if cache_key in GRAPH_CACHE:
        return GRAPH_CACHE[cache_key]

    subdomain_dir = data_dir / subdomain
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
        node["_category"] = paper_category(node, fallback=subdomain.replace("_", "."))
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
        name=subdomain,
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
    protected: set[str] = set()

    query_tokens = [token for token in re.split(r"\s+", (search_text or "").strip().lower()) if token]
    if query_tokens:
        matches = {
            node_id
            for node_id, node in graph.nodes.items()
            if node_id in eligible and all(token in node.get("_search_blob", "") for token in query_tokens)
        }
        protected |= matches
        ids = expand_neighbors(graph, matches, depth=1) & eligible
        return cap_nodes(graph, ids, max_nodes=max_nodes, protected=protected)

    if mode == "Selected paper neighborhood" and selected_node in graph.nodes:
        protected.add(selected_node)
        ids = expand_neighbors(graph, {selected_node}, depth=2) & eligible
        return cap_nodes(graph, ids, max_nodes=max_nodes, protected=protected)

    seed_ids = {
        node_id for node_id, node in graph.nodes.items() if node_id in eligible and role_label(node).startswith("seed")
    }

    if mode == "Seed papers and references":
        ids = set(seed_ids)
        for seed_id in seed_ids:
            ids.update(graph.in_neighbors.get(seed_id, set()))
            ids.update(graph.out_neighbors.get(seed_id, set()))
        protected |= seed_ids
        return cap_nodes(graph, ids & eligible, max_nodes=max_nodes, protected=protected)

    if mode == "Most cited within graph":
        ids = top_nodes(graph, eligible, key=lambda node_id: graph.out_degree[node_id], limit=max_nodes)
        return set(ids)

    if mode == "Most connected papers":
        ids = top_nodes(
            graph,
            eligible,
            key=lambda node_id: graph.in_degree[node_id] + graph.out_degree[node_id],
            limit=max_nodes,
        )
        return set(ids)

    ids = set(seed_ids)
    ids.update(
        top_nodes(
            graph,
            eligible - ids,
            key=lambda node_id: graph.nodes[node_id].get("_score", 0),
            limit=max_nodes - len(ids),
        )
    )
    protected |= seed_ids
    return cap_nodes(graph, ids, max_nodes=max_nodes, protected=protected)


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


def cap_nodes(graph: CitationGraph, ids: set[str], max_nodes: int, protected: set[str]) -> set[str]:
    if len(ids) <= max_nodes:
        return ids
    protected = {node_id for node_id in protected if node_id in ids}
    remaining = ids - protected
    keep_count = max(0, max_nodes - len(protected))
    kept = set(
        sorted(
            remaining,
            key=lambda node_id: graph.nodes[node_id].get("_score", 0),
            reverse=True,
        )[:keep_count]
    )
    return protected | kept


def layout_positions(graph: CitationGraph, node_ids: set[str]) -> tuple[dict[str, tuple[float, float]], list[str]]:
    categories = [
        category
        for category, _ in graph.category_counts.most_common()
        if any(graph.nodes[node_id].get("_category") == category for node_id in node_ids)
    ]
    if not categories:
        categories = ["unknown"]
    category_index = {category: index for index, category in enumerate(categories)}

    positions: dict[str, tuple[float, float]] = {}
    for node_id in node_ids:
        node = graph.nodes[node_id]
        year = safe_int(node.get("year")) or graph.year_min
        category = node.get("_category") or categories[0]
        y_base = category_index.get(category, len(categories))
        y = y_base + stable_jitter(node_id, scale=0.72)
        positions[node_id] = (year, y)
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


def make_figure(
    graph: CitationGraph,
    node_ids: set[str],
    show_edges: bool,
    size_by: str,
) -> Any:
    import plotly.graph_objects as go

    positions, categories = layout_positions(graph, node_ids)
    edges = filtered_edges(graph, node_ids) if show_edges else []

    figure = go.Figure()
    if edges:
        edge_x: list[float | None] = []
        edge_y: list[float | None] = []
        for edge in edges:
            source = edge["source"]
            target = edge["target"]
            x0, y0 = positions[source]
            x1, y1 = positions[target]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        figure.add_trace(
            go.Scattergl(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line={"width": 0.7, "color": "rgba(71,85,105,0.20)"},
                hoverinfo="skip",
                name="citation edge",
                showlegend=False,
            )
        )

    label_ids = set(
        sorted(
            node_ids,
            key=lambda node_id: graph.nodes[node_id].get("_score", 0),
            reverse=True,
        )[:18]
    )

    for role, style in ROLE_STYLES.items():
        role_ids = [node_id for node_id in node_ids if graph.nodes[node_id].get("_role_label") == role]
        if not role_ids:
            continue
        figure.add_trace(
            go.Scattergl(
                x=[positions[node_id][0] for node_id in role_ids],
                y=[positions[node_id][1] for node_id in role_ids],
                mode="markers+text",
                marker={
                    "size": [marker_size(graph.nodes[node_id], size_by) for node_id in role_ids],
                    "color": style["color"],
                    "symbol": style["symbol"],
                    "line": {"width": 1, "color": "#0f172a"},
                    "opacity": 0.84,
                },
                text=[short_title(graph.nodes[node_id]) if node_id in label_ids else "" for node_id in role_ids],
                textposition="top center",
                textfont={"size": 10, "color": "#111827"},
                hovertemplate="%{customdata}<extra></extra>",
                customdata=[hover_html(graph.nodes[node_id]) for node_id in role_ids],
                name=role,
            )
        )

    title = f"{graph.name.replace('_', '.')} citation graph"
    subtitle = (
        f"{len(node_ids):,} shown nodes, {len(edges):,} shown edges. "
        "Edges run cited/reference paper -> later citing paper."
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
        height=760,
        margin={"l": 42, "r": 24, "t": 72, "b": 44},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8fafc",
        hovermode="closest",
        xaxis={
            "title": "Publication year",
            "gridcolor": "#e2e8f0",
            "zeroline": False,
            "range": [x_min - x_pad, x_max + x_pad],
        },
        yaxis={
            "title": "Primary category lane",
            "tickmode": "array",
            "tickvals": list(range(len(categories))),
            "ticktext": categories,
            "gridcolor": "#e2e8f0",
            "zeroline": False,
        },
    )
    return figure


def marker_size(node: dict[str, Any], size_by: str) -> float:
    if size_by == "Semantic Scholar citations":
        value = safe_int(node.get("citation_count")) or 0
        return min(28, 7 + math.sqrt(value) * 0.26)
    if size_by == "In-graph cited by count":
        return min(30, 7 + (safe_int(node.get("_graph_cited_by")) or 0) * 1.8)
    value = (safe_int(node.get("_graph_refs")) or 0) + (safe_int(node.get("_graph_cited_by")) or 0)
    return min(30, 7 + value * 1.15)


def hover_html(node: dict[str, Any]) -> str:
    title = html.escape(str(node.get("title") or "Untitled"))
    authors = html.escape(node.get("_authors_text") or "Unknown authors")
    venue = html.escape(str(node.get("venue") or ""))
    arxiv_id = html.escape(str(node.get("arxiv_id") or ""))
    return (
        f"<b>{title}</b><br>"
        f"{authors}<br>"
        f"Year: {node.get('year') or 'unknown'} | Role: {node.get('_role_label')}<br>"
        f"arXiv: {arxiv_id or 'n/a'} | Venue: {venue or 'n/a'}<br>"
        f"S2 citations: {node.get('citation_count') or 0:,} | "
        f"In-graph cited by: {node.get('_graph_cited_by') or 0} | "
        f"In-graph refs: {node.get('_graph_refs') or 0}"
    )


def short_title(node: dict[str, Any], max_len: int = 42) -> str:
    title = str(node.get("title") or "Untitled")
    return title if len(title) <= max_len else title[: max_len - 1] + "..."


def table_rows(graph: CitationGraph, node_ids: set[str], limit: int = 80) -> list[list[Any]]:
    rows = []
    for node_id in sorted(node_ids, key=lambda item: graph.nodes[item].get("_score", 0), reverse=True)[:limit]:
        node = graph.nodes[node_id]
        rows.append(
            [
                node.get("title") or "Untitled",
                node.get("year") or "",
                node.get("_role_label"),
                node.get("_graph_cited_by") or 0,
                node.get("_graph_refs") or 0,
                node.get("citation_count") or 0,
                node.get("arxiv_id") or "",
                node.get("venue") or "",
            ]
        )
    return rows


def paper_choices(graph: CitationGraph, limit: int = 1200) -> list[tuple[str, str]]:
    ranked_ids = sorted(graph.nodes, key=lambda node_id: graph.nodes[node_id].get("_score", 0), reverse=True)
    choices = []
    for node_id in ranked_ids[:limit]:
        node = graph.nodes[node_id]
        label = f"[{node.get('year') or '?'}] {short_title(node, 86)}"
        arxiv_id = node.get("arxiv_id")
        if arxiv_id:
            label += f" ({arxiv_id})"
        choices.append((label, node_id))
    return choices


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
| Role | {node.get("_role_label")} |
| Primary category | {node.get("_category") or ""} |
| Semantic Scholar citations | {node.get("citation_count") or 0:,} |
| In-graph papers citing this | {node.get("_graph_cited_by") or 0} |
| In-graph references cited by this | {node.get("_graph_refs") or 0} |

**Papers this cites inside the graph**

{references}

**Papers citing this inside the graph**

{citing}

**Abstract**

{escape_markdown(abstract) if abstract else "No abstract available."}
"""


def ranked_neighbor_list(graph: CitationGraph, node_ids: set[str], limit: int) -> str:
    if not node_ids:
        return "None in the displayed graph data."
    lines = []
    for node_id in sorted(node_ids, key=lambda item: graph.nodes[item].get("_score", 0), reverse=True)[:limit]:
        node = graph.nodes[node_id]
        year = node.get("year") or "?"
        lines.append(f"- [{year}] {escape_markdown(short_title(node, 96))}")
    return "\n".join(lines)


def escape_markdown(text: Any) -> str:
    return str(text).replace("|", "\\|")


def stats_markdown(graph: CitationGraph, node_ids: set[str], edges: list[dict[str, Any]]) -> str:
    shown_roles = Counter(graph.nodes[node_id].get("_role_label") for node_id in node_ids)
    metadata = graph.metadata or {}
    skipped = metadata.get("skipped") or {}
    role_text = " | ".join(f"{role}: {count:,}" for role, count in shown_roles.items())
    return f"""**Dataset**

| Field | Value |
| --- | ---: |
| Subdomain | {graph.name.replace("_", ".")} |
| Total nodes | {len(graph.nodes):,} |
| Total edges | {len(graph.edges):,} |
| Seed papers | {metadata.get("seed_paper_count", "")} |
| Shown nodes | {len(node_ids):,} |
| Shown edges | {len(edges):,} |
| Year span | {graph.year_min} to {graph.year_max} |
| Nontemporal edges skipped at build time | {skipped.get("nontemporal_edges", 0):,} |

**Shown roles**

{role_text or "None"}
"""


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
) -> tuple[Any, list[list[Any]], str, str]:
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
    edges = filtered_edges(graph, node_ids) if show_edges else []
    figure = make_figure(graph, node_ids, show_edges=show_edges, size_by=size_by)
    return (
        figure,
        table_rows(graph, node_ids),
        paper_details(graph, selected_node),
        stats_markdown(graph, node_ids, edges),
    )


def initial_values(data_dir: Path) -> tuple[str, int, int, str | None]:
    subdomains = discover_subdomains(data_dir)
    if not subdomains:
        raise FileNotFoundError(f"No graph subdomains found under {data_dir}")
    graph = load_graph(data_dir, subdomains[0])
    choices = paper_choices(graph)
    selected = choices[0][1] if choices else None
    return subdomains[0], graph.default_year_min, graph.default_year_max, selected


def build_app(data_dir: str | Path = DEFAULT_DATA_DIR) -> Any:
    import gradio as gr

    data_dir = Path(data_dir).resolve()
    subdomains = discover_subdomains(data_dir)
    if not subdomains:
        raise FileNotFoundError(f"No graph subdomains found under {data_dir}")

    initial_subdomain, initial_min_year, initial_max_year, initial_selected = initial_values(data_dir)
    initial_graph = load_graph(data_dir, initial_subdomain)

    with gr.Blocks(title="arXiv Citation Graph Viewer") as demo:
        data_dir_state = gr.State(str(data_dir))

        gr.Markdown(
            "## arXiv Citation Graph Viewer\n"
            "Edges are oriented from cited/reference papers to later papers that cite them."
        )

        with gr.Row():
            with gr.Column(scale=1, min_width=310):
                subdomain_input = gr.Dropdown(
                    label="Subdomain",
                    choices=subdomains,
                    value=initial_subdomain,
                )
                mode_input = gr.Radio(
                    label="View",
                    choices=[
                        "Balanced overview",
                        "Seed papers and references",
                        "Most cited within graph",
                        "Most connected papers",
                        "Selected paper neighborhood",
                    ],
                    value="Balanced overview",
                )
                max_nodes_input = gr.Slider(
                    label="Max nodes",
                    minimum=50,
                    maximum=1200,
                    step=25,
                    value=DEFAULT_MAX_NODES,
                )
                with gr.Row():
                    min_year_input = gr.Number(label="Min year", value=initial_min_year, precision=0)
                    max_year_input = gr.Number(label="Max year", value=initial_max_year, precision=0)
                search_input = gr.Textbox(label="Search title, author, arXiv ID, venue, abstract", value="")
                selected_input = gr.Dropdown(
                    label="Selected paper",
                    choices=paper_choices(initial_graph),
                    value=initial_selected,
                    filterable=True,
                )
                show_edges_input = gr.Checkbox(label="Show citation edges", value=True)
                size_by_input = gr.Radio(
                    label="Node size",
                    choices=[
                        "Semantic Scholar citations",
                        "In-graph cited by count",
                        "In-graph total degree",
                    ],
                    value="In-graph cited by count",
                )
                update_button = gr.Button("Update", variant="primary")
                stats_output = gr.Markdown()

            with gr.Column(scale=3):
                plot_output = gr.Plot(label="Graph")
                with gr.Row():
                    table_output = gr.Dataframe(
                        label="Top shown papers",
                        headers=[
                            "Title",
                            "Year",
                            "Role",
                            "In-graph cited by",
                            "In-graph refs",
                            "S2 citations",
                            "arXiv",
                            "Venue",
                        ],
                        datatype=["str", "number", "str", "number", "number", "number", "str", "str"],
                        wrap=True,
                    )
                details_output = gr.Markdown(label="Paper details")

        controls = [
            data_dir_state,
            subdomain_input,
            mode_input,
            max_nodes_input,
            min_year_input,
            max_year_input,
            search_input,
            selected_input,
            show_edges_input,
            size_by_input,
        ]
        outputs = [plot_output, table_output, details_output, stats_output]

        def update_subdomain(
            data_dir_value: str,
            subdomain_value: str,
            mode_value: str,
            max_nodes_value: int,
            search_value: str,
            show_edges_value: bool,
            size_by_value: str,
        ) -> tuple[Any, int, int, Any, list[list[Any]], str, str]:
            graph = load_graph(data_dir_value, subdomain_value)
            choices = paper_choices(graph)
            selected = choices[0][1] if choices else None
            figure, rows, details, stats = render_view(
                data_dir_value,
                subdomain_value,
                mode_value,
                max_nodes_value,
                graph.default_year_min,
                graph.default_year_max,
                search_value,
                selected,
                show_edges_value,
                size_by_value,
            )
            return (
                gr.update(choices=choices, value=selected),
                graph.default_year_min,
                graph.default_year_max,
                figure,
                rows,
                details,
                stats,
            )

        subdomain_input.change(
            update_subdomain,
            inputs=[
                data_dir_state,
                subdomain_input,
                mode_input,
                max_nodes_input,
                search_input,
                show_edges_input,
                size_by_input,
            ],
            outputs=[
                selected_input,
                min_year_input,
                max_year_input,
                plot_output,
                table_output,
                details_output,
                stats_output,
            ],
        )
        update_button.click(render_view, inputs=controls, outputs=outputs)
        for control in [
            mode_input,
            max_nodes_input,
            min_year_input,
            max_year_input,
            search_input,
            selected_input,
            show_edges_input,
            size_by_input,
        ]:
            control.change(render_view, inputs=controls, outputs=outputs)

        demo.load(render_view, inputs=controls, outputs=outputs)

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
