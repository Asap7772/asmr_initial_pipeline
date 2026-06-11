from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from arxiv_paper import citation_graph_viewer as viewer


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def write_graph(path: Path, name: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    write_jsonl(
        path / "nodes.jsonl",
        [
            {
                "node_id": f"{name}:a",
                "title": f"{name} A",
                "year": 2021,
                "roles": ["seed"],
            },
            {
                "node_id": f"{name}:b",
                "title": f"{name} B",
                "year": 2022,
                "roles": ["reference"],
            },
        ],
    )
    write_jsonl(path / "edges.jsonl", [{"source": f"{name}:b", "target": f"{name}:a"}])
    (path / "manifest.json").write_text(json.dumps({"seed_paper_count": 1}), encoding="utf-8")


def write_many_node_graph(path: Path, name: str, *, count: int = 80, role: str | None = None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    nodes = []
    for index in range(count):
        node: dict[str, object] = {
            "node_id": f"{name}:{index}",
            "title": f"common paper {index}",
            "year": 2000 + index % 10,
            "citation_count": count - index,
        }
        if role:
            node["roles"] = [role]
        nodes.append(node)
    write_jsonl(path / "nodes.jsonl", nodes)
    write_jsonl(path / "edges.jsonl", [])
    (path / "manifest.json").write_text(json.dumps({"seed_paper_count": count if role == "seed" else 0}), encoding="utf-8")


class CitationGraphViewerDiscoveryTest(unittest.TestCase):
    def test_resolve_default_data_dir_checks_default_graph_names(self) -> None:
        with TemporaryDirectory() as temp_dir:
            old_repo_root = viewer.REPO_ROOT
            try:
                root = Path(temp_dir)
                viewer.REPO_ROOT = root
                expected = root / "data" / viewer.DEFAULT_DATA_DIRNAMES[0]
                write_graph(expected / "tcs_core", "tcs")

                self.assertEqual(viewer.resolve_default_data_dir(), expected)
            finally:
                viewer.REPO_ROOT = old_repo_root

    def test_discover_subdomains_recurses_into_nested_graph_dirs(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_graph(root / "flat", "flat")
            write_graph(root / "topic" / "nested", "nested")

            self.assertEqual(viewer.discover_subdomains(root), ["flat", "topic/nested"])

            graph = viewer.load_graph(root, "topic/nested")
            self.assertEqual(graph.name, "topic/nested")
            self.assertEqual(set(graph.nodes), {"nested:a", "nested:b"})

    def test_data_dir_can_be_graph_dir_itself(self) -> None:
        with TemporaryDirectory() as temp_dir:
            graph_dir = Path(temp_dir) / "tcs_core"
            write_graph(graph_dir, "tcs")

            self.assertEqual(viewer.discover_subdomains(graph_dir), ["tcs_core"])

            graph = viewer.load_graph(graph_dir, "tcs_core")
            self.assertEqual(graph.name, "tcs_core")
            self.assertEqual(len(graph.edges), 1)

    def test_selection_cap_limits_many_seed_papers(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_many_node_graph(root / "many", "many", role="seed")
            graph = viewer.load_graph(root, "many")
            selected = "many:79"

            node_ids = viewer.selected_node_ids(
                graph,
                "Balanced overview",
                max_nodes=50,
                min_year=None,
                max_year=None,
                search_text="",
                selected_node=selected,
            )

            self.assertLessEqual(len(node_ids), 50)
            self.assertIn(selected, node_ids)

    def test_selection_cap_limits_broad_search_matches(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_many_node_graph(root / "many", "many")
            graph = viewer.load_graph(root, "many")
            selected = "many:79"

            node_ids = viewer.selected_node_ids(
                graph,
                "Balanced overview",
                max_nodes=50,
                min_year=None,
                max_year=None,
                search_text="common",
                selected_node=selected,
            )

            self.assertLessEqual(len(node_ids), 50)
            self.assertIn(selected, node_ids)


if __name__ == "__main__":
    unittest.main()
