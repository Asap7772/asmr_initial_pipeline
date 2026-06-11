from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow.parquet as pq

from arxiv_paper import proc_citation_graph as proc


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def node(
    node_id: str,
    title: str,
    abstract: str,
    year: int,
    *,
    depth: int,
    citations: int = 0,
) -> dict[str, object]:
    return {
        "node_id": node_id,
        "paper_id": node_id,
        "title": title,
        "abstract": abstract,
        "year": year,
        "citation_count": citations,
        "seed_reference_depth": depth,
        "arxiv_id": f"{year}.{len(node_id):05d}",
    }


def write_fixture_graph(root: Path) -> Path:
    topic = root / "graph_algorithms"
    topic.mkdir(parents=True)
    write_jsonl(
        topic / "nodes.jsonl",
        [
            node(
                "p1",
                "Sparse Cut Algorithms",
                "We introduce a sparse cut routine with three phases for graph partitioning.",
                2001,
                depth=3,
                citations=50,
            ),
            node(
                "p2",
                "Flow Rounding Frameworks",
                "We prove that flow rounding can preserve 2 approximation guarantees.",
                2002,
                depth=3,
                citations=40,
            ),
            node(
                "c1",
                "Balanced Separators from Flow Rounding",
                "We present an algorithm that combines two sparse cuts with flow rounding.",
                2005,
                depth=2,
                citations=30,
            ),
            node(
                "c2",
                "Dynamic Balanced Separator Algorithms",
                "We show how separator routines can be maintained under 10 graph updates.",
                2008,
                depth=1,
                citations=20,
            ),
            node(
                "c3",
                "Online Separator Algorithms for Streaming Graphs",
                "We study online graph streams using four dynamic separator routines.",
                2012,
                depth=0,
                citations=10,
            ),
            node(
                "future",
                "Future Parent Should Be Filtered",
                "We should not be selected as a parent because this paper is too new.",
                2020,
                depth=1,
            ),
        ],
    )
    write_jsonl(
        topic / "edges.jsonl",
        [
            {"source": "p1", "target": "c1", "relation": "cited_by"},
            {"source": "p2", "target": "c1", "relation": "cited_by"},
            {"source": "future", "target": "c1", "relation": "cited_by"},
            {"source": "c1", "target": "c2", "relation": "cited_by"},
            {"source": "p2", "target": "c2", "relation": "cited_by"},
            {"source": "c2", "target": "c3", "relation": "cited_by"},
            {"source": "p2", "target": "c3", "relation": "cited_by"},
        ],
    )
    (topic / "manifest.json").write_text(
        json.dumps({"query": "graph algorithms", "seed_paper_count": 1}),
        encoding="utf-8",
    )
    return topic


class ProcCitationGraphTest(unittest.TestCase):
    def test_select_parent_ids_filters_future_parent(self) -> None:
        with TemporaryDirectory() as temp_dir:
            topic = write_fixture_graph(Path(temp_dir))
            graph = proc.load_topic_graph(topic)
            config = proc.ConversionConfig(input_root=Path(temp_dir), output_dir=Path(temp_dir) / "out")

            parent_ids = proc.select_parent_ids("c1", graph, proc.topic_tokens(graph), config)

            self.assertEqual(parent_ids, ["p1", "p2"])
            self.assertNotIn("future", parent_ids)

    def test_run_conversion_writes_verl_sft_parquet(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "input"
            out = Path(temp_dir) / "out"
            write_fixture_graph(root)
            config = proc.ConversionConfig(
                input_root=root,
                output_dir=out,
                valid_fraction=0.0,
                test_fraction=0.0,
                min_trajectory_turns=2,
                max_trajectory_turns=3,
                max_trajectories_per_topic=2,
            )

            manifest = proc.run_conversion(config)

            self.assertGreaterEqual(manifest["num_single_step_examples"], 3)
            self.assertGreaterEqual(manifest["num_trajectory_examples"], 1)
            table = pq.read_table(out / "sft" / "train.parquet")
            rows = table.to_pylist()
            self.assertEqual(table.schema.names, ["messages", "id", "kind", "topic", "child_node_id", "parent_node_ids"])
            self.assertEqual(len(rows), manifest["split_counts"]["train"])

            first = rows[0]
            self.assertIsInstance(first["messages"], list)
            self.assertEqual(first["messages"][0]["role"], "user")
            self.assertEqual(first["messages"][-1]["role"], "assistant")
            self.assertTrue(first["messages"][0]["content"])
            self.assertTrue(first["messages"][-1]["content"])
            self.assertIn("# Research Direction Proposal\n\n## Task", first["messages"][0]["content"])
            self.assertIn("## Proposed Child Paper", first["messages"][-1]["content"])

            examples = [json.loads(line) for line in (out / "examples.jsonl").read_text(encoding="utf-8").splitlines()]
            single = next(row for row in examples if row["kind"] == "single_step")
            child_title = single["child_paper"]["title"]
            self.assertNotIn(child_title, single["prompt"])
            self.assertIn(child_title, single["completion"])
            for paper in [*single["parent_papers"], single["child_paper"]]:
                self.assertFalse(proc.contains_digit(paper["abstract"]))
                self.assertFalse(any(proc.token_has_quantity_word(token) for token in paper["abstract"].split()))
                self.assertEqual(paper["abstract_processing"]["mode"], "deterministic")
                self.assertEqual(paper["abstract_processing"]["source_kind"], "abstract")
                self.assertTrue(paper["abstract_processing"]["source_sha1"])

    def test_split_jsonl_and_sft_parquet_counts_match(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "input"
            out = Path(temp_dir) / "out"
            write_fixture_graph(root)
            config = proc.ConversionConfig(
                input_root=root,
                output_dir=out,
                valid_fraction=0.0,
                test_fraction=0.0,
                max_trajectories_per_topic=0,
            )

            manifest = proc.run_conversion(config)

            jsonl_count = len((out / "train.jsonl").read_text(encoding="utf-8").splitlines())
            parquet_count = pq.read_table(out / "sft" / "train.parquet").num_rows
            self.assertEqual(jsonl_count, parquet_count)
            self.assertEqual(jsonl_count, manifest["sft_counts"]["train"])

    def test_injected_gemini_processor_path(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "input"
            out = Path(temp_dir) / "out"
            write_fixture_graph(root)

            def fake_processor(
                nodes_by_id: dict[str, dict[str, object]],
                config: proc.ConversionConfig,
            ) -> dict[str, dict[str, object]]:
                return {
                    node_id: proc.build_abstract_record(
                        node_id,
                        node_record["abstract"],
                        "A qualitative rewrite that preserves the research idea without exact quantities.",
                        mode="gemini",
                        model=config.abstract_model,
                    )
                    for node_id, node_record in nodes_by_id.items()
                }

            config = proc.ConversionConfig(
                input_root=root,
                output_dir=out,
                valid_fraction=0.0,
                test_fraction=0.0,
                max_trajectories_per_topic=0,
                abstract_processing_mode="gemini",
                abstract_processor=fake_processor,
            )

            manifest = proc.run_conversion(config)
            examples = [json.loads(line) for line in (out / "examples.jsonl").read_text(encoding="utf-8").splitlines()]
            first = examples[0]

            self.assertEqual(manifest["abstract_processing"]["processor"], "injected")
            self.assertIn("# Research Direction Proposal\n\n## Task", first["prompt"])
            self.assertIn("## Parent Papers", first["prompt"])
            self.assertIn("## Processed summary", first["completion"])
            self.assertEqual(first["child_paper"]["abstract_processing"]["mode"], "gemini")
            self.assertFalse(proc.contains_digit(first["child_paper"]["abstract"]))

    def test_full_text_source_is_preferred_when_present(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            html_dir = root / "paper_html"
            html_dir.mkdir()
            html_path = html_dir / "paper.html"
            html_path.write_text(
                "<html><body><h1>Paper</h1><p>The full paper method uses nine phases and proves two qualitative results.</p></body></html>",
                encoding="utf-8",
            )
            paper = {
                "node_id": "paper",
                "title": "Full Paper Method",
                "abstract": "Fallback abstract with 2 details.",
                "arxiv_html_path": "paper_html/paper.html",
                "_graph_dir": str(root),
            }
            config = proc.ConversionConfig(input_root=root, output_dir=root / "out")

            source_text, source_kind, source_path = proc.read_paper_source_text(paper, config)
            records = proc.deterministic_abstract_records({"paper": paper}, config)

            self.assertEqual(source_kind, "full_text")
            self.assertEqual(source_path, str(html_path))
            self.assertIn("full paper method", source_text.lower())
            self.assertIn("full paper method", records["paper"]["processed_abstract"].lower())
            self.assertFalse(proc.contains_digit(records["paper"]["processed_abstract"]))
            self.assertFalse(
                any(proc.token_has_quantity_word(token) for token in records["paper"]["processed_abstract"].split())
            )


if __name__ == "__main__":
    unittest.main()
