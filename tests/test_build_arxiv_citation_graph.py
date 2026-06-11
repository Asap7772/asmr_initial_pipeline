from __future__ import annotations

import argparse
import unittest
from typing import Any

from arxiv_paper import build_arxiv_citation_graph as graph_builder


MISSING = object()


class FakeSemanticScholarClient:
    def __init__(self, papers_by_lookup_id: dict[str, dict[str, Any]]) -> None:
        self.papers_by_lookup_id = papers_by_lookup_id
        self.requests: list[list[str]] = []

    def paper_batch(
        self,
        ids: list[str],
        *,
        request_label: str | None = None,
    ) -> list[dict[str, Any] | None]:
        self.requests.append(list(ids))
        return [self.papers_by_lookup_id.get(paper_id) for paper_id in ids]


class FakeRetrySession:
    def __init__(self, statuses: list[int]) -> None:
        self.statuses = statuses
        self.requests = 0

    def request(self, method: str, url: str, **kwargs: Any) -> graph_builder.requests.Response:
        status = self.statuses[self.requests]
        self.requests += 1
        response = graph_builder.requests.Response()
        response.status_code = status
        response.url = url
        return response


def paper(
    paper_id: str,
    title: str,
    year: int,
    citation_count: int,
    references: object = MISSING,
    arxiv_id: str | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "paperId": paper_id,
        "title": title,
        "year": year,
        "citationCount": citation_count,
        "influentialCitationCount": 0,
        "referenceCount": 0,
        "externalIds": {},
    }
    if arxiv_id:
        record["externalIds"] = {"ArXiv": arxiv_id}
    if references is not MISSING:
        record["references"] = list(references)  # type: ignore[arg-type]
        record["referenceCount"] = len(record["references"])
    return record


def arxiv_entry(arxiv_id: str, year: int = 2021) -> dict[str, Any]:
    return {
        "arxiv_id": arxiv_id,
        "title": f"arXiv {arxiv_id}",
        "summary": None,
        "published": f"{year}-01-01",
        "authors": [],
        "primary_category": "cs.LG",
        "categories": ["cs.LG"],
        "entry_id": f"https://arxiv.org/abs/{arxiv_id}",
    }


def args(**overrides: Any) -> argparse.Namespace:
    values: dict[str, Any] = {
        "target_node_count": 0,
        "citation_depth": 1,
        "max_references_per_paper": 0,
        "max_expansion_papers_per_depth": 0,
        "internal_only": False,
        "arxiv_references_only": False,
        "allow_nontemporal_edges": False,
        "s2_batch_size": 500,
        "s2_sleep": 0.0,
        "s2_concurrency": 1,
        "seed_publication_types": None,
        "require_seed_publication": False,
        "seed_include_keywords": None,
        "seed_exclude_keywords": None,
        "reference_include_keywords": None,
        "reference_exclude_keywords": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def quiet_progress() -> graph_builder.ProgressReporter:
    return graph_builder.ProgressReporter(enabled=False, interval_seconds=0.0)


class BuildArxivCitationGraphTest(unittest.TestCase):
    def test_derive_s2_search_defaults_from_arxiv_category(self) -> None:
        self.assertEqual(
            graph_builder.derive_s2_search_query("cat:cs.LG"),
            "machine learning",
        )
        self.assertEqual(
            graph_builder.derive_s2_fields_of_study("cat:cs.LG"),
            ["Computer Science"],
        )

    def test_s2_paper_to_arxiv_candidate(self) -> None:
        candidate = graph_builder.s2_paper_to_arxiv_candidate(
            {
                "paperId": "S2",
                "corpusId": 123,
                "externalIds": {"ArXiv": "2101.00001v2", "DOI": "10.1/example"},
                "title": " Test Paper\n",
                "abstract": " Abstract\n",
                "publicationDate": "2021-01-02",
                "authors": [{"name": "Ada Lovelace"}],
                "citationCount": 42,
                "s2FieldsOfStudy": [{"category": "Computer Science"}],
            }
        )

        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual(candidate["arxiv_id"], "2101.00001")
        self.assertEqual(candidate["doi"], "10.1/example")
        self.assertEqual(candidate["title"], "Test Paper")
        self.assertEqual(candidate["summary"], "Abstract")
        self.assertEqual(candidate["authors"], ["Ada Lovelace"])
        self.assertEqual(candidate["primary_category"], "Computer Science")
        self.assertEqual(candidate["candidate_source"], "s2-search")

    def test_seed_keyword_filters(self) -> None:
        theory_entry = arxiv_entry("2101.00001")
        empirical_entry = arxiv_entry("2101.00002")
        unrelated_entry = arxiv_entry("2101.00003")
        theory_paper = paper("T", "Circuit lower bounds for small-depth proof systems", 2021, 10)
        empirical_paper = paper("E", "A dataset benchmark for neural image segmentation", 2021, 10)
        unrelated_paper = paper("U", "A survey of database systems", 2021, 10)
        s2_papers = {
            "ARXIV:2101.00001": theory_paper,
            "ARXIV:2101.00002": empirical_paper,
            "ARXIV:2101.00003": unrelated_paper,
        }

        selected = graph_builder.select_seed_papers(
            [theory_entry, empirical_entry, unrelated_entry],
            s2_papers,
            args(
                min_citations=0,
                max_seed_papers=0,
                min_year=None,
                max_year=None,
                seed_include_keywords="lower bounds,proof systems",
                seed_exclude_keywords="dataset,benchmark,neural,image segmentation",
            ),
            quiet_progress(),
        )

        self.assertEqual([(entry["arxiv_id"], paper["paperId"]) for entry, paper in selected], [("2101.00001", "T")])

    def test_seed_publication_filters(self) -> None:
        conference_entry = arxiv_entry("2101.00001")
        journal_entry = arxiv_entry("2101.00002")
        preprint_entry = arxiv_entry("2101.00003")
        missing_signal_entry = arxiv_entry("2101.00004")

        conference_paper = paper("C", "A Faster Approximation Algorithm", 2021, 30)
        conference_paper["publicationTypes"] = ["Conference"]
        conference_paper["venue"] = "ACM Symposium on Theory of Computing"

        journal_paper = paper("J", "Circuit Lower Bounds", 2020, 20)
        journal_paper["publicationTypes"] = ["JournalArticle"]
        journal_paper["venue"] = "Journal of the ACM"

        preprint_paper = paper("P", "An Unpublished Preprint", 2021, 100)
        preprint_paper["publicationTypes"] = ["Preprint"]
        preprint_paper["venue"] = "arXiv.org"

        missing_signal_paper = paper("M", "No Publication Signal", 2021, 100)

        self.assertTrue(graph_builder.paper_has_publication_signal(conference_paper))
        self.assertTrue(graph_builder.paper_has_publication_signal(journal_paper))
        self.assertFalse(graph_builder.paper_has_publication_signal(preprint_paper))

        s2_papers = {
            "ARXIV:2101.00001": conference_paper,
            "ARXIV:2101.00002": journal_paper,
            "ARXIV:2101.00003": preprint_paper,
            "ARXIV:2101.00004": missing_signal_paper,
        }

        selected = graph_builder.select_seed_papers(
            [conference_entry, journal_entry, preprint_entry, missing_signal_entry],
            s2_papers,
            args(
                min_citations=0,
                max_seed_papers=0,
                min_year=None,
                max_year=None,
                seed_publication_types="Conference,JournalArticle",
                require_seed_publication=True,
            ),
            quiet_progress(),
        )

        self.assertEqual(
            [(entry["arxiv_id"], paper["paperId"]) for entry, paper in selected],
            [("2101.00001", "C"), ("2101.00002", "J")],
        )

    def test_retry_after_uses_custom_429_backoff_bounds(self) -> None:
        response = graph_builder.requests.Response()
        response.status_code = 429

        self.assertEqual(
            graph_builder.retry_after_seconds(
                response,
                0,
                rate_limit_base_sleep=300.0,
                rate_limit_max_sleep=900.0,
                retry_jitter=0.0,
            ),
            300.0,
        )
        self.assertEqual(
            graph_builder.retry_after_seconds(
                response,
                4,
                rate_limit_base_sleep=300.0,
                rate_limit_max_sleep=900.0,
                retry_jitter=0.0,
            ),
            900.0,
        )

    def test_retry_after_header_overrides_429_backoff(self) -> None:
        response = graph_builder.requests.Response()
        response.status_code = 429
        response.headers["Retry-After"] = "17"

        self.assertEqual(
            graph_builder.retry_after_seconds(
                response,
                4,
                rate_limit_base_sleep=300.0,
                rate_limit_max_sleep=900.0,
                retry_jitter=0.0,
            ),
            17.0,
        )

    def test_request_before_attempt_runs_for_each_retry(self) -> None:
        session = FakeRetrySession([429, 200])
        before_attempt_calls: list[str] = []

        response = graph_builder.request_with_retries(
            session,  # type: ignore[arg-type]
            "GET",
            "https://example.test",
            max_retries=1,
            timeout=1.0,
            rate_limit_base_sleep=0.0,
            rate_limit_max_sleep=0.0,
            retry_jitter=0.0,
            before_attempt=lambda: before_attempt_calls.append("called"),
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(session.requests, 2)
        self.assertEqual(before_attempt_calls, ["called", "called"])

    def test_depth_two_fetches_frontier_and_applies_reference_breadth(self) -> None:
        skipped_direct = paper("A", "Lower ranked reference", 2019, 1)
        selected_direct = paper("B", "Higher ranked reference", 2020, 100)
        selected_deep = paper("C", "Higher ranked deep reference", 2018, 50)
        skipped_deep = paper("D", "Lower ranked deep reference", 2018, 2)
        seed = paper(
            "S",
            "Seed",
            2021,
            500,
            references=[skipped_direct, selected_direct],
            arxiv_id="2101.00001",
        )
        fetched_selected_direct = paper(
            "B",
            "Higher ranked reference",
            2020,
            100,
            references=[skipped_deep, selected_deep],
        )
        client = FakeSemanticScholarClient({"B": fetched_selected_direct})

        nodes, edges, skipped, stats = graph_builder.build_graph(
            [(arxiv_entry("2101.00001"), seed)],
            args(citation_depth=2, max_references_per_paper=1),
            quiet_progress(),
            client,
        )

        node_ids = {node["node_id"] for node in nodes}
        edge_pairs = {(edge["source"], edge["target"]) for edge in edges}
        self.assertEqual(client.requests, [["B"]])
        self.assertIn("s2:B", node_ids)
        self.assertIn("s2:C", node_ids)
        self.assertNotIn("s2:A", node_ids)
        self.assertNotIn("s2:D", node_ids)
        self.assertEqual(edge_pairs, {("s2:B", "s2:S"), ("s2:C", "s2:B")})
        self.assertEqual(skipped["references_breadth_cap"], 2)
        self.assertEqual(stats["expanded_depth_counts"], {"0": 1, "1": 1})

    def test_frontier_breadth_caps_recursive_expansion_only(self) -> None:
        expanded_reference = paper("A", "Expanded reference", 2020, 100)
        unexpanded_reference = paper("B", "Unexpanded reference", 2020, 50)
        deep_reference = paper("C", "Deep reference", 2019, 10)
        skipped_deep_reference = paper("D", "Skipped deep reference", 2019, 10)
        seed = paper(
            "S",
            "Seed",
            2021,
            500,
            references=[unexpanded_reference, expanded_reference],
            arxiv_id="2101.00001",
        )
        client = FakeSemanticScholarClient(
            {
                "A": paper(
                    "A",
                    "Expanded reference",
                    2020,
                    100,
                    references=[deep_reference],
                ),
                "B": paper(
                    "B",
                    "Unexpanded reference",
                    2020,
                    50,
                    references=[skipped_deep_reference],
                ),
            }
        )

        nodes, edges, skipped, _ = graph_builder.build_graph(
            [(arxiv_entry("2101.00001"), seed)],
            args(citation_depth=2, max_expansion_papers_per_depth=1),
            quiet_progress(),
            client,
        )

        node_ids = {node["node_id"] for node in nodes}
        edge_pairs = {(edge["source"], edge["target"]) for edge in edges}
        self.assertEqual(client.requests, [["A"]])
        self.assertIn("s2:A", node_ids)
        self.assertIn("s2:B", node_ids)
        self.assertIn("s2:C", node_ids)
        self.assertNotIn("s2:D", node_ids)
        self.assertEqual(
            edge_pairs,
            {("s2:A", "s2:S"), ("s2:B", "s2:S"), ("s2:C", "s2:A")},
        )
        self.assertEqual(skipped["expansion_breadth_cap"], 1)

    def test_target_node_count_skips_new_reference_nodes_after_budget(self) -> None:
        first_reference = paper("A", "First reference", 2020, 10)
        second_reference = paper("B", "Second reference", 2020, 10)
        seed = paper(
            "S",
            "Seed",
            2021,
            500,
            references=[first_reference, second_reference],
            arxiv_id="2101.00001",
        )

        nodes, edges, skipped, stats = graph_builder.build_graph(
            [(arxiv_entry("2101.00001"), seed)],
            args(citation_depth=1, target_node_count=2),
            quiet_progress(),
        )

        node_ids = {node["node_id"] for node in nodes}
        edge_pairs = {(edge["source"], edge["target"]) for edge in edges}
        self.assertEqual(node_ids, {"s2:S", "s2:A"})
        self.assertEqual(edge_pairs, {("s2:A", "s2:S")})
        self.assertEqual(skipped["target_node_budget"], 1)
        self.assertTrue(stats["target_node_budget_reached"])

    def test_reference_keyword_exclude_skips_reference_nodes(self) -> None:
        theory_reference = paper("A", "Proof complexity lower bounds", 2020, 10)
        empirical_reference = paper("B", "A neural benchmark dataset", 2020, 10)
        seed = paper(
            "S",
            "Seed",
            2021,
            500,
            references=[theory_reference, empirical_reference],
            arxiv_id="2101.00001",
        )

        nodes, edges, skipped, _ = graph_builder.build_graph(
            [(arxiv_entry("2101.00001"), seed)],
            args(citation_depth=1, reference_exclude_keywords="neural,benchmark,dataset"),
            quiet_progress(),
        )

        node_ids = {node["node_id"] for node in nodes}
        edge_pairs = {(edge["source"], edge["target"]) for edge in edges}
        self.assertEqual(node_ids, {"s2:S", "s2:A"})
        self.assertEqual(edge_pairs, {("s2:A", "s2:S")})
        self.assertEqual(skipped["reference_excluded_keywords"], 1)


if __name__ == "__main__":
    unittest.main()
