from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from datasets import Dataset, DatasetDict, load_dataset
from tqdm import tqdm


BRIGHT_DATASET = "xlangai/BRIGHT"
DEFAULT_OUTPUT_ROOT = Path("data/bright")
DEFAULT_QUERIES_PER_DOMAIN = 10
DEFAULT_SEED = 0
HASH_LENGTH = 12
MAX_DOCID_SLUG_LENGTH = 96


@dataclass(frozen=True)
class SelectedExample:
    domain: str
    query_id: str
    question_id: str
    query: str
    gold_answer: str
    gold_ids: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a browse-comp-style gold-only filesystem dataset from BRIGHT."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--queries-per-domain", type=int, default=DEFAULT_QUERIES_PER_DOMAIN)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--domains",
        nargs="+",
        default=None,
        help="Optional BRIGHT domain split names to process. Defaults to all domains.",
    )
    args = parser.parse_args()

    if args.queries_per_domain < 1:
        parser.error("--queries-per-domain must be at least 1")
    return args


def slugify(value: object) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_")
    return re.sub(r"_+", "_", slug) or "unknown"


def shorten_slug(slug: str, hash_source: str, max_length: int) -> str:
    if len(slug) <= max_length:
        return slug

    digest = hashlib.sha256(hash_source.encode("utf-8")).hexdigest()[:HASH_LENGTH]
    prefix_length = max_length - HASH_LENGTH - 1
    return f"{slug[:prefix_length].rstrip('_')}_{digest}"


def sanitize_docid(docid: object) -> str:
    raw_docid = str(docid or "unknown")
    return shorten_slug(slugify(raw_docid), raw_docid, MAX_DOCID_SLUG_LENGTH)


def unique_doc_ids(doc_ids: Iterable[Any]) -> tuple[str, ...]:
    seen: set[str] = set()
    unique: list[str] = []
    for raw_doc_id in doc_ids:
        doc_id = str(raw_doc_id)
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        unique.append(doc_id)
    return tuple(unique)


def doc_filename(doc_id: str, used_names: set[str]) -> str:
    base_name = sanitize_docid(doc_id)
    filename = f"{base_name}.txt"

    suffix = 2
    while filename in used_names:
        filename = f"{base_name}_{suffix}.txt"
        suffix += 1

    used_names.add(filename)
    return filename


def write_text(path: Path, text: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(text or ""), encoding="utf-8")


def load_bright() -> tuple[DatasetDict, DatasetDict]:
    examples = load_dataset(BRIGHT_DATASET, "examples")
    documents = load_dataset(BRIGHT_DATASET, "documents")
    if not isinstance(examples, DatasetDict) or not isinstance(documents, DatasetDict):
        raise TypeError("Expected BRIGHT examples and documents to load as DatasetDict objects.")
    return examples, documents


def resolve_domains(
    examples: DatasetDict,
    documents: DatasetDict,
    requested_domains: list[str] | None,
) -> list[str]:
    available_domains = list(examples.keys())
    missing_document_domains = sorted(set(available_domains) - set(documents.keys()))
    if missing_document_domains:
        raise ValueError(
            "BRIGHT document splits are missing for examples domains: "
            + ", ".join(missing_document_domains)
        )

    if requested_domains is None:
        return available_domains

    requested_unique = list(dict.fromkeys(requested_domains))
    unknown_domains = sorted(set(requested_unique) - set(available_domains))
    if unknown_domains:
        raise ValueError(
            "Unknown BRIGHT domain(s): "
            + ", ".join(unknown_domains)
            + ". Available domains: "
            + ", ".join(available_domains)
        )
    return requested_unique


def select_domain_examples(
    examples: Dataset,
    *,
    domain: str,
    queries_per_domain: int,
    seed: int,
) -> list[SelectedExample]:
    if len(examples) < queries_per_domain:
        raise ValueError(
            f"Domain {domain!r} only has {len(examples)} examples; "
            f"cannot select {queries_per_domain}."
        )

    rng = random.Random(f"{seed}:{domain}")
    selected_indices = sorted(rng.sample(range(len(examples)), queries_per_domain))
    selected: list[SelectedExample] = []

    for index in selected_indices:
        row = examples[index]
        query_id = str(row["id"])
        gold_ids = unique_doc_ids(row.get("gold_ids", []))
        if not gold_ids:
            raise ValueError(f"Selected example {domain}/{query_id} has no gold_ids.")
        selected.append(
            SelectedExample(
                domain=domain,
                query_id=query_id,
                question_id=f"{slugify(domain)}__{slugify(query_id)}",
                query=str(row["query"]),
                gold_answer=str(row.get("gold_answer", "")),
                gold_ids=gold_ids,
            )
        )

    return selected


def collect_selected_examples(
    examples: DatasetDict,
    domains: list[str],
    *,
    queries_per_domain: int,
    seed: int,
) -> dict[str, list[SelectedExample]]:
    selected_by_domain: dict[str, list[SelectedExample]] = {}
    for domain in domains:
        selected_by_domain[domain] = select_domain_examples(
            examples[domain],
            domain=domain,
            queries_per_domain=queries_per_domain,
            seed=seed,
        )
    return selected_by_domain


def collect_required_doc_ids(selected_examples: list[SelectedExample]) -> set[str]:
    required_doc_ids: set[str] = set()
    for example in selected_examples:
        required_doc_ids.update(example.gold_ids)
    return required_doc_ids


def load_required_documents(documents: Dataset, required_doc_ids: set[str]) -> dict[str, str]:
    if not required_doc_ids:
        return {}

    remaining_doc_ids = set(required_doc_ids)
    loaded_docs: dict[str, str] = {}

    for row in documents:
        doc_id = str(row["id"])
        if doc_id not in remaining_doc_ids:
            continue

        loaded_docs[doc_id] = str(row.get("content", ""))
        remaining_doc_ids.remove(doc_id)
        if not remaining_doc_ids:
            break

    return loaded_docs


def validate_required_documents(
    selected_by_domain: dict[str, list[SelectedExample]],
    docs_by_domain: dict[str, dict[str, str]],
) -> None:
    missing_by_domain: dict[str, list[str]] = {}
    for domain, selected_examples in selected_by_domain.items():
        available_doc_ids = set(docs_by_domain[domain])
        missing_doc_ids = sorted(collect_required_doc_ids(selected_examples) - available_doc_ids)
        if missing_doc_ids:
            missing_by_domain[domain] = missing_doc_ids

    if not missing_by_domain:
        return

    lines = ["Selected BRIGHT gold_ids were missing from the documents split:"]
    for domain, missing_doc_ids in missing_by_domain.items():
        preview = ", ".join(missing_doc_ids[:10])
        suffix = "" if len(missing_doc_ids) <= 10 else f", ... ({len(missing_doc_ids)} total)"
        lines.append(f"- {domain}: {preview}{suffix}")
    raise ValueError("\n".join(lines))


def manifest_doc_record(
    *,
    example: SelectedExample,
    doc_id: str,
    label: str,
    filename: str,
) -> dict[str, str]:
    return {
        "domain": example.domain,
        "bright_query_id": example.query_id,
        "label": label,
        "docid": doc_id,
        "agent_file": str(Path(example.question_id) / filename),
        "privileged_file": str(Path(example.question_id) / label / filename),
    }


def write_query_files(
    *,
    example: SelectedExample,
    domain_gold_ids: set[str],
    docs_by_id: dict[str, str],
    train_dir: Path,
    privileged_dir: Path,
) -> dict[str, Any]:
    agent_query_dir = train_dir / example.question_id
    privileged_query_dir = privileged_dir / example.question_id
    if agent_query_dir.exists():
        shutil.rmtree(agent_query_dir)
    if privileged_query_dir.exists():
        shutil.rmtree(privileged_query_dir)
    agent_query_dir.mkdir(parents=True, exist_ok=True)
    privileged_query_dir.mkdir(parents=True, exist_ok=True)

    write_text(privileged_query_dir / "query.txt", example.query)
    write_text(privileged_query_dir / "answer.txt", example.gold_answer)

    manifest: dict[str, Any] = {
        "query_id": example.question_id,
        "domain": example.domain,
        "bright_query_id": example.query_id,
        "query_file": "query.txt",
        "answer_file": "answer.txt",
        "documents": [],
    }

    used_names: set[str] = set()
    current_gold_ids = set(example.gold_ids)
    negative_ids = sorted(domain_gold_ids - current_gold_ids)

    for label, doc_ids in (("gold_docs", example.gold_ids), ("negative_docs", negative_ids)):
        label_dir = privileged_query_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)

        for doc_id in doc_ids:
            filename = doc_filename(doc_id, used_names)
            text = docs_by_id[doc_id]
            write_text(agent_query_dir / filename, text)
            write_text(label_dir / filename, text)
            manifest["documents"].append(
                manifest_doc_record(
                    example=example,
                    doc_id=doc_id,
                    label=label,
                    filename=filename,
                )
            )

    (privileged_query_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    return {
        "question_id": example.question_id,
        "question": example.query,
        "domain": example.domain,
        "bright_query_id": example.query_id,
    }


def write_dataset(
    *,
    selected_by_domain: dict[str, list[SelectedExample]],
    docs_by_domain: dict[str, dict[str, str]],
    output_root: Path,
    queries_per_domain: int,
    seed: int,
) -> None:
    train_dir = output_root / "train"
    privileged_dir = output_root / "train_privileged"
    train_dir.mkdir(parents=True, exist_ok=True)
    privileged_dir.mkdir(parents=True, exist_ok=True)

    questions: list[dict[str, Any]] = []
    for domain, selected_examples in selected_by_domain.items():
        domain_gold_ids = collect_required_doc_ids(selected_examples)
        for example in tqdm(selected_examples, desc=f"Writing {domain}"):
            questions.append(
                write_query_files(
                    example=example,
                    domain_gold_ids=domain_gold_ids,
                    docs_by_id=docs_by_domain[domain],
                    train_dir=train_dir,
                    privileged_dir=privileged_dir,
                )
            )

    payload = {
        "name": f"bright_{queries_per_domain}_questions_per_domain",
        "source_dataset": BRIGHT_DATASET,
        "queries_per_domain": queries_per_domain,
        "seed": seed,
        "num_questions": len(questions),
        "questions": questions,
    }
    (output_root / f"heldout_{queries_per_domain}_questions.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    examples, documents = load_bright()
    domains = resolve_domains(examples, documents, args.domains)
    selected_by_domain = collect_selected_examples(
        examples,
        domains,
        queries_per_domain=args.queries_per_domain,
        seed=args.seed,
    )

    docs_by_domain: dict[str, dict[str, str]] = {}
    for domain in domains:
        required_doc_ids = collect_required_doc_ids(selected_by_domain[domain])
        docs_by_domain[domain] = load_required_documents(documents[domain], required_doc_ids)

    validate_required_documents(selected_by_domain, docs_by_domain)
    write_dataset(
        selected_by_domain=selected_by_domain,
        docs_by_domain=docs_by_domain,
        output_root=args.output_root,
        queries_per_domain=args.queries_per_domain,
        seed=args.seed,
    )

    print(
        f"Wrote {sum(len(examples) for examples in selected_by_domain.values())} BRIGHT queries "
        f"across {len(domains)} domain(s) to {args.output_root}"
    )


if __name__ == "__main__":
    main()
