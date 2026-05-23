#!/usr/bin/env python3
"""Build Tinker synthetic-FS index files from the local filesystem dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create index.jsonl files consumed by train_synthetic_fs_rl.py from "
            "data/train and data/train_privileged."
        )
    )
    parser.add_argument("--agent-dir", default="../data/train")
    parser.add_argument("--privileged-dir", default="../data/train_privileged")
    parser.add_argument("--out-dir", default="../data/tinker_synthetic_fs_alltrain")
    parser.add_argument(
        "--heldout-questions-json",
        default="/iris/u/asap7772/asmr_private/data/heldout_50_questions.json",
        help=(
            "Optional JSON file containing held-out question_id/query_id values. "
            "When set with --eval-out-dir, these rows are written to the eval index "
            "and excluded from the train index."
        ),
    )
    parser.add_argument("--eval-out-dir", default="../data/tinker_synthetic_fs_alltest")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--max-documents",
        "--max-docs",
        dest="max_documents",
        type=int,
        default=0,
        help=(
            "Maximum number of agent-visible documents per question. "
            "When a row exceeds this cap, negative_docs are pruned first. "
            "Set to 0 to disable document pruning."
        ),
    )
    parser.add_argument(
        "--evidence-gold-only",
        type=bool,
        help="Keep only documents labeled evidence_docs or gold_docs.",
        default=False,
    )
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be >= 0")
    if args.max_documents < 0:
        parser.error("--max-documents must be >= 0")
    return args


def qid_sort_key(path: Path) -> tuple[int, int | str]:
    name = path.name
    if name.isdigit():
        return (0, int(name))
    return (1, name)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_heldout_qids(path: Path) -> set[str]:
    if not path:
        return set()
    payload = load_json(path)
    rows = payload.get("questions", payload) if isinstance(payload, dict) else payload
    qids: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        value = row.get("question_id") or row.get("query_id") or row.get("id")
        if value is not None:
            qids.add(str(value))
    return qids


def relative_agent_path(raw_agent_file: object, qid: str) -> str:
    path = PurePosixPath(str(raw_agent_file))
    parts = path.parts
    if parts and parts[0] == qid:
        return str(PurePosixPath(*parts[1:]))
    return str(path)


def load_document_metadata(privileged_query_dir: Path, qid: str) -> dict[str, dict[str, Any]]:
    manifest_path = privileged_query_dir / "manifest.json"
    if not manifest_path.exists():
        return {}

    manifest = load_json(manifest_path)
    by_relative_path: dict[str, dict[str, Any]] = {}
    for doc in manifest.get("documents", []):
        if not isinstance(doc, dict) or not doc.get("agent_file"):
            continue
        label = str(doc.get("label", ""))
        relative_path = relative_agent_path(doc["agent_file"], qid)
        by_relative_path[relative_path] = {
            "relative_path": relative_path,
            "doc_id": doc.get("docid") or doc.get("doc_id"),
            "url": doc.get("url"),
            "is_evidence": label in {"evidence_docs", "gold_docs"},
            "is_gold": label == "gold_docs",
            "is_negative": label == "negative_docs",
            "label": label,
        }
    return by_relative_path


def prune_negative_documents(
    files: list[dict[str, Any]], max_documents: int, qid: str
) -> list[dict[str, Any]]:
    if max_documents <= 0 or len(files) <= max_documents:
        return files

    protected_count = sum(1 for file in files if not file.get("is_negative", False))
    if protected_count > max_documents:
        raise ValueError(
            f"Cannot limit {qid} to {max_documents} documents without pruning "
            f"non-negative documents ({protected_count} non-negative docs present)."
        )

    negatives_to_keep = max_documents - protected_count
    kept_negatives = 0
    pruned_files: list[dict[str, Any]] = []
    for file in files:
        if file.get("is_negative", False):
            if kept_negatives >= negatives_to_keep:
                continue
            kept_negatives += 1
        pruned_files.append(file)
    return pruned_files


def keep_evidence_gold_documents(
    files: list[dict[str, Any]], qid: str
) -> list[dict[str, Any]]:
    kept_files = [
        file
        for file in files
        if file.get("label") in {"evidence_docs", "gold_docs"}
    ]
    if not kept_files:
        raise ValueError(f"No evidence_docs or gold_docs found for {qid}.")
    return kept_files


def build_row(
    agent_query_dir: Path,
    privileged_query_dir: Path,
    max_documents: int = 0,
    evidence_gold_only: bool = False,
) -> dict[str, Any]:
    qid = agent_query_dir.name
    query_path = privileged_query_dir / "query.txt"
    answer_path = privileged_query_dir / "answer.txt"
    if not query_path.exists():
        raise ValueError(f"Missing query.txt for {qid}: {query_path}")
    if not answer_path.exists():
        raise ValueError(f"Missing answer.txt for {qid}: {answer_path}")

    metadata = load_document_metadata(privileged_query_dir, qid)
    files: list[dict[str, Any]] = []
    for path in sorted(p for p in agent_query_dir.rglob("*") if p.is_file()):
        relative_path = path.relative_to(agent_query_dir).as_posix()
        info = dict(metadata.get(relative_path, {}))
        if not info:
            info = {
                "relative_path": relative_path,
                "doc_id": path.stem.split("_", 1)[0],
                "url": None,
                "is_evidence": True,
                "is_gold": False,
                "is_negative": False,
                "label": "unknown",
            }
        files.append(info)

    if not files:
        raise ValueError(f"No agent-visible files found for {qid}: {agent_query_dir}")

    if evidence_gold_only:
        files = keep_evidence_gold_documents(files, qid)
    files = prune_negative_documents(files, max_documents, qid)

    return {
        "question_id": qid,
        "agent_query_dir": str(agent_query_dir.resolve()),
        "privileged_query_dir": str(privileged_query_dir.resolve()),
        "num_docs": len(files),
        "dataset_type": "browsecomp_plus_local_filesystem",
        "files": files,
    }


def write_index(out_dir: Path, rows: list[dict[str, Any]], *, source: dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.jsonl"
    with index_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        **source,
        "out_dir": str(out_dir.resolve()),
        "index_jsonl": str(index_path.resolve()),
        "num_examples": len(rows),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return index_path


def main() -> None:
    args = parse_args()
    agent_dir = Path(args.agent_dir)
    privileged_dir = Path(args.privileged_dir)
    if not agent_dir.exists():
        raise SystemExit(f"agent dir does not exist: {agent_dir}")
    if not privileged_dir.exists():
        raise SystemExit(f"privileged dir does not exist: {privileged_dir}")

    heldout_qids = (
        load_heldout_qids(Path(args.heldout_questions_json))
        if args.heldout_questions_json
        else set()
    )

    all_rows: list[dict[str, Any]] = []
    for agent_query_dir in sorted(agent_dir.iterdir(), key=qid_sort_key):
        if not agent_query_dir.is_dir():
            continue
        privileged_query_dir = privileged_dir / agent_query_dir.name
        if not privileged_query_dir.exists():
            raise ValueError(f"Missing privileged dir for {agent_query_dir.name}: {privileged_query_dir}")
        all_rows.append(
            build_row(
                agent_query_dir,
                privileged_query_dir,
                max_documents=args.max_documents,
                evidence_gold_only=args.evidence_gold_only,
            )
        )
        if args.limit and len(all_rows) >= args.limit:
            break

    if not all_rows:
        raise SystemExit("No dataset rows were built.")

    if args.eval_out_dir and heldout_qids:
        train_rows = [row for row in all_rows if row["question_id"] not in heldout_qids]
        eval_rows = [row for row in all_rows if row["question_id"] in heldout_qids]
    else:
        train_rows = all_rows
        eval_rows = []

    source = {
        "source": "local data/train plus data/train_privileged",
        "agent_dir": str(agent_dir.resolve()),
        "privileged_dir": str(privileged_dir.resolve()),
        "heldout_questions_json": str(Path(args.heldout_questions_json).resolve())
        if args.heldout_questions_json
        else "",
        "heldout_qids": len(heldout_qids),
        "max_documents": args.max_documents,
        "evidence_gold_only": args.evidence_gold_only,
    }
    train_index = write_index(Path(args.out_dir), train_rows, source=source)
    print(f"wrote train index: {train_index}")
    print(f"train_rows={len(train_rows)}")

    if args.eval_out_dir and heldout_qids:
        eval_index = write_index(Path(args.eval_out_dir), eval_rows, source=source)
        print(f"wrote eval index: {eval_index}")
        print(f"eval_rows={len(eval_rows)}")
        missing = heldout_qids - {row["question_id"] for row in eval_rows}
        if missing:
            print(f"warning: heldout qids missing from local data: {sorted(missing)}")


if __name__ == "__main__":
    main()
