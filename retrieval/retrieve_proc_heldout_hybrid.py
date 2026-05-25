from __future__ import annotations

import argparse
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

try:
    from retrieval.retrieve_proc_heldout import (
        DEFAULT_DOCS_DIR,
        DEFAULT_PRIVILEGED_DIR,
        DEFAULT_QUESTIONS_PATH,
        CandidateDocument,
        Qwen3EmbeddingVllm,
        batches,
        load_candidate_documents,
        load_heldout_questions,
        maybe_normalize,
        select_questions,
        write_jsonl,
    )
except ModuleNotFoundError as exc:
    if exc.name not in {"retrieval", "retrieval.retrieve_proc_heldout"}:
        raise
    from retrieve_proc_heldout import (
        DEFAULT_DOCS_DIR,
        DEFAULT_PRIVILEGED_DIR,
        DEFAULT_QUESTIONS_PATH,
        CandidateDocument,
        Qwen3EmbeddingVllm,
        batches,
        load_candidate_documents,
        load_heldout_questions,
        maybe_normalize,
        select_questions,
        write_jsonl,
    )


logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLD_AND_SUPPORT_ONLY=False
DEFAULT_OUTPUT_PATH = REPO_ROOT / "retrieval" / "heldout_retrieval_hybrid.jsonl"
TOKEN_RE = re.compile(r"(?u)\b\w+\b")


@dataclass(frozen=True)
class HybridRetrievedDocument:
    document: CandidateDocument
    retrieval_score: float
    retrieval_rank: int
    embedding_score: float
    embedding_rank: int
    bm25_score: float
    bm25_rank: int
    embedding_normalized_score: float | None = None
    bm25_normalized_score: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hybrid BM25 + Qwen embedding retrieval for heldout questions."
    )
    parser.add_argument("--questions-path", type=Path, default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS_DIR)
    parser.add_argument(
        "--privileged-dir",
        type=Path,
        default=DEFAULT_PRIVILEGED_DIR,
        help=(
            "Optional manifest source for docid/url/label metadata, e.g. data/train_privileged. "
            "Required by --gold-and-support-only."
        ),
    )
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)

    parser.add_argument(
        "--model-name-or-path",
        "--embedding-model-name-or-path",
        dest="embedding_model_name_or_path",
        default="Qwen/Qwen3-Embedding-4B",
    )
    parser.add_argument(
        "--instruction",
        "--embedding-instruction",
        dest="embedding_instruction",
        default=None,
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--dim", "--embedding-dim", dest="embedding_dim", type=int, default=1024)
    parser.add_argument(
        "--batch-size",
        "--embedding-batch-size",
        dest="embedding_batch_size",
        type=int,
        default=128,
    )
    parser.add_argument("--max-doc-chars", type=int, default=30_000)
    parser.add_argument(
        "--gold-and-support-only",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_GOLD_AND_SUPPORT_ONLY,
        help=(
            "Only retrieve from manifest docs labeled as gold or supporting/evidence docs. "
            "Requires --privileged-dir."
        ),
    )
    parser.add_argument(
        "--tensor-parallel-size",
        "--embedding-tensor-parallel-size",
        dest="embedding_tensor_parallel_size",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--max-model-len",
        "--embedding-max-model-len",
        dest="embedding_max_model_len",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        "--embedding-gpu-memory-utilization",
        dest="embedding_gpu_memory_utilization",
        type=float,
        default=None,
    )
    parser.add_argument("--normalize", action="store_true", help="L2-normalize embeddings before scoring.")

    parser.add_argument("--bm25-k1", type=float, default=1.5)
    parser.add_argument("--bm25-b", type=float, default=0.75)
    parser.add_argument("--embedding-weight", type=float, default=0.5)
    parser.add_argument("--bm25-weight", type=float, default=0.5)
    parser.add_argument(
        "--fusion-method",
        choices=("weighted-sum", "rrf"),
        default="weighted-sum",
        help="Use normalized score fusion or reciprocal rank fusion.",
    )
    parser.add_argument(
        "--score-normalization",
        choices=("minmax", "zscore", "none"),
        default="minmax",
        help="Per-question score normalization for weighted-sum fusion.",
    )
    parser.add_argument("--rrf-k", type=float, default=60.0)

    parser.add_argument("--no-text", action="store_true", help="Omit document text from the output JSONL.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--query-id",
        action="append",
        default=None,
        help="Only process this question id. Can be supplied multiple times.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    if args.embedding_batch_size < 1:
        parser.error("--embedding-batch-size must be at least 1")
    if args.max_doc_chars < 0:
        parser.error("--max-doc-chars must be >= 0")
    if args.gold_and_support_only and args.privileged_dir is None:
        parser.error("--gold-and-support-only requires --privileged-dir")
    if args.bm25_k1 <= 0:
        parser.error("--bm25-k1 must be > 0")
    if not 0 <= args.bm25_b <= 1:
        parser.error("--bm25-b must be between 0 and 1")
    if args.embedding_weight < 0 or args.bm25_weight < 0:
        parser.error("--embedding-weight and --bm25-weight must be >= 0")
    if args.embedding_weight == 0 and args.bm25_weight == 0:
        parser.error("At least one of --embedding-weight or --bm25-weight must be > 0")
    if args.rrf_k <= 0:
        parser.error("--rrf-k must be > 0")
    return args


def tokenize_for_bm25(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def compute_bm25_scores(
    question: str,
    documents: list[CandidateDocument],
    *,
    k1: float,
    b: float,
) -> list[float]:
    if not documents:
        return []

    tokenized_docs = [tokenize_for_bm25(doc.text) for doc in documents]
    query_terms = Counter(tokenize_for_bm25(question))
    if not query_terms:
        return [0.0 for _ in documents]

    term_frequencies: list[Counter[str]] = []
    document_frequencies: Counter[str] = Counter()
    doc_lengths: list[int] = []
    for tokens in tokenized_docs:
        counts = Counter(tokens)
        term_frequencies.append(counts)
        document_frequencies.update(counts.keys())
        doc_lengths.append(len(tokens))

    num_docs = len(documents)
    avg_doc_len = sum(doc_lengths) / num_docs if num_docs else 0.0
    scores: list[float] = []
    for counts, doc_len in zip(term_frequencies, doc_lengths):
        score = 0.0
        length_norm = k1 * (1 - b + b * (doc_len / avg_doc_len)) if avg_doc_len > 0 else k1
        for term, query_frequency in query_terms.items():
            term_frequency = counts.get(term, 0)
            if term_frequency <= 0:
                continue
            doc_frequency = document_frequencies.get(term, 0)
            idf = math.log(1 + (num_docs - doc_frequency + 0.5) / (doc_frequency + 0.5))
            numerator = term_frequency * (k1 + 1)
            denominator = term_frequency + length_norm
            score += query_frequency * idf * (numerator / denominator)
        scores.append(float(score))
    return scores


def compute_embedding_scores(
    model: Qwen3EmbeddingVllm,
    question: str,
    documents: list[CandidateDocument],
    *,
    dim: int,
    batch_size: int,
    normalize: bool,
) -> list[float]:
    if not documents:
        return []

    with torch.inference_mode():
        query_embedding = model.encode(question, is_query=True, dim=dim)
        query_embedding = maybe_normalize(query_embedding, normalize)

        doc_embeddings: list[torch.Tensor] = []
        for batch in batches(documents, batch_size):
            batch_embeddings = model.encode([doc.text for doc in batch], dim=dim)
            doc_embeddings.append(maybe_normalize(batch_embeddings, normalize))

        all_doc_embeddings = torch.cat(doc_embeddings, dim=0)
        scores = ((query_embedding @ all_doc_embeddings.T).squeeze(0) * 100).detach().cpu()

    return [float(score) for score in scores.tolist()]


def normalize_scores(scores: list[float], method: str) -> list[float]:
    if not scores:
        return []
    if method == "none":
        return [float(score) for score in scores]
    if method == "minmax":
        low = min(scores)
        high = max(scores)
        if math.isclose(high, low):
            return [0.0 for _ in scores]
        return [(float(score) - low) / (high - low) for score in scores]
    if method == "zscore":
        mean = sum(scores) / len(scores)
        variance = sum((score - mean) ** 2 for score in scores) / len(scores)
        std = math.sqrt(variance)
        if math.isclose(std, 0.0):
            return [0.0 for _ in scores]
        return [(float(score) - mean) / std for score in scores]
    raise ValueError(f"Unsupported score normalization method: {method}")


def ranked_indices(scores: list[float]) -> list[int]:
    return sorted(range(len(scores)), key=lambda index: (-scores[index], index))


def ranks_from_scores(scores: list[float]) -> list[int]:
    ranks = [0 for _ in scores]
    for rank, index in enumerate(ranked_indices(scores), start=1):
        ranks[index] = rank
    return ranks


def retrieve_hybrid_documents(
    model: Qwen3EmbeddingVllm,
    question: str,
    documents: list[CandidateDocument],
    *,
    top_k: int,
    embedding_dim: int,
    embedding_batch_size: int,
    normalize_embeddings: bool,
    bm25_k1: float,
    bm25_b: float,
    embedding_weight: float,
    bm25_weight: float,
    fusion_method: str,
    score_normalization: str,
    rrf_k: float,
) -> list[HybridRetrievedDocument]:
    if not documents:
        return []

    embedding_scores = compute_embedding_scores(
        model,
        question,
        documents,
        dim=embedding_dim,
        batch_size=embedding_batch_size,
        normalize=normalize_embeddings,
    )
    bm25_scores = compute_bm25_scores(question, documents, k1=bm25_k1, b=bm25_b)

    embedding_ranks = ranks_from_scores(embedding_scores)
    bm25_ranks = ranks_from_scores(bm25_scores)

    embedding_normalized_scores: list[float] | None = None
    bm25_normalized_scores: list[float] | None = None
    if fusion_method == "weighted-sum":
        embedding_normalized_scores = normalize_scores(embedding_scores, score_normalization)
        bm25_normalized_scores = normalize_scores(bm25_scores, score_normalization)
        fused_scores = [
            embedding_weight * embedding_score + bm25_weight * bm25_score
            for embedding_score, bm25_score in zip(embedding_normalized_scores, bm25_normalized_scores)
        ]
    elif fusion_method == "rrf":
        fused_scores = [
            (embedding_weight / (rrf_k + embedding_rank)) + (bm25_weight / (rrf_k + bm25_rank))
            for embedding_rank, bm25_rank in zip(embedding_ranks, bm25_ranks)
        ]
    else:
        raise ValueError(f"Unsupported fusion method: {fusion_method}")

    retrieved: list[HybridRetrievedDocument] = []
    for retrieval_rank, index in enumerate(ranked_indices(fused_scores)[: min(top_k, len(documents))], start=1):
        retrieved.append(
            HybridRetrievedDocument(
                document=documents[index],
                retrieval_score=float(fused_scores[index]),
                retrieval_rank=retrieval_rank,
                embedding_score=float(embedding_scores[index]),
                embedding_rank=embedding_ranks[index],
                bm25_score=float(bm25_scores[index]),
                bm25_rank=bm25_ranks[index],
                embedding_normalized_score=(
                    None if embedding_normalized_scores is None else float(embedding_normalized_scores[index])
                ),
                bm25_normalized_score=(
                    None if bm25_normalized_scores is None else float(bm25_normalized_scores[index])
                ),
            )
        )
    return retrieved


def hybrid_retrieved_document_to_record(
    doc: HybridRetrievedDocument,
    *,
    include_text: bool,
) -> dict[str, Any]:
    candidate = doc.document
    record: dict[str, Any] = {
        "rank": doc.retrieval_rank,
        "retrieval_rank": doc.retrieval_rank,
        "retrieval_score": doc.retrieval_score,
        "embedding_score": doc.embedding_score,
        "embedding_rank": doc.embedding_rank,
        "bm25_score": doc.bm25_score,
        "bm25_rank": doc.bm25_rank,
        "path": candidate.rel_path,
        "filename": candidate.path.name,
        "truncated": candidate.truncated,
    }
    if doc.embedding_normalized_score is not None:
        record["embedding_normalized_score"] = doc.embedding_normalized_score
    if doc.bm25_normalized_score is not None:
        record["bm25_normalized_score"] = doc.bm25_normalized_score
    record.update({key: value for key, value in candidate.metadata.items() if value is not None})
    if include_text:
        record["text"] = candidate.text
    return record


def build_hybrid_retrieval_record(
    question_entry: dict[str, str],
    documents: list[CandidateDocument],
    retrieved_documents: list[HybridRetrievedDocument],
    *,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "question_id": question_entry["question_id"],
        "question": question_entry["question"],
        "num_candidate_docs": len(documents),
        "top_k": args.top_k,
        "gold_and_support_only": bool(args.gold_and_support_only),
        "retrieval_model": "hybrid_bm25_qwen_embedding",
        "embedding_model": args.embedding_model_name_or_path,
        "embedding_dim": args.embedding_dim,
        "normalized": bool(args.normalize),
        "embedding_normalized": bool(args.normalize),
        "bm25_k1": args.bm25_k1,
        "bm25_b": args.bm25_b,
        "fusion_method": args.fusion_method,
        "score_normalization": args.score_normalization,
        "embedding_weight": args.embedding_weight,
        "bm25_weight": args.bm25_weight,
        "rrf_k": args.rrf_k,
        "documents": [
            hybrid_retrieved_document_to_record(doc, include_text=not args.no_text)
            for doc in retrieved_documents
        ],
    }


def build_error_record(
    question_entry: dict[str, str],
    error: Exception,
    *,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "question_id": question_entry["question_id"],
        "question": question_entry["question"],
        "gold_and_support_only": bool(args.gold_and_support_only),
        "retrieval_model": "hybrid_bm25_qwen_embedding",
        "embedding_model": args.embedding_model_name_or_path,
        "fusion_method": args.fusion_method,
        "error": str(error),
        "documents": [],
    }


def run_hybrid_retrieval(args: argparse.Namespace) -> list[dict[str, Any]]:
    questions = select_questions(load_heldout_questions(args.questions_path), args.query_id, args.limit)
    logger.info("Loaded %d heldout questions", len(questions))
    if not questions:
        write_jsonl([], args.output_path)
        logger.info("Wrote 0 records to %s", args.output_path)
        return []

    model = Qwen3EmbeddingVllm(
        args.embedding_model_name_or_path,
        instruction=args.embedding_instruction,
        tensor_parallel_size=args.embedding_tensor_parallel_size,
        max_model_len=args.embedding_max_model_len,
        gpu_memory_utilization=args.embedding_gpu_memory_utilization,
    )

    records: list[dict[str, Any]] = []
    try:
        for question_entry in tqdm(questions, desc="Hybrid retrieving"):
            question_id = question_entry["question_id"]
            try:
                documents = load_candidate_documents(
                    args.docs_dir,
                    args.privileged_dir,
                    question_id,
                    args.max_doc_chars,
                    gold_and_support_only=bool(args.gold_and_support_only),
                )
                retrieved_documents = retrieve_hybrid_documents(
                    model,
                    question_entry["question"],
                    documents,
                    top_k=args.top_k,
                    embedding_dim=args.embedding_dim,
                    embedding_batch_size=args.embedding_batch_size,
                    normalize_embeddings=args.normalize,
                    bm25_k1=args.bm25_k1,
                    bm25_b=args.bm25_b,
                    embedding_weight=args.embedding_weight,
                    bm25_weight=args.bm25_weight,
                    fusion_method=args.fusion_method,
                    score_normalization=args.score_normalization,
                    rrf_k=args.rrf_k,
                )
                record = build_hybrid_retrieval_record(question_entry, documents, retrieved_documents, args=args)
            except Exception as exc:
                logger.exception("Failed hybrid retrieval for question %s", question_id)
                record = build_error_record(question_entry, exc, args=args)
            records.append(record)
    finally:
        model.stop()

    write_jsonl(records, args.output_path)
    logger.info("Wrote %d records to %s", len(records), args.output_path)
    return records


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    run_hybrid_retrieval(args)


if __name__ == "__main__":
    main()
