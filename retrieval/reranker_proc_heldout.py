from __future__ import annotations

import argparse
import gc
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel

try:
    from vllm.inputs import TokensPrompt
except ImportError:
    from vllm.inputs.data import TokensPrompt

try:
    from retrieval.retrieve_proc_heldout import (
        DEFAULT_DOCS_DIR,
        DEFAULT_QUESTIONS_PATH,
        CandidateDocument,
        Qwen3EmbeddingVllm,
        RetrievedDocument,
        batches,
        load_candidate_documents,
        load_heldout_questions,
        retrieve_documents,
        select_questions,
        write_jsonl,
    )
except ModuleNotFoundError as exc:
    if exc.name not in {"retrieval", "retrieval.retrieve_proc_heldout"}:
        raise
    from retrieve_proc_heldout import (
        DEFAULT_DOCS_DIR,
        DEFAULT_QUESTIONS_PATH,
        CandidateDocument,
        Qwen3EmbeddingVllm,
        RetrievedDocument,
        batches,
        load_candidate_documents,
        load_heldout_questions,
        retrieve_documents,
        select_questions,
        write_jsonl,
    )


logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = REPO_ROOT / "retrieval" / "heldout_retrieval_reranked.jsonl"


@dataclass(frozen=True)
class RetrievalResult:
    question_entry: dict[str, str]
    documents: list[CandidateDocument]
    retrieved_documents: list[RetrievedDocument]
    error: str | None = None


@dataclass(frozen=True)
class RerankedDocument:
    retrieved_document: RetrievedDocument
    rerank_score: float
    rerank_rank: int


class Qwen3RerankerVllm:
    def __init__(
        self,
        model_name_or_path: str,
        instruction: str = "Retrieve documents that can answer the user's query",
        max_length: int = 8192,
        tensor_parallel_size: int | None = None,
        max_model_len: int = 10000,
        gpu_memory_utilization: float = 0.8,
        distributed_executor_backend: str | None = None,
    ) -> None:
        if tensor_parallel_size is None:
            tensor_parallel_size = max(torch.cuda.device_count(), 1)

        self.instruction = instruction
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.suffix = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.max_length = max_length
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)
        self.true_token = self.tokenizer("yes", add_special_tokens=False).input_ids[0]
        self.false_token = self.tokenizer("no", add_special_tokens=False).input_ids[0]
        self.sampling_params = SamplingParams(
            temperature=0,
            top_p=0.95,
            max_tokens=1,
            logprobs=20,
            allowed_token_ids=[self.true_token, self.false_token],
        )

        llm_kwargs: dict[str, Any] = {
            "model": model_name_or_path,
            "tensor_parallel_size": tensor_parallel_size,
            "max_model_len": max_model_len,
            "enable_prefix_caching": True,
            "gpu_memory_utilization": gpu_memory_utilization,
        }
        if distributed_executor_backend:
            llm_kwargs["distributed_executor_backend"] = distributed_executor_backend

        self.lm = LLM(**llm_kwargs)

    def format_instruction(self, instruction: str, query: str, doc: str) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "Judge whether the Document meets the requirements based on the Query and "
                    'the Instruct provided. Note that the answer can only be "yes" or "no".'
                ),
            },
            {"role": "user", "content": f"<Instruct>: {instruction}\n\n<Query>: {query}\n\n<Document>: {doc}"},
        ]

    def compute_scores(self, pairs: list[tuple[str, str]]) -> list[float]:
        messages = [self.format_instruction(self.instruction, query, doc) for query, doc in pairs]
        tokenized_messages = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        tokenized_messages = [tokens[: self.max_length] + self.suffix_tokens for tokens in tokenized_messages]
        prompts = [TokensPrompt(prompt_token_ids=tokens) for tokens in tokenized_messages]

        outputs = self.lm.generate(prompts, self.sampling_params, use_tqdm=False)
        scores: list[float] = []
        for output in outputs:
            final_logits = output.outputs[0].logprobs[-1]
            true_logit = final_logits[self.true_token].logprob if self.true_token in final_logits else -10
            false_logit = final_logits[self.false_token].logprob if self.false_token in final_logits else -10
            true_score = math.exp(true_logit)
            false_score = math.exp(false_logit)
            scores.append(true_score / (true_score + false_score))
        return scores

    def stop(self) -> None:
        destroy_model_parallel()
        destroy_distributed_environment()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve and rerank candidate documents for heldout questions.")
    parser.add_argument("--questions-path", type=Path, default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS_DIR)
    parser.add_argument(
        "--privileged-dir",
        type=Path,
        default=None,
        help="Optional manifest source for docid/url/label metadata, e.g. data/train_privileged. Scores never use this.",
    )
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)

    parser.add_argument("--embedding-model-name-or-path", default="Qwen/Qwen3-Embedding-4B")
    parser.add_argument("--embedding-instruction", default=None)
    parser.add_argument("--embedding-dim", type=int, default=1024)
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--embedding-tensor-parallel-size", type=int, default=None)
    parser.add_argument("--embedding-max-model-len", type=int, default=None)
    parser.add_argument("--embedding-gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--normalize", action="store_true", help="L2-normalize embeddings before retrieval scoring.")

    parser.add_argument("--reranker-model-name-or-path", default="Qwen/Qwen3-Reranker-0.6B")
    parser.add_argument("--reranker-instruction", default="Retrieve documents that can answer the user's query")
    parser.add_argument("--reranker-max-length", type=int, default=2048)
    parser.add_argument("--reranker-max-model-len", type=int, default=10000)
    parser.add_argument("--reranker-batch-size", type=int, default=16)
    parser.add_argument("--reranker-tensor-parallel-size", type=int, default=None)
    parser.add_argument("--reranker-gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--reranker-distributed-executor-backend", default=None)

    parser.add_argument("--retrieval-top-k", type=int, default=50)
    parser.add_argument("--rerank-top-k", type=int, default=20)
    parser.add_argument("--max-doc-chars", type=int, default=30_000)
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

    if args.retrieval_top_k < 1:
        parser.error("--retrieval-top-k must be at least 1")
    if args.rerank_top_k < 1:
        parser.error("--rerank-top-k must be at least 1")
    if args.embedding_batch_size < 1:
        parser.error("--embedding-batch-size must be at least 1")
    if args.reranker_batch_size < 1:
        parser.error("--reranker-batch-size must be at least 1")
    if args.max_doc_chars < 0:
        parser.error("--max-doc-chars must be >= 0")
    return args


def collect_retrieval_candidates(args: argparse.Namespace) -> list[RetrievalResult]:
    questions = select_questions(load_heldout_questions(args.questions_path), args.query_id, args.limit)
    logger.info("Loaded %d heldout questions", len(questions))

    model = Qwen3EmbeddingVllm(
        args.embedding_model_name_or_path,
        instruction=args.embedding_instruction,
        tensor_parallel_size=args.embedding_tensor_parallel_size,
        max_model_len=args.embedding_max_model_len,
        gpu_memory_utilization=args.embedding_gpu_memory_utilization,
    )

    results: list[RetrievalResult] = []
    try:
        for question_entry in tqdm(questions, desc="Retrieving"):
            question_id = question_entry["question_id"]
            try:
                documents = load_candidate_documents(
                    args.docs_dir,
                    args.privileged_dir,
                    question_id,
                    args.max_doc_chars,
                )
                retrieved_documents = retrieve_documents(
                    model,
                    question_entry["question"],
                    documents,
                    top_k=args.retrieval_top_k,
                    dim=args.embedding_dim,
                    batch_size=args.embedding_batch_size,
                    normalize=args.normalize,
                )
                results.append(
                    RetrievalResult(
                        question_entry=question_entry,
                        documents=documents,
                        retrieved_documents=retrieved_documents,
                    )
                )
            except Exception as exc:
                logger.exception("Failed retrieval for question %s", question_id)
                results.append(
                    RetrievalResult(
                        question_entry=question_entry,
                        documents=[],
                        retrieved_documents=[],
                        error=str(exc),
                    )
                )
    finally:
        model.stop()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def rerank_documents(
    reranker: Qwen3RerankerVllm,
    question: str,
    retrieved_documents: list[RetrievedDocument],
    *,
    batch_size: int,
    rerank_top_k: int,
) -> list[RerankedDocument]:
    if not retrieved_documents:
        return []

    pairs = [(question, doc.document.text) for doc in retrieved_documents]
    scores: list[float] = []
    for batch in batches(pairs, batch_size):
        scores.extend(reranker.compute_scores(batch))

    indexed_scores = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
    reranked: list[RerankedDocument] = []
    for rank, (index, score) in enumerate(indexed_scores[:rerank_top_k], start=1):
        reranked.append(
            RerankedDocument(
                retrieved_document=retrieved_documents[index],
                rerank_score=float(score),
                rerank_rank=rank,
            )
        )
    return reranked


def reranked_document_to_record(doc: RerankedDocument, *, include_text: bool) -> dict[str, Any]:
    retrieved = doc.retrieved_document
    candidate = retrieved.document
    record: dict[str, Any] = {
        "rank": doc.rerank_rank,
        "rerank_rank": doc.rerank_rank,
        "rerank_score": doc.rerank_score,
        "retrieval_rank": retrieved.retrieval_rank,
        "retrieval_score": retrieved.retrieval_score,
        "path": candidate.rel_path,
        "filename": candidate.path.name,
        "truncated": candidate.truncated,
    }
    record.update({key: value for key, value in candidate.metadata.items() if value is not None})
    if include_text:
        record["text"] = candidate.text
    return record


def build_rerank_record(
    retrieval_result: RetrievalResult,
    reranked_documents: list[RerankedDocument],
    *,
    args: argparse.Namespace,
) -> dict[str, Any]:
    question_entry = retrieval_result.question_entry
    record: dict[str, Any] = {
        "question_id": question_entry["question_id"],
        "question": question_entry["question"],
        "num_candidate_docs": len(retrieval_result.documents),
        "retrieval_top_k": args.retrieval_top_k,
        "rerank_top_k": args.rerank_top_k,
        "retrieval_model": args.embedding_model_name_or_path,
        "reranker_model": args.reranker_model_name_or_path,
        "embedding_dim": args.embedding_dim,
        "normalized": bool(args.normalize),
        "documents": [
            reranked_document_to_record(doc, include_text=not args.no_text)
            for doc in reranked_documents
        ],
    }
    if retrieval_result.error:
        record["error"] = retrieval_result.error
    return record


def run_retrieval_and_reranking(args: argparse.Namespace) -> list[dict[str, Any]]:
    retrieval_results = collect_retrieval_candidates(args)

    reranker = Qwen3RerankerVllm(
        args.reranker_model_name_or_path,
        instruction=args.reranker_instruction,
        max_length=args.reranker_max_length,
        tensor_parallel_size=args.reranker_tensor_parallel_size,
        max_model_len=args.reranker_max_model_len,
        gpu_memory_utilization=args.reranker_gpu_memory_utilization,
        distributed_executor_backend=args.reranker_distributed_executor_backend,
    )

    records: list[dict[str, Any]] = []
    try:
        for result in tqdm(retrieval_results, desc="Reranking"):
            if result.error:
                records.append(build_rerank_record(result, [], args=args))
                continue
            reranked_documents = rerank_documents(
                reranker,
                result.question_entry["question"],
                result.retrieved_documents,
                batch_size=args.reranker_batch_size,
                rerank_top_k=args.rerank_top_k,
            )
            records.append(build_rerank_record(result, reranked_documents, args=args))
    finally:
        reranker.stop()

    write_jsonl(records, args.output_path)
    logger.info("Wrote %d records to %s", len(records), args.output_path)
    return records


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    run_retrieval_and_reranking(args)


if __name__ == "__main__":
    main()
