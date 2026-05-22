from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel
from vllm.inputs.data import TokensPrompt

try:
    from retrieval.retrieve_proc_heldout import DEFAULT_DOCS_DIR, write_jsonl
except ModuleNotFoundError as exc:
    if exc.name not in {"retrieval", "retrieval.retrieve_proc_heldout"}:
        raise
    from retrieve_proc_heldout import DEFAULT_DOCS_DIR, write_jsonl


logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = REPO_ROOT / "retrieval" / "heldout_retrieval.jsonl"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "retrieval" / "heldout_answers.jsonl"

SYSTEM_PROMPT = """You answer questions using only the retrieved documents provided.

Rules:
1. Some retrieved documents may be irrelevant or incomplete.
2. Use only facts supported by the documents.
3. If the documents do not support an answer, return "Unknown".
4. Return only the final short answer, with no explanation or markdown.
"""

USER_PROMPT_TEMPLATE = """QUESTION:
{question}

RETRIEVED DOCUMENTS:
{documents}

Answer the question with only the final short answer.
"""


@dataclass(frozen=True)
class PromptDocument:
    path: str
    rank: int | None
    retrieval_rank: int | None
    rerank_rank: int | None
    retrieval_score: float | None
    rerank_score: float | None
    text: str
    truncated_for_prompt: bool


@dataclass(frozen=True)
class AnswerPrompt:
    messages: list[dict[str, str]]
    prompt_documents: list[PromptDocument]
    prompt_token_count: int


class QwenAnswerAgent:
    def __init__(
        self,
        model_name_or_path: str,
        *,
        tensor_parallel_size: int | None,
        max_model_len: int,
        gpu_memory_utilization: float,
        dtype: str,
        trust_remote_code: bool,
        distributed_executor_backend: str | None,
        disable_thinking: bool,
    ) -> None:
        if tensor_parallel_size is None:
            tensor_parallel_size = max(torch.cuda.device_count(), 1)

        self.tensor_parallel_size = tensor_parallel_size
        self.disable_thinking = disable_thinking
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )

        llm_kwargs: dict[str, Any] = {
            "model": model_name_or_path,
            "tensor_parallel_size": tensor_parallel_size,
            "max_model_len": max_model_len,
            "gpu_memory_utilization": gpu_memory_utilization,
            "dtype": dtype,
            "trust_remote_code": trust_remote_code,
        }
        if distributed_executor_backend:
            llm_kwargs["distributed_executor_backend"] = distributed_executor_backend

        self.llm = LLM(**llm_kwargs)

    def apply_chat_template(self, messages: list[dict[str, str]]) -> list[int]:
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=not self.disable_thinking,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )

    def generate(
        self,
        prompts: list[AnswerPrompt],
        sampling_params: SamplingParams,
    ) -> list[dict[str, Any]]:
        token_prompts = [
            TokensPrompt(prompt_token_ids=self.apply_chat_template(prompt.messages))
            for prompt in prompts
        ]
        outputs = self.llm.generate(token_prompts, sampling_params, use_tqdm=False)

        results: list[dict[str, Any]] = []
        for output in outputs:
            generation = output.outputs[0]
            results.append(
                {
                    "response": clean_response(generation.text),
                    "raw_response": generation.text,
                    "generated_token_count": len(generation.token_ids),
                    "finish_reason": generation.finish_reason,
                }
            )
        return results

    def stop(self) -> None:
        destroy_model_parallel()
        destroy_distributed_environment()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Answer heldout questions using retrieved or reranked documents."
    )
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS_DIR)
    parser.add_argument("--model-name-or-path", default="Qwen/Qwen3.5-4B")
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=None,
        help="Defaults to all visible CUDA devices. Use 4 for a 4x H200 tensor-parallel run.",
    )
    parser.add_argument("--distributed-executor-backend", default=None)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-prompt-tokens", type=int, default=30000)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--top-docs", type=int, default=10)
    parser.add_argument("--max-doc-chars", type=int, default=12_000)
    parser.add_argument("--min-doc-chars", type=int, default=800)
    parser.add_argument(
        "--disable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass enable_thinking=False to Qwen chat templates when supported.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--query-id",
        action="append",
        default=None,
        help="Only process this question id. Can be supplied multiple times.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to output and skip question ids already present there.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.top_docs < 1:
        parser.error("--top-docs must be at least 1")
    if args.max_doc_chars < 1:
        parser.error("--max-doc-chars must be at least 1")
    if args.min_doc_chars < 0:
        parser.error("--min-doc-chars must be >= 0")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be at least 1")
    if args.max_model_len <= args.max_new_tokens:
        parser.error("--max-model-len must be greater than --max-new-tokens")
    if args.max_prompt_tokens < 1:
        parser.error("--max-prompt-tokens must be at least 1")
    if args.tensor_parallel_size is not None and args.tensor_parallel_size < 1:
        parser.error("--tensor-parallel-size must be at least 1")
    return args


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
    return records


def load_completed_question_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()

    completed: set[str] = set()
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if "question_id" in record:
                    completed.add(str(record["question_id"]))
            except json.JSONDecodeError:
                continue
    return completed


def select_records(
    records: list[dict[str, Any]],
    query_ids: list[str] | None,
    limit: int | None,
    completed_ids: set[str],
) -> list[dict[str, Any]]:
    if query_ids:
        selected = {str(query_id) for query_id in query_ids}
        records = [record for record in records if str(record.get("question_id")) in selected]
    if completed_ids:
        records = [record for record in records if str(record.get("question_id")) not in completed_ids]
    if limit is not None:
        records = records[:limit]
    return records


def batches(items: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def doc_sort_key(doc: dict[str, Any]) -> tuple[int, int]:
    rank = doc.get("rank", doc.get("rerank_rank", doc.get("retrieval_rank", 10**9)))
    retrieval_rank = doc.get("retrieval_rank", 10**9)
    return safe_int(rank, 10**9), safe_int(retrieval_rank, 10**9)


def safe_int(value: Any, default: int | None = None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_doc_text(doc: dict[str, Any], docs_dir: Path) -> str:
    text = doc.get("text")
    if isinstance(text, str):
        return text

    rel_path = doc.get("path")
    if not rel_path:
        return ""
    path = (docs_dir / str(rel_path)).resolve()
    try:
        path.relative_to(docs_dir.resolve())
    except ValueError:
        raise ValueError(f"Retrieved document path escapes docs dir: {rel_path}")
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Retrieved document text missing and file not found: {path}")
    return path.read_text(encoding="utf-8", errors="replace")


def render_document(doc: PromptDocument) -> str:
    metadata = [f"path: {doc.path}"]
    if doc.rerank_rank is not None:
        metadata.append(f"rerank_rank: {doc.rerank_rank}")
    if doc.rerank_score is not None:
        metadata.append(f"rerank_score: {doc.rerank_score:.6f}")
    if doc.retrieval_rank is not None:
        metadata.append(f"retrieval_rank: {doc.retrieval_rank}")
    if doc.retrieval_score is not None:
        metadata.append(f"retrieval_score: {doc.retrieval_score:.6f}")
    if doc.truncated_for_prompt:
        metadata.append("truncated_for_prompt: true")
    return "\n".join(metadata) + "\ncontent:\n" + doc.text.strip()


def render_documents(docs: list[PromptDocument]) -> str:
    return "\n\n".join(
        f"[Document {index}]\n{render_document(doc)}"
        for index, doc in enumerate(docs, start=1)
    )


def build_messages(question: str, docs: list[PromptDocument]) -> list[dict[str, str]]:
    documents_text = render_documents(docs) if docs else "<no retrieved documents>"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(
                question=question.strip(),
                documents=documents_text,
            ),
        },
    ]


def count_prompt_tokens(agent: QwenAnswerAgent, messages: list[dict[str, str]]) -> int:
    return len(agent.apply_chat_template(messages))


def prompt_budget(args: argparse.Namespace) -> int:
    model_budget = args.max_model_len - args.max_new_tokens - 16
    return max(1, min(args.max_prompt_tokens, model_budget))


def make_prompt_document(raw_doc: dict[str, Any], text: str, truncated_for_prompt: bool) -> PromptDocument:
    return PromptDocument(
        path=str(raw_doc.get("path", raw_doc.get("filename", ""))),
        rank=safe_int(raw_doc.get("rank")),
        retrieval_rank=safe_int(raw_doc.get("retrieval_rank")),
        rerank_rank=safe_int(raw_doc.get("rerank_rank")),
        retrieval_score=safe_float(raw_doc.get("retrieval_score")),
        rerank_score=safe_float(raw_doc.get("rerank_score")),
        text=text,
        truncated_for_prompt=truncated_for_prompt,
    )


def fit_prompt(
    agent: QwenAnswerAgent,
    record: dict[str, Any],
    *,
    args: argparse.Namespace,
) -> AnswerPrompt:
    question = str(record.get("question", ""))
    budget = prompt_budget(args)
    raw_documents = sorted(record.get("documents", []), key=doc_sort_key)[: args.top_docs]

    prompt_documents: list[PromptDocument] = []
    for raw_doc in raw_documents:
        full_text = load_doc_text(raw_doc, args.docs_dir)
        clipped_text = full_text[: args.max_doc_chars]
        full_prompt_doc = make_prompt_document(
            raw_doc,
            clipped_text,
            truncated_for_prompt=len(full_text) > len(clipped_text),
        )

        candidate_docs = prompt_documents + [full_prompt_doc]
        candidate_messages = build_messages(question, candidate_docs)
        if count_prompt_tokens(agent, candidate_messages) <= budget:
            prompt_documents.append(full_prompt_doc)
            continue

        low = 0
        high = len(clipped_text)
        best_text = ""
        while low <= high:
            mid = (low + high) // 2
            test_text = clipped_text[:mid]
            test_doc = make_prompt_document(raw_doc, test_text, truncated_for_prompt=True)
            test_messages = build_messages(question, prompt_documents + [test_doc])
            if count_prompt_tokens(agent, test_messages) <= budget:
                best_text = test_text
                low = mid + 1
            else:
                high = mid - 1

        if len(best_text.strip()) >= args.min_doc_chars:
            prompt_documents.append(make_prompt_document(raw_doc, best_text, truncated_for_prompt=True))
        break

    messages = build_messages(question, prompt_documents)
    return AnswerPrompt(
        messages=messages,
        prompt_documents=prompt_documents,
        prompt_token_count=count_prompt_tokens(agent, messages),
    )


def clean_response(text: str) -> str:
    cleaned = text.strip()
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[1].strip()
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()
    return cleaned


def prompt_doc_to_record(doc: PromptDocument) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": doc.path,
        "truncated_for_prompt": doc.truncated_for_prompt,
    }
    for key in ("rank", "retrieval_rank", "rerank_rank", "retrieval_score", "rerank_score"):
        value = getattr(doc, key)
        if value is not None:
            record[key] = value
    return record


def build_sampling_params(args: argparse.Namespace) -> SamplingParams:
    kwargs: dict[str, Any] = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_new_tokens,
    }
    if args.top_k > 0:
        kwargs["top_k"] = args.top_k
    return SamplingParams(**kwargs)


def answer_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    completed_ids = load_completed_question_ids(args.output_path) if args.resume else set()
    records = select_records(load_jsonl(args.input_path), args.query_id, args.limit, completed_ids)
    logger.info("Loaded %d records to answer", len(records))

    agent = QwenAnswerAgent(
        args.model_name_or_path,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        distributed_executor_backend=args.distributed_executor_backend,
        disable_thinking=args.disable_thinking,
    )
    sampling_params = build_sampling_params(args)

    output_records: list[dict[str, Any]] = []
    try:
        for batch in tqdm(list(batches(records, args.batch_size)), desc="Answering"):
            prompt_records: list[tuple[dict[str, Any], AnswerPrompt | None, str | None]] = []
            prompts: list[AnswerPrompt] = []

            for record in batch:
                try:
                    prompt = fit_prompt(agent, record, args=args)
                    prompt_records.append((record, prompt, None))
                    prompts.append(prompt)
                except Exception as exc:
                    prompt_records.append((record, None, str(exc)))

            generations = agent.generate(prompts, sampling_params) if prompts else []
            generation_iter = iter(generations)

            for record, prompt, error in prompt_records:
                base_output = {
                    "question_id": str(record.get("question_id")),
                    "question": str(record.get("question", "")),
                    "input_path": str(args.input_path),
                    "answer_model": args.model_name_or_path,
                    "tensor_parallel_size": agent.tensor_parallel_size,
                }
                if error is not None or prompt is None:
                    output_records.append({**base_output, "response": "", "error": error})
                    continue

                generation = next(generation_iter)
                output_records.append(
                    {
                        **base_output,
                        **generation,
                        "num_retrieved_documents": len(record.get("documents", [])),
                        "num_prompt_documents": len(prompt.prompt_documents),
                        "prompt_token_count": prompt.prompt_token_count,
                        "prompt_documents": [
                            prompt_doc_to_record(doc)
                            for doc in prompt.prompt_documents
                        ],
                    }
                )
    finally:
        agent.stop()

    if args.resume and args.output_path.exists():
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        with args.output_path.open("a", encoding="utf-8") as f:
            for record in output_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    else:
        write_jsonl(output_records, args.output_path)

    logger.info("Wrote %d answer records to %s", len(output_records), args.output_path)
    return output_records


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    answer_records(args)


if __name__ == "__main__":
    main()
