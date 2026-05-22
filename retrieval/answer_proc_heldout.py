from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

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
    from retrieval.retrieve_proc_heldout import DEFAULT_DOCS_DIR, DEFAULT_PRIVILEGED_DIR, write_jsonl
except ModuleNotFoundError as exc:
    if exc.name not in {"retrieval", "retrieval.retrieve_proc_heldout"}:
        raise
    from retrieve_proc_heldout import DEFAULT_DOCS_DIR, DEFAULT_PRIVILEGED_DIR, write_jsonl

try:
    from inference.collect_llm import (
        PROVIDER_DEFAULT,
        ReasoningEffort,
        create_async_client,
        generate_responses,
        get_provider_config,
    )
except ModuleNotFoundError as exc:
    if exc.name not in {"inference", "inference.collect_llm"}:
        raise
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from inference.collect_llm import (
        PROVIDER_DEFAULT,
        ReasoningEffort,
        create_async_client,
        generate_responses,
        get_provider_config,
    )


logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = REPO_ROOT / "retrieval" / "heldout_retrieval.jsonl"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "retrieval" / "heldout_answers_gemini.jsonl"
DEFAULT_LOCAL_MODEL = "Qwen/Qwen3.5-4B"

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

JUDGE_PROMPT_TEMPLATE = """You are given a question, sampled response, and gold answer.
Decide whether the sampled response is semantically equivalent to the gold answer.
Minor formatting, capitalization, punctuation, and unit-format differences are acceptable
when the answer means the same thing.

Return only valid JSON with this shape:
{{"correctness": 0 or 1}}

QUESTION:
{question}

SAMPLED RESPONSE:
{response}

GOLD ANSWER:
{gold_answer}
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


class LLMAnswerAgent:
    def __init__(
        self,
        provider: str,
        model_name_or_path: str = "",
        max_concurrent: int = 8,
        max_completion_tokens: int = 128,
        reasoning_effort: ReasoningEffort = PROVIDER_DEFAULT,
        use_cache: bool = True,
    ) -> None:
        self.provider = provider
        config = get_provider_config(provider)
        self.model_name_or_path = model_name_or_path or config.default_model
        self.client = create_async_client(provider)
        self.max_concurrent = max_concurrent
        self.max_completion_tokens = max_completion_tokens
        self.reasoning_effort = reasoning_effort
        self.use_cache = use_cache
        self.tensor_parallel_size = None

    def count_prompt_tokens(self, messages: list[dict[str, str]]) -> int:
        # A conservative approximation for API models; local vLLM uses exact tokens.
        return max(1, (len(render_messages_for_llm(messages)) + 3) // 4)

    def generate(
        self,
        prompts: list[AnswerPrompt],
        _sampling_params: SamplingParams | None = None,
    ) -> list[dict[str, Any]]:
        prompt_texts = [render_messages_for_llm(prompt.messages) for prompt in prompts]
        response_groups = asyncio.run(
            generate_responses(
                self.client,
                prompt_texts,
                N=1,
                max_completion_tokens=self.max_completion_tokens,
                reasoning_effort=self.reasoning_effort,
                use_cache=self.use_cache,
                max_concurrent=self.max_concurrent,
                model_name=self.model_name_or_path,
                provider=self.provider,
            )
        )

        results: list[dict[str, Any]] = []
        for response_group in response_groups:
            raw_response = response_group[0] if response_group else ""
            results.append(
                {
                    "response": clean_response(raw_response),
                    "raw_response": raw_response,
                    "generated_token_count": None,
                    "finish_reason": None,
                }
            )
        return results

    def stop(self) -> None:
        try:
            asyncio.run(self.client.close())
        except:
            pass


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
        self._coerce_vllm_tokenizer_max_token_id()

    def apply_chat_template(self, messages: list[dict[str, str]]) -> list[int]:
        try:
            tokenized = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=not self.disable_thinking,
            )
        except TypeError:
            tokenized = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
        return normalize_token_ids(tokenized)

    def _coerce_vllm_tokenizer_max_token_id(self) -> None:
        engine = getattr(self.llm, "llm_engine", None)
        input_processor = getattr(engine, "input_processor", None)
        tokenizer = getattr(input_processor, "tokenizer", None)
        if tokenizer is not None:
            coerce_tokenizer_max_token_id(tokenizer)

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


def normalize_token_ids(tokenized: Any) -> list[int]:
    if isinstance(tokenized, dict):
        if "input_ids" in tokenized:
            tokenized = tokenized["input_ids"]
        elif "prompt_token_ids" in tokenized:
            tokenized = tokenized["prompt_token_ids"]
        else:
            raise TypeError(f"Tokenizer output dict is missing input_ids: {sorted(tokenized)}")

    if hasattr(tokenized, "input_ids"):
        tokenized = tokenized.input_ids
    if hasattr(tokenized, "tolist"):
        tokenized = tokenized.tolist()

    if (
        isinstance(tokenized, (list, tuple))
        and len(tokenized) == 1
        and isinstance(tokenized[0], (list, tuple))
    ):
        tokenized = tokenized[0]

    if not isinstance(tokenized, (list, tuple)):
        raise TypeError(f"Expected token ids as a list, got {type(tokenized).__name__}")

    token_ids: list[int] = []
    for token_id in tokenized:
        if isinstance(token_id, bool):
            raise TypeError("Boolean token id is not valid")
        try:
            token_ids.append(int(token_id))
        except (TypeError, ValueError) as exc:
            raise TypeError(f"Token id is not int-convertible: {token_id!r}") from exc
    return token_ids


def coerce_tokenizer_max_token_id(tokenizer: Any) -> None:
    raw_max_token_id = getattr(tokenizer, "max_token_id", None)
    if isinstance(raw_max_token_id, int):
        return

    max_token_id = safe_int(raw_max_token_id)
    if max_token_id is None:
        vocab_getter = getattr(tokenizer, "get_vocab", None)
        if callable(vocab_getter):
            vocab_values = []
            for value in vocab_getter().values():
                parsed = safe_int(value)
                if parsed is not None:
                    vocab_values.append(parsed)
            if vocab_values:
                max_token_id = max(vocab_values)

    vocab_size = safe_int(getattr(tokenizer, "vocab_size", None))
    if vocab_size is not None:
        max_token_id = vocab_size if max_token_id is None else max(max_token_id, vocab_size)

    if max_token_id is None:
        return

    try:
        setattr(tokenizer, "max_token_id", max_token_id)
        return
    except (AttributeError, TypeError):
        pass

    try:
        setattr(tokenizer.__class__, "max_token_id", property(lambda _self, value=max_token_id: value))
    except (AttributeError, TypeError):
        logger.warning("Could not patch vLLM tokenizer max_token_id=%r", raw_max_token_id)


def load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key:
            os.environ.setdefault(key, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Answer heldout questions using retrieved or reranked documents."
    )
    parser.add_argument(
        "--non-local-answerer",
        "--non_local_answerer",
        action="store_true",
        default=False,
        dest="non_local_answerer",
        help="Use inference.collect_llm instead of local vLLM/Qwen.",
    )
    parser.add_argument('--provider', default='gemini')
    parser.add_argument(
        "--model-name-or-path",
        default="",
        help="Defaults to Qwen locally, or the selected provider's default for --non-local-answerer.",
    )
    parser.add_argument("--max-concurrent", type=int, default=8)
    parser.add_argument(
        "--answer-reasoning-effort",
        default="provider_default",
        help="For --non-local-answerer: use 'none', 'provider_default', or a provider-supported effort value.",
    )
    parser.add_argument(
        "--answer-use-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For --non-local-answerer: cache answer responses through inference.collect_llm.",
    )
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS_DIR)
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
    parser.add_argument('--n', type=int, default=1, help="Reserved for compatibility; one answer row is written per question.")
    parser.add_argument(
        "--privileged-dir",
        type=Path,
        default=DEFAULT_PRIVILEGED_DIR,
        help="Gold answer source for --judge; expects <question_id>/answer.txt.",
    )
    parser.add_argument(
        "--judge",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Gemini via inference.collect_llm to judge each generated answer against the gold answer.",
    )
    parser.add_argument("--judge-provider", default="gemini")
    parser.add_argument(
        "--judge-model-name",
        default="",
        help="Defaults to the selected provider's configured model.",
    )
    parser.add_argument("--judge-max-concurrent", type=int, default=4)
    parser.add_argument("--judge-max-completion-tokens", type=int, default=64)
    parser.add_argument(
        "--judge-reasoning-effort",
        default="none",
        help="Use 'none', 'provider_default', or a provider-supported effort value.",
    )
    parser.add_argument(
        "--judge-use-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cache judge responses through inference.collect_llm.",
    )
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
    if args.max_concurrent < 1:
        parser.error("--max-concurrent must be at least 1")
    if args.n < 1:
        parser.error("--n must be at least 1")
    if args.judge_max_concurrent < 1:
        parser.error("--judge-max-concurrent must be at least 1")
    if args.judge_max_completion_tokens < 1:
        parser.error("--judge-max-completion-tokens must be at least 1")
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


def resolve_reasoning_arg(value: str) -> ReasoningEffort:
    if value == "none":
        return None
    if value == "provider_default":
        return PROVIDER_DEFAULT
    return value


def read_gold_answer(record: dict[str, Any], privileged_dir: Path) -> str | None:
    for key in ("gold_answer", "answer"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    question_id = str(record.get("question_id", "")).strip()
    if not question_id:
        return None

    root = privileged_dir.resolve()
    answer_path = (root / question_id / "answer.txt").resolve()
    try:
        answer_path.relative_to(root)
    except ValueError:
        raise ValueError(f"Gold answer path escapes privileged dir for question id: {question_id}")

    if not answer_path.exists():
        return None
    return answer_path.read_text(encoding="utf-8", errors="replace").strip()


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


def render_messages_for_llm(messages: list[dict[str, str]]) -> str:
    rendered: list[str] = []
    for message in messages:
        role = str(message.get("role", "user")).upper()
        content = str(message.get("content", "")).strip()
        rendered.append(f"{role}:\n{content}")
    return "\n\n".join(rendered).strip()


def count_prompt_tokens(agent: Any, messages: list[dict[str, str]]) -> int:
    counter = getattr(agent, "count_prompt_tokens", None)
    if callable(counter):
        return int(counter(messages))
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
    agent: Any,
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


def parse_correctness(content: str) -> float:
    text = content.strip()
    parsed: Any | None = None

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))

    if isinstance(parsed, dict) and "correctness" in parsed:
        value = parsed["correctness"]
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, (int, float)):
            return 1.0 if value >= 0.5 else 0.0
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "correct", "yes"}:
                return 1.0
            if normalized in {"0", "false", "incorrect", "no"}:
                return 0.0

    match = re.search(r'"?correctness"?\s*[:=]\s*"?([01])"?', text, flags=re.IGNORECASE)
    if match:
        return float(match.group(1))

    normalized_text = text.lower()
    if normalized_text in {"1", "true", "correct", "yes"}:
        return 1.0
    if normalized_text in {"0", "false", "incorrect", "no"}:
        return 0.0

    raise ValueError(f"Could not parse correctness from judge response: {content!r}")


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


def resolve_answer_model(args: argparse.Namespace) -> str:
    if args.model_name_or_path:
        return args.model_name_or_path
    if args.non_local_answerer:
        return get_provider_config(args.provider).default_model
    return DEFAULT_LOCAL_MODEL


async def judge_answer_records(
    output_records: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    config = get_provider_config(args.judge_provider)
    judge_model = args.judge_model_name or config.default_model
    reasoning_effort = resolve_reasoning_arg(args.judge_reasoning_effort)

    prompts: list[str] = []
    prompt_indices: list[int] = []
    judged_records = [dict(record) for record in output_records]

    for index, record in enumerate(judged_records):
        record["judge_provider"] = config.name
        record["judge_model"] = judge_model
        record["correctness"] = None
        record["judge_response"] = ""
        record["judge_error"] = None

        try:
            gold_answer = read_gold_answer(record, args.privileged_dir)
        except Exception as exc:
            record["gold_answer"] = None
            record["judge_error"] = str(exc)
            continue

        record["gold_answer"] = gold_answer
        if not gold_answer:
            record["judge_error"] = "Missing gold answer"
            continue

        if record.get("error"):
            record["judge_error"] = f"Answer generation error: {record['error']}"
            continue

        response = str(record.get("response", "")).strip()
        if not response:
            record["correctness"] = 0.0
            record["judge_error"] = "Missing sampled response"
            continue

        prompts.append(
            JUDGE_PROMPT_TEMPLATE.format(
                question=str(record.get("question", "")).strip(),
                response=response,
                gold_answer=gold_answer,
            )
        )
        prompt_indices.append(index)

    if not prompts:
        return judged_records

    try:
        client = create_async_client(config)
    except Exception as exc:
        for index in prompt_indices:
            judged_records[index]["judge_error"] = f"Judge client error: {exc}"
        return judged_records

    try:
        try:
            responses = await generate_responses(
                client,
                prompts,
                N=1,
                max_completion_tokens=args.judge_max_completion_tokens,
                reasoning_effort=reasoning_effort,
                use_cache=args.judge_use_cache,
                max_concurrent=args.judge_max_concurrent,
                model_name=judge_model,
                provider=config,
            )
        except Exception as exc:
            for index in prompt_indices:
                judged_records[index]["judge_error"] = f"Judge request error: {exc}"
            return judged_records
    finally:
        await client.close()

    for index, response_group in zip(prompt_indices, responses):
        content = response_group[0] if response_group else ""
        judged_records[index]["judge_response"] = content
        try:
            judged_records[index]["correctness"] = parse_correctness(content)
        except Exception as exc:
            judged_records[index]["judge_error"] = str(exc)

    return judged_records


def answer_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    completed_ids = load_completed_question_ids(args.output_path) if args.resume else set()
    records = select_records(load_jsonl(args.input_path), args.query_id, args.limit, completed_ids)
    logger.info("Loaded %d records to answer", len(records))
    answer_model = resolve_answer_model(args)

    if args.non_local_answerer:
        agent = LLMAnswerAgent(
            args.provider,
            answer_model,
            args.max_concurrent,
            args.max_new_tokens,
            resolve_reasoning_arg(args.answer_reasoning_effort),
            args.answer_use_cache,
        )
    else:
        agent = QwenAnswerAgent(
            answer_model,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
            distributed_executor_backend=args.distributed_executor_backend,
            disable_thinking=args.disable_thinking,
        )
    sampling_params = None if args.non_local_answerer else build_sampling_params(args)

    output_records: list[dict[str, Any]] = []
    try:
        with tqdm(total=len(records), desc="Answering", unit="question") as progress:
            for batch in batches(records, args.batch_size):
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
                        "answer_model": answer_model,
                        "answer_provider": getattr(agent, "provider", "local"),
                        "tensor_parallel_size": getattr(agent, "tensor_parallel_size", None),
                    }
                    if args.judge:
                        for answer_key in ("gold_answer", "answer"):
                            answer_value = record.get(answer_key)
                            if isinstance(answer_value, str) and answer_value.strip():
                                base_output["gold_answer"] = answer_value.strip()
                                break
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
                progress.update(len(batch))
    finally:
        agent.stop()

    if args.judge:
        output_records = asyncio.run(judge_answer_records(output_records, args))

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
    load_env_file(REPO_ROOT / ".env")
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    answer_records(args)


if __name__ == "__main__":
    main()
