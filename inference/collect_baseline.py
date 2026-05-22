from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI, RateLimitError
from tqdm.asyncio import tqdm_asyncio

from inference.collect_llm import (
    PROVIDER_DEFAULT,
    ProviderLike,
    ReasoningEffort,
    _resolve_reasoning_effort,
    create_async_client,
    get_provider_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AGENT_DATA_DIR = REPO_ROOT / "data" / "train"
DEFAULT_PRIVILEGED_DATA_DIR = REPO_ROOT / "data" / "train_privileged"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "inference" / "single_layer_outputs.jsonl"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read one or more files from the provided filesystem hierarchy.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Relative file paths from the hierarchy, e.g. ['123/document.txt'] or ['document.txt'].",
                    }
                },
                "required": ["file_path"],
            },
        },
    }
]

SYSTEM_PROMPT = """You answer questions by inspecting a provided filesystem.

You will receive:
- a user query
- a filesystem hierarchy containing candidate documents

Rules:
1. Use only information from files you read with the read_file tool.
2. The hierarchy lists available files but does not itself prove facts.
3. Read files iteratively until you have enough evidence to answer.
4. Do not use outside knowledge.
5. Return only the final short answer, with no explanation or markdown.
"""

USER_PROMPT_TEMPLATE = """QUERY:
{query}

FILESYSTEM HIERARCHY:
{filesystem_hierarchy}

Use read_file to inspect relevant files, then answer the query with only the final short answer.
"""


@dataclass
class ToolRead:
    requested_paths: list[str]
    resolved_paths: list[str]
    errors: list[str] = field(default_factory=list)


@dataclass
class InferenceResult:
    query_id: str
    query: str
    response: str
    gold_answer: str | None
    tool_reads: list[ToolRead]
    sample_idx: int = 0
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run iterative file-reading inference over data/train query filesystems."
    )
    parser.add_argument("--agent-data-dir", type=Path, default=DEFAULT_AGENT_DATA_DIR)
    parser.add_argument("--privileged-data-dir", type=Path, default=DEFAULT_PRIVILEGED_DATA_DIR)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--N", type=int, default=4)  # number of independent traces to generate for each query
    parser.add_argument("--temperature", type=float, default=1.0)  # temperature to use for sampling (only relevant if provider supports temperature)
    parser.add_argument("--top_p", type=float, default=0.95)  # top p to use for sampling (only relevant if provider supports top p)
    parser.add_argument("--top_k", type=int, default=20)  # top k to use for sampling (only relevant if provider supports top k)
    parser.add_argument("--provider", default="openrouter")
    parser.add_argument("--model-name", default="qwen/qwen3.5-35b-a3b") 
    parser.add_argument("--max-concurrent", type=int, default=4)
    parser.add_argument("--max-problems", type=int, default=50)  # if greater than 0, only run this many problems
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-file-chars", type=int, default=120_000)
    parser.add_argument("--max-completion-tokens", type=int, default=32_768)
    parser.add_argument("--reasoning-effort", default="none")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--query-id",
        action="append",
        default=None,
        help="Run only this query id. Can be supplied multiple times.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip query id/sample pairs already present in the output JSONL.",
    )
    args = parser.parse_args()
    if args.N < 1:
        parser.error("--N must be at least 1")
    if args.max_concurrent < 1:
        parser.error("--max-concurrent must be at least 1")
    return args


def read_text_if_exists(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def query_sort_key(path: Path) -> tuple[int, str]:
    return (0, f"{int(path.name):012d}") if path.name.isdigit() else (1, path.name)


def discover_query_ids(agent_data_dir: Path, selected: list[str] | None, limit: int | None) -> list[str]:
    if selected:
        query_ids = selected
    else:
        query_ids = [p.name for p in sorted(agent_data_dir.iterdir(), key=query_sort_key) if p.is_dir()]
    return query_ids[:limit] if limit is not None else query_ids


def load_completed_samples(output_path: Path) -> dict[str, set[int]]:
    if not output_path.exists():
        return {}

    completed: dict[str, set[int]] = {}
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                query_id = str(record["query_id"])
                sample_idx = int(record.get("sample_idx", 0))
                completed.setdefault(query_id, set()).add(sample_idx)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return completed


def build_filesystem_hierarchy(query_dir: Path, display_root: str) -> str:
    files = sorted((p for p in query_dir.rglob("*") if p.is_file()), key=lambda p: p.relative_to(query_dir).as_posix())
    if not files:
        return f"{display_root}/\n  <no files>"

    lines = [f"{display_root}/"]
    for path in files:
        rel = path.relative_to(query_dir).as_posix()
        lines.append(f"  {rel}")
    return "\n".join(lines)


def resolve_agent_path(raw_path: str, query_id: str, query_dir: Path) -> Path | None:
    raw = raw_path.strip()
    if not raw:
        return None

    candidate = Path(raw)
    if candidate.is_absolute():
        resolved = candidate.resolve()
        try:
            resolved.relative_to(query_dir.resolve())
        except ValueError:
            return None
        return resolved

    parts = candidate.parts
    if parts and parts[0] == query_id:
        candidate = Path(*parts[1:]) if len(parts) > 1 else Path()

    resolved = (query_dir / candidate).resolve()
    try:
        resolved.relative_to(query_dir.resolve())
    except ValueError:
        return None
    return resolved


def read_requested_files(
    arguments: str,
    *,
    query_id: str,
    query_dir: Path,
    max_file_chars: int,
) -> tuple[str, ToolRead]:
    try:
        parsed = json.loads(arguments or "{}")
    except json.JSONDecodeError as exc:
        tool_read = ToolRead(requested_paths=[], resolved_paths=[], errors=[f"Invalid JSON arguments: {exc}"])
        return json.dumps({"error": tool_read.errors[0]}), tool_read

    requested = parsed.get("file_path", [])
    if isinstance(requested, str):
        requested = [requested]
    if not isinstance(requested, list):
        tool_read = ToolRead(requested_paths=[], resolved_paths=[], errors=["file_path must be a list of strings"])
        return json.dumps({"error": tool_read.errors[0]}), tool_read

    outputs: list[dict[str, Any]] = []
    resolved_paths: list[str] = []
    errors: list[str] = []

    for item in requested:
        raw_path = str(item)
        resolved = resolve_agent_path(raw_path, query_id, query_dir)
        if resolved is None:
            error = f"Path is outside the allowed query filesystem: {raw_path}"
            errors.append(error)
            outputs.append({"path": raw_path, "error": error})
            continue
        if not resolved.exists() or not resolved.is_file():
            error = f"File not found: {raw_path}"
            errors.append(error)
            outputs.append({"path": raw_path, "error": error})
            continue

        text = resolved.read_text(encoding="utf-8", errors="replace")
        truncated = len(text) > max_file_chars
        if truncated:
            text = text[:max_file_chars]
        rel = resolved.relative_to(query_dir).as_posix()
        resolved_paths.append(rel)
        outputs.append({"path": rel, "truncated": truncated, "content": text})

    tool_read = ToolRead(requested_paths=[str(x) for x in requested], resolved_paths=resolved_paths, errors=errors)
    return json.dumps({"files": outputs}, ensure_ascii=False), tool_read


def build_request_kwargs(
    *,
    provider: ProviderLike,
    messages: list[dict[str, Any]],
    model_name: str,
    max_completion_tokens: int,
    reasoning_effort: ReasoningEffort,
    temperature: float,
    top_p: float,
    top_k: int,
    include_tools: bool,
) -> dict[str, Any]:
    config = get_provider_config(provider)
    resolved_reasoning_effort = _resolve_reasoning_effort(config, reasoning_effort)
    kwargs: dict[str, Any] = {
        "model": model_name or config.default_model,
        "messages": messages,
        config.max_tokens_param: max_completion_tokens,
    }

    if temperature >= 0:
        kwargs["temperature"] = temperature
    if top_p > 0:
        kwargs["top_p"] = top_p
    if top_k > 0 and config.name != "openai":
        kwargs.setdefault("extra_body", {})["top_k"] = top_k

    if include_tools:
        kwargs["tools"] = TOOLS
        kwargs["tool_choice"] = "auto"

    if resolved_reasoning_effort is not None:
        if config.reasoning_mode == "request_param":
            kwargs["reasoning_effort"] = resolved_reasoning_effort
        elif config.reasoning_mode == "openrouter_extra_body":
            kwargs.setdefault("extra_body", {})["reasoning"] = {"effort": resolved_reasoning_effort}

    return kwargs


async def create_completion_with_retries(
    client: AsyncOpenAI,
    *,
    provider: ProviderLike,
    messages: list[dict[str, Any]],
    model_name: str,
    max_completion_tokens: int,
    reasoning_effort: ReasoningEffort,
    temperature: float,
    top_p: float,
    top_k: int,
    include_tools: bool,
    max_retries: int = 10,
) -> Any:
    for attempt in range(max_retries):
        try:
            return await client.chat.completions.create(
                **build_request_kwargs(
                    provider=provider,
                    messages=messages,
                    model_name=model_name,
                    max_completion_tokens=max_completion_tokens,
                    reasoning_effort=reasoning_effort,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    include_tools=include_tools,
                )
            )
        except RateLimitError:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(min(60, 2 ** (attempt + 1)))
        except (APIConnectionError, APITimeoutError, APIStatusError):
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(min(30, 2**attempt))

    raise RuntimeError("unreachable retry state")


async def run_query(
    client: AsyncOpenAI,
    *,
    query_id: str,
    sample_idx: int,
    agent_data_dir: Path,
    privileged_data_dir: Path,
    provider: ProviderLike,
    model_name: str,
    max_steps: int,
    max_file_chars: int,
    max_completion_tokens: int,
    reasoning_effort: ReasoningEffort,
    temperature: float,
    top_p: float,
    top_k: int,
    semaphore: asyncio.Semaphore,
) -> InferenceResult:
    query_dir = agent_data_dir / query_id
    privileged_query_dir = privileged_data_dir / query_id
    query = read_text_if_exists(privileged_query_dir / "query.txt")
    gold_answer = read_text_if_exists(privileged_query_dir / "answer.txt")

    if query is None:
        return InferenceResult(
            query_id=query_id,
            query="",
            response="",
            gold_answer=gold_answer,
            tool_reads=[],
            sample_idx=sample_idx,
            error="Missing query.txt",
        )
    if not query_dir.exists():
        return InferenceResult(
            query_id=query_id,
            query=query,
            response="",
            gold_answer=gold_answer,
            tool_reads=[],
            sample_idx=sample_idx,
            error="Missing agent data directory",
        )

    filesystem_hierarchy = build_filesystem_hierarchy(query_dir, query_id)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(
                query=query.strip(),
                filesystem_hierarchy=filesystem_hierarchy,
            ),
        },
    ]
    tool_reads: list[ToolRead] = []

    async with semaphore:
        try:
            for _ in range(max_steps):
                completion = await create_completion_with_retries(
                    client,
                    provider=provider,
                    messages=messages,
                    model_name=model_name,
                    max_completion_tokens=max_completion_tokens,
                    reasoning_effort=reasoning_effort,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    include_tools=True,
                )
                message = completion.choices[0].message
                messages.append(message.model_dump(exclude_none=True))

                tool_calls = message.tool_calls or []
                if not tool_calls:
                    return InferenceResult(
                        query_id=query_id,
                        query=query.strip(),
                        response=(message.content or "").strip(),
                        gold_answer=gold_answer.strip() if gold_answer is not None else None,
                        tool_reads=tool_reads,
                        sample_idx=sample_idx,
                    )

                for tool_call in tool_calls:
                    if tool_call.function.name != "read_file":
                        content = json.dumps({"error": f"Unknown tool: {tool_call.function.name}"})
                    else:
                        content, tool_read = read_requested_files(
                            tool_call.function.arguments,
                            query_id=query_id,
                            query_dir=query_dir,
                            max_file_chars=max_file_chars,
                        )
                        tool_reads.append(tool_read)
                    messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": content})

            completion = await create_completion_with_retries(
                client,
                provider=provider,
                messages=messages
                + [
                    {
                        "role": "user",
                        "content": "You have reached the file-read step limit. Answer now with the best-supported short answer.",
                    }
                ],
                model_name=model_name,
                max_completion_tokens=max_completion_tokens,
                reasoning_effort=reasoning_effort,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                include_tools=False,
            )
            response = completion.choices[0].message.content or ""
            return InferenceResult(
                query_id=query_id,
                query=query.strip(),
                response=response.strip(),
                gold_answer=gold_answer.strip() if gold_answer is not None else None,
                tool_reads=tool_reads,
                sample_idx=sample_idx,
                error="max_steps_reached",
            )
        except Exception as exc:
            return InferenceResult(
                query_id=query_id,
                query=query.strip(),
                response="",
                gold_answer=gold_answer.strip() if gold_answer is not None else None,
                tool_reads=tool_reads,
                sample_idx=sample_idx,
                error=str(exc),
            )


async def run_all(args: argparse.Namespace) -> list[InferenceResult]:
    limit = args.limit
    if args.max_problems > 0:
        limit = args.max_problems if limit is None else min(limit, args.max_problems)

    query_ids = discover_query_ids(args.agent_data_dir, args.query_id, limit)
    sample_jobs = [(query_id, sample_idx) for query_id in query_ids for sample_idx in range(args.N)]
    if args.resume:
        completed = load_completed_samples(args.output_path)
        sample_jobs = [
            (query_id, sample_idx)
            for query_id, sample_idx in sample_jobs
            if sample_idx not in completed.get(query_id, set())
        ]

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    client = create_async_client(args.provider)
    semaphore = asyncio.Semaphore(args.max_concurrent)
    reasoning_effort: ReasoningEffort = None if args.reasoning_effort == "none" else args.reasoning_effort
    if args.reasoning_effort == "provider_default":
        reasoning_effort = PROVIDER_DEFAULT

    tasks = [
        run_query(
            client,
            query_id=query_id,
            sample_idx=sample_idx,
            agent_data_dir=args.agent_data_dir,
            privileged_data_dir=args.privileged_data_dir,
            provider=args.provider,
            model_name=args.model_name,
            max_steps=args.max_steps,
            max_file_chars=args.max_file_chars,
            max_completion_tokens=args.max_completion_tokens,
            reasoning_effort=reasoning_effort,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            semaphore=semaphore,
        )
        for query_id, sample_idx in sample_jobs
    ]
    results = await tqdm_asyncio.gather(*tasks, desc="Running file-reading inference") if tasks else []

    mode = "a" if args.resume else "w"
    with args.output_path.open(mode, encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")

    return results


def main() -> None:
    args = parse_args()
    results = asyncio.run(run_all(args))
    errors = sum(1 for result in results if result.error)
    print(f"Wrote {len(results)} results to {args.output_path} ({errors} with errors).")


if __name__ == "__main__":
    main()
