from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
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
DEFAULT_INPUT_PATH = REPO_ROOT / "inference" / "single_layer_outputs.jsonl"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "inference" / "single_layer_outputs_with_correctness.jsonl"

SIMPLE_JUDGE_PROMPT = """You are given a query, sampled response, and gold answer.
Decide whether the sampled response is semantically equivalent to the gold answer.
Minor formatting differences are acceptable if the answer means the same thing.

Return only valid JSON with this shape:
{{"correctness": 0 or 1}}

QUERY:
{query}

SAMPLED RESPONSE:
{response}

GOLD ANSWER:
{gold_answer}
"""

REQUIRED_COLUMNS = {"query", "response", "gold_answer"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Judge response correctness for inference JSONL outputs.")
    parser.add_argument("--provider", default="gemini")
    parser.add_argument("--model-name", default="", help="Defaults to the selected provider's configured model.")
    parser.add_argument("--path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--max-concurrent", type=int, default=4)
    parser.add_argument("--max-completion-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--reasoning-effort",
        default="none",
        help="Use 'none', 'provider_default', or a provider-supported effort value.",
    )
    args = parser.parse_args()
    if args.max_concurrent < 1:
        parser.error("--max-concurrent must be at least 1")
    if args.max_retries < 1:
        parser.error("--max-retries must be at least 1")
    return args


def resolve_reasoning_arg(value: str) -> ReasoningEffort:
    if value == "none":
        return None
    if value == "provider_default":
        return PROVIDER_DEFAULT
    return value


def row_text(row: pd.Series, column: str) -> str:
    value = row[column]
    if pd.isna(value):
        return ""
    return str(value).strip()


def build_judge_request_kwargs(
    *,
    provider: ProviderLike,
    model_name: str,
    prompt: str,
    max_completion_tokens: int,
    reasoning_effort: ReasoningEffort,
    temperature: float,
) -> dict[str, Any]:
    config = get_provider_config(provider)
    resolved_reasoning_effort = _resolve_reasoning_effort(config, reasoning_effort)
    kwargs: dict[str, Any] = {
        "model": model_name or config.default_model,
        "messages": [{"role": "user", "content": prompt}],
        config.max_tokens_param: max_completion_tokens,
        "temperature": temperature,
    }

    if resolved_reasoning_effort is not None:
        if config.reasoning_mode == "request_param":
            kwargs["reasoning_effort"] = resolved_reasoning_effort
        elif config.reasoning_mode == "openrouter_extra_body":
            kwargs["extra_body"] = {"reasoning": {"effort": resolved_reasoning_effort}}

    return kwargs


async def create_judge_completion_with_retries(
    client: AsyncOpenAI,
    *,
    provider: ProviderLike,
    model_name: str,
    prompt: str,
    max_completion_tokens: int,
    reasoning_effort: ReasoningEffort,
    temperature: float,
    max_retries: int,
) -> Any:
    for attempt in range(max_retries):
        try:
            return await client.chat.completions.create(
                **build_judge_request_kwargs(
                    provider=provider,
                    model_name=model_name,
                    prompt=prompt,
                    max_completion_tokens=max_completion_tokens,
                    reasoning_effort=reasoning_effort,
                    temperature=temperature,
                )
            )
        except RateLimitError:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(min(60, 2 ** (attempt + 1)))
        except (APIConnectionError, APITimeoutError):
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(min(30, 2**attempt))
        except APIStatusError as exc:
            if exc.status_code < 500 or attempt == max_retries - 1:
                raise
            await asyncio.sleep(min(30, 2**attempt))

    raise RuntimeError("unreachable retry state")


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


async def evaluate_single_example(
    row: pd.Series,
    client: AsyncOpenAI,
    *,
    provider: ProviderLike,
    model_name: str,
    max_completion_tokens: int,
    reasoning_effort: ReasoningEffort,
    temperature: float,
    max_retries: int,
) -> tuple[float | None, str, str | None]:
    prompt = SIMPLE_JUDGE_PROMPT.format(
        query=row_text(row, "query"),
        response=row_text(row, "response"),
        gold_answer=row_text(row, "gold_answer"),
    )

    try:
        completion = await create_judge_completion_with_retries(
            client,
            provider=provider,
            model_name=model_name,
            prompt=prompt,
            max_completion_tokens=max_completion_tokens,
            reasoning_effort=reasoning_effort,
            temperature=temperature,
            max_retries=max_retries,
        )
        content = completion.choices[0].message.content or ""
        return parse_correctness(content), content, None
    except Exception as exc:
        return None, "", str(exc)


async def evaluate_dataset(
    df: pd.DataFrame,
    client: AsyncOpenAI,
    *,
    provider: ProviderLike,
    model_name: str,
    max_concurrent: int,
    max_completion_tokens: int,
    reasoning_effort: ReasoningEffort,
    temperature: float,
    max_retries: int,
) -> pd.DataFrame:
    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Input JSONL is missing required columns: {missing}")

    df_c = df.copy()
    df_c["correctness"] = pd.NA
    df_c["judge_response"] = ""
    df_c["judge_error"] = pd.NA

    semaphore = asyncio.Semaphore(max_concurrent)

    async def evaluate_row(index: Any, row: pd.Series) -> tuple[Any, float | None, str, str | None]:
        async with semaphore:
            correctness, judge_response, judge_error = await evaluate_single_example(
                row,
                client,
                provider=provider,
                model_name=model_name,
                max_completion_tokens=max_completion_tokens,
                reasoning_effort=reasoning_effort,
                temperature=temperature,
                max_retries=max_retries,
            )
        return index, correctness, judge_response, judge_error

    tasks = [evaluate_row(index, row) for index, row in df.iterrows()]
    results = await tqdm_asyncio.gather(*tasks, desc="Judging responses") if tasks else []

    for index, correctness, judge_response, judge_error in results:
        df_c.loc[index, "correctness"] = correctness
        df_c.loc[index, "judge_response"] = judge_response
        df_c.loc[index, "judge_error"] = judge_error

    return df_c


async def main_async(args: argparse.Namespace) -> None:
    df = pd.read_json(args.path, lines=True)
    if args.limit is not None:
        df = df.head(args.limit)

    client = create_async_client(args.provider)
    df = await evaluate_dataset(
        df,
        client,
        provider=args.provider,
        model_name=args.model_name,
        max_concurrent=args.max_concurrent,
        max_completion_tokens=args.max_completion_tokens,
        reasoning_effort=resolve_reasoning_arg(args.reasoning_effort),
        temperature=args.temperature,
        max_retries=args.max_retries,
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(args.output_path, orient="records", lines=True, force_ascii=False)

    judged = pd.to_numeric(df["correctness"], errors="coerce").dropna()
    errors = df["judge_error"].notna().sum()
    if len(judged):
        print(f"Accuracy: {judged.mean():.3f} ({int(judged.sum())}/{len(judged)})")
    print(f"Wrote {len(df)} rows to {args.output_path} ({errors} judge errors).")


def main() -> None:
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
