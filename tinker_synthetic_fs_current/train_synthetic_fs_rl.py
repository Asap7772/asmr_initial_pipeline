from __future__ import annotations

import asyncio
import gc
import json
import logging
import os
import pickle
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import chz
import torch

from synthetic_fs_env import SyntheticFilesystemDatasetBuilder
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.rl import train
from tinker_cookbook.utils.misc_utils import safezip

logger = logging.getLogger(__name__)


def _known_model_context_window_tokens(model_name: str) -> int | None:
    normalized = model_name.strip().lower()
    if normalized == "qwen/qwen3.5-4b":
        return 65536
    return None


def _effective_model_context_window_tokens(
    *,
    model_name: str,
    configured_context_window_tokens: int | None,
) -> int | None:
    known_context_window_tokens = _known_model_context_window_tokens(model_name)
    if known_context_window_tokens is None:
        return configured_context_window_tokens
    if configured_context_window_tokens is None:
        logger.info(
            "Using known context window for %s: %d tokens",
            model_name,
            known_context_window_tokens,
        )
        return known_context_window_tokens
    if configured_context_window_tokens > known_context_window_tokens:
        logger.warning(
            "Configured model_context_window_tokens=%d exceeds the known served context "
            "window for %s (%d); using %d instead.",
            configured_context_window_tokens,
            model_name,
            known_context_window_tokens,
            known_context_window_tokens,
        )
        return known_context_window_tokens
    return configured_context_window_tokens


def _install_sampler_context_window_patch(context_window_tokens: int | None) -> None:
    if context_window_tokens is None or context_window_tokens <= 0:
        return

    from tinker_cookbook.rl import rollouts

    original_completer = getattr(
        rollouts,
        "_synthetic_fs_original_tinker_token_completer",
        None,
    )
    if original_completer is None:
        original_completer = rollouts.TinkerTokenCompleter
        rollouts._synthetic_fs_original_tinker_token_completer = original_completer

    def context_window_tinker_token_completer(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("context_window", context_window_tokens)
        return original_completer(*args, **kwargs)

    rollouts.TinkerTokenCompleter = context_window_tinker_token_completer
    rollouts._synthetic_fs_sampler_context_window_tokens = context_window_tokens
    logger.info("Installed Tinker sampler context-window cap: %d tokens", context_window_tokens)


def _install_rl_diagnostics_patch(
    *,
    ppo_clip_low_threshold: float,
    ppo_clip_high_threshold: float,
) -> None:
    """Add local RL diagnostics without changing Tinker's optimizer code."""
    if getattr(train, "_synthetic_fs_rl_diagnostics_patch", False):
        return

    original_prepare_minibatch = train.prepare_minibatch
    original_compute_full_batch = train.compute_full_batch_metrics_and_get_sampling_client

    async def prepare_minibatch_with_diagnostics(*args: Any, **kwargs: Any):
        data_D, metrics = await original_prepare_minibatch(*args, **kwargs)
        trajectory_groups_P = args[1] if len(args) > 1 else kwargs.get("trajectory_groups_P", [])
        advantage_chunks = []
        action_token_counts = []
        target_token_counts = []
        for datum in data_D:
            advantages = datum.loss_fn_inputs["advantages"].to_torch()
            mask = datum.loss_fn_inputs["mask"].to_torch() > 0
            if mask.any():
                action_advantages = advantages[mask].float()
                advantage_chunks.append(action_advantages)
                action_token_counts.append(float(mask.sum().item()))
            target_token_counts.append(float(len(datum.loss_fn_inputs["target_tokens"].data)))
        if advantage_chunks:
            flat_advantages = torch.cat(advantage_chunks)
            metrics.update(
                {
                    "optim/advantage_mean": float(flat_advantages.mean().item()),
                    "optim/advantage_std": float(flat_advantages.std(unbiased=False).item()),
                    "optim/advantage_min": float(flat_advantages.min().item()),
                    "optim/advantage_max": float(flat_advantages.max().item()),
                    "optim/advantage_abs_mean": float(flat_advantages.abs().mean().item()),
                    "optim/zero_advantage_ratio": float(
                        (flat_advantages.abs() <= 1e-12).float().mean().item()
                    ),
                }
            )
        if action_token_counts:
            metrics["optim/action_tokens_max"] = max(action_token_counts)
            metrics["optim/action_tokens_mean"] = sum(action_token_counts) / len(action_token_counts)
        if target_token_counts:
            metrics["optim/train_datum_tokens_max"] = max(target_token_counts)
            metrics["optim/train_datum_tokens_mean"] = sum(target_token_counts) / len(
                target_token_counts
            )
        trajectory_lengths = [
            float(len(traj.transitions))
            for group in trajectory_groups_P
            for traj in group.trajectories_G
        ]
        if trajectory_lengths:
            metrics["optim/trajectory_turns_max"] = max(trajectory_lengths)
            metrics["optim/trajectory_turns_mean"] = sum(trajectory_lengths) / len(
                trajectory_lengths
            )
        return data_D, metrics

    async def compute_full_batch_with_diagnostics(
        training_client,
        checkpoint_mgr,
        i_batch,
        data_D,
        training_logprobs_D,
        do_compute_post_kl,
    ):
        if training_logprobs_D:
            ratio_chunks = []
            for datum, training_logprobs in safezip(data_D, training_logprobs_D):
                sampled_logprobs = datum.loss_fn_inputs["logprobs"].to_torch()
                action_mask = datum.loss_fn_inputs["mask"].to_torch() > 0
                if action_mask.any():
                    ratio_chunks.append(
                        torch.exp(training_logprobs[action_mask] - sampled_logprobs[action_mask])
                    )
            if ratio_chunks:
                flat_ratios = torch.cat(ratio_chunks).float()
                ppo_metrics = {
                    "optim/ppo_ratio_mean": float(flat_ratios.mean().item()),
                    "optim/ppo_ratio_min": float(flat_ratios.min().item()),
                    "optim/ppo_ratio_max": float(flat_ratios.max().item()),
                    "optim/ppo_clip_low_ratio": float(
                        (flat_ratios < ppo_clip_low_threshold).float().mean().item()
                    ),
                    "optim/ppo_clip_high_ratio": float(
                        (flat_ratios > ppo_clip_high_threshold).float().mean().item()
                    ),
                    "optim/ppo_clip_low_threshold": float(ppo_clip_low_threshold),
                    "optim/ppo_clip_high_threshold": float(ppo_clip_high_threshold),
                }
            else:
                ppo_metrics = {}
        else:
            ppo_metrics = {}
        sampling_client, metrics = await original_compute_full_batch(
            training_client,
            checkpoint_mgr,
            i_batch,
            data_D,
            training_logprobs_D,
            do_compute_post_kl,
        )
        metrics.update(ppo_metrics)
        return sampling_client, metrics

    train.prepare_minibatch = prepare_minibatch_with_diagnostics
    train.compute_full_batch_metrics_and_get_sampling_client = compute_full_batch_with_diagnostics
    train._synthetic_fs_rl_diagnostics_patch = True


class _TagsOnlyEnvGroupBuilder:
    """Tiny stand-in used after spooling; prepare_minibatch only needs tags."""

    def __init__(self, tags: list[str]):
        self._tags = list(tags)

    def logging_tags(self) -> list[str]:
        return list(self._tags)


class _SamplingMetricRecord:
    """Tiny object shaped for train.compute_sampling_client_metrics."""

    def __init__(self, *, sampling_client_step: int, metrics: dict[str, Any]):
        self.sampling_client_step = sampling_client_step
        self.metrics = metrics


class _SpooledGroupRecord:
    """Small task result; completed asyncio tasks retain this instead of a trajectory."""

    def __init__(
        self,
        *,
        path: Path | None,
        spool_bytes: int,
        worker_metrics: dict[str, Any],
        uniform_rewards: bool,
        included_for_training: bool,
    ):
        self.path = path
        self.spool_bytes = spool_bytes
        self.worker_metrics = worker_metrics
        self.uniform_rewards = uniform_rewards
        self.included_for_training = included_for_training


class _MetricMeanAccumulator:
    def __init__(self) -> None:
        self._sum: dict[str, float] = {}
        self._count: dict[str, float] = {}

    def add(self, key: str, value: float, weight: float = 1.0) -> None:
        if weight <= 0:
            return
        self._sum[key] = self._sum.get(key, 0.0) + float(value) * weight
        self._count[key] = self._count.get(key, 0.0) + weight

    def metrics(self) -> dict[str, float]:
        return {
            key: self._sum[key] / self._count[key]
            for key in self._sum
            if self._count.get(key, 0.0) > 0
        }


class _TrajectoryScopeStats:
    def __init__(self) -> None:
        self.n_groups = 0
        self.n_mixed = 0
        self.n_good = 0
        self.n_bad = 0
        self.total_episodes = 0
        self.total_turns = 0
        self.total_ac_tokens = 0
        self.total_ob_tokens = 0
        self.reward_sum = 0.0
        self.reward_count = 0
        self.metric_sum: dict[str, float] = {}
        self.metric_count: dict[str, int] = {}

    @staticmethod
    def _all_same(values: list[float]) -> bool:
        return all(value == values[0] for value in values)

    def _add_metric_dict(self, metrics: dict[str, Any]) -> None:
        for key, value in metrics.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            self.metric_sum[key] = self.metric_sum.get(key, 0.0) + float(value)
            self.metric_count[key] = self.metric_count.get(key, 0) + 1

    def add_group(self, trajectory_group: Any, good_thresh: float = 0.5) -> None:
        self.n_groups += 1
        rewards = [float(reward) for reward in trajectory_group.get_total_rewards()]
        if rewards:
            if self._all_same(rewards):
                if rewards[0] >= good_thresh:
                    self.n_good += 1
                else:
                    self.n_bad += 1
            else:
                self.n_mixed += 1
            self.reward_sum += sum(rewards)
            self.reward_count += len(rewards)

        for traj in trajectory_group.trajectories_G:
            self.total_episodes += 1
            self.total_turns += len(traj.transitions)
            for transition in traj.transitions:
                self.total_ac_tokens += len(transition.ac.tokens)
                self.total_ob_tokens += transition.ob.length
                if transition.metrics:
                    self._add_metric_dict(transition.metrics)
        for traj_metrics in trajectory_group.metrics_G:
            self._add_metric_dict(traj_metrics)

    def metrics(self, prefix: str) -> dict[str, float]:
        out: dict[str, float] = {
            f"{prefix}ac_tokens_per_turn": (
                self.total_ac_tokens / self.total_turns if self.total_turns > 0 else 0.0
            ),
            f"{prefix}ob_tokens_per_turn": (
                self.total_ob_tokens / self.total_turns if self.total_turns > 0 else 0.0
            ),
            f"{prefix}turns_per_episode": (
                self.total_turns / self.total_episodes if self.total_episodes > 0 else 0.0
            ),
            f"{prefix}total_episodes": float(self.total_episodes),
            f"{prefix}total_turns": float(self.total_turns),
            f"{prefix}total_ac_tokens": float(self.total_ac_tokens),
            f"{prefix}total_ob_tokens": float(self.total_ob_tokens),
            f"{prefix}reward/total": (
                self.reward_sum / self.reward_count if self.reward_count > 0 else 0.0
            ),
            f"{prefix}by_group/frac_mixed": (
                self.n_mixed / self.n_groups if self.n_groups > 0 else 0.0
            ),
            f"{prefix}by_group/frac_all_good": (
                self.n_good / self.n_groups if self.n_groups > 0 else 0.0
            ),
            f"{prefix}by_group/frac_all_bad": (
                self.n_bad / self.n_groups if self.n_groups > 0 else 0.0
            ),
        }
        for key, total in self.metric_sum.items():
            count = self.metric_count.get(key, 0)
            if count > 0:
                out[f"{prefix}{key}"] = total / count
        return out


class _TrajectoryMetricsAccumulator:
    def __init__(self) -> None:
        self._scopes: dict[str, _TrajectoryScopeStats] = {"all": _TrajectoryScopeStats()}
        self._tag_counts: dict[str, int] = {}
        self._total_groups = 0

    def add(self, tags: list[str], trajectory_group: Any) -> None:
        self._total_groups += 1
        self._scopes["all"].add_group(trajectory_group)
        for tag in tags:
            self._tag_counts[tag] = self._tag_counts.get(tag, 0) + 1
            self._scopes.setdefault(tag, _TrajectoryScopeStats()).add_group(trajectory_group)

    def metrics(self) -> dict[str, float]:
        out: dict[str, float] = {}
        have_nontrivial_tags = any(
            count < self._total_groups for count in self._tag_counts.values()
        )
        if have_nontrivial_tags:
            for tag in sorted(self._tag_counts):
                out.update(self._scopes[tag].metrics(f"env/{tag}/"))
        out.update(self._scopes["all"].metrics("env/all/"))
        return out


class _DatumStatsAccumulator:
    def __init__(self) -> None:
        self.action_token_sum = 0.0
        self.action_token_count = 0
        self.action_token_max = 0.0
        self.datum_token_sum = 0.0
        self.datum_count = 0
        self.datum_token_max = 0.0
        self.adv_sum = 0.0
        self.adv_sumsq = 0.0
        self.adv_abs_sum = 0.0
        self.adv_count = 0
        self.adv_zero_count = 0
        self.adv_min: float | None = None
        self.adv_max: float | None = None
        self.trajectory_turn_sum = 0.0
        self.trajectory_count = 0
        self.trajectory_turn_max = 0.0

    def add(self, data_D: list[Any], trajectory_groups_P: list[Any]) -> None:
        for datum in data_D:
            advantages = datum.loss_fn_inputs["advantages"].to_torch()
            mask = datum.loss_fn_inputs["mask"].to_torch() > 0
            action_tokens = float(mask.sum().item())
            if action_tokens > 0:
                action_advantages = advantages[mask].float()
                self.action_token_sum += action_tokens
                self.action_token_count += 1
                self.action_token_max = max(self.action_token_max, action_tokens)
                self.adv_sum += float(action_advantages.sum().item())
                self.adv_sumsq += float((action_advantages**2).sum().item())
                self.adv_abs_sum += float(action_advantages.abs().sum().item())
                self.adv_count += int(action_advantages.numel())
                self.adv_zero_count += int((action_advantages.abs() <= 1e-12).sum().item())
                chunk_min = float(action_advantages.min().item())
                chunk_max = float(action_advantages.max().item())
                self.adv_min = chunk_min if self.adv_min is None else min(self.adv_min, chunk_min)
                self.adv_max = chunk_max if self.adv_max is None else max(self.adv_max, chunk_max)
            target_tokens = float(len(datum.loss_fn_inputs["target_tokens"].data))
            self.datum_token_sum += target_tokens
            self.datum_count += 1
            self.datum_token_max = max(self.datum_token_max, target_tokens)
        for group in trajectory_groups_P:
            for traj in group.trajectories_G:
                turns = float(len(traj.transitions))
                self.trajectory_turn_sum += turns
                self.trajectory_count += 1
                self.trajectory_turn_max = max(self.trajectory_turn_max, turns)

    def metrics(self) -> dict[str, float]:
        out: dict[str, float] = {}
        if self.adv_count > 0:
            mean = self.adv_sum / self.adv_count
            variance = max(0.0, self.adv_sumsq / self.adv_count - mean * mean)
            out.update(
                {
                    "optim/advantage_mean": mean,
                    "optim/advantage_std": variance**0.5,
                    "optim/advantage_min": float(self.adv_min),
                    "optim/advantage_max": float(self.adv_max),
                    "optim/advantage_abs_mean": self.adv_abs_sum / self.adv_count,
                    "optim/zero_advantage_ratio": self.adv_zero_count / self.adv_count,
                }
            )
        if self.action_token_count > 0:
            out["optim/action_tokens_max"] = self.action_token_max
            out["optim/action_tokens_mean"] = self.action_token_sum / self.action_token_count
        if self.datum_count > 0:
            out["optim/train_datum_tokens_max"] = self.datum_token_max
            out["optim/train_datum_tokens_mean"] = self.datum_token_sum / self.datum_count
        if self.trajectory_count > 0:
            out["optim/trajectory_turns_max"] = self.trajectory_turn_max
            out["optim/trajectory_turns_mean"] = self.trajectory_turn_sum / self.trajectory_count
        return out


class _KLPPOStatsAccumulator:
    def __init__(self, *, clip_low: float, clip_high: float) -> None:
        self.clip_low = clip_low
        self.clip_high = clip_high
        self.diff_sum = 0.0
        self.diff_sumsq = 0.0
        self.sample_logprob_sum = 0.0
        self.ratio_sum = 0.0
        self.ratio_min: float | None = None
        self.ratio_max: float | None = None
        self.clip_low_count = 0
        self.clip_high_count = 0
        self.count = 0

    def add(self, data_D: list[Any], training_logprobs_D: list[torch.Tensor]) -> None:
        for datum, training_logprobs in safezip(data_D, training_logprobs_D):
            sampled_logprobs = datum.loss_fn_inputs["logprobs"].to_torch()
            action_mask = datum.loss_fn_inputs["mask"].to_torch() > 0
            if not action_mask.any():
                continue
            sampled_actions = sampled_logprobs[action_mask].float()
            training_actions = training_logprobs[action_mask].float()
            diffs = sampled_actions - training_actions
            ratios = torch.exp(training_actions - sampled_actions)
            self.diff_sum += float(diffs.sum().item())
            self.diff_sumsq += float((diffs**2).sum().item())
            self.sample_logprob_sum += float(sampled_actions.sum().item())
            self.ratio_sum += float(ratios.sum().item())
            self.ratio_min = (
                float(ratios.min().item())
                if self.ratio_min is None
                else min(self.ratio_min, float(ratios.min().item()))
            )
            self.ratio_max = (
                float(ratios.max().item())
                if self.ratio_max is None
                else max(self.ratio_max, float(ratios.max().item()))
            )
            self.clip_low_count += int((ratios < self.clip_low).sum().item())
            self.clip_high_count += int((ratios > self.clip_high).sum().item())
            self.count += int(ratios.numel())

    def metrics(self) -> dict[str, float]:
        if self.count <= 0:
            return {}
        return {
            "optim/kl_sample_train_v1": self.diff_sum / self.count,
            "optim/kl_sample_train_v2": 0.5 * self.diff_sumsq / self.count,
            "optim/entropy": -self.sample_logprob_sum / self.count,
            "optim/ppo_ratio_mean": self.ratio_sum / self.count,
            "optim/ppo_ratio_min": float(self.ratio_min),
            "optim/ppo_ratio_max": float(self.ratio_max),
            "optim/ppo_clip_low_ratio": self.clip_low_count / self.count,
            "optim/ppo_clip_high_ratio": self.clip_high_count / self.count,
            "optim/ppo_clip_low_threshold": float(self.clip_low),
            "optim/ppo_clip_high_threshold": float(self.clip_high),
        }


def _current_rss_mb() -> float | None:
    try:
        with Path("/proc/self/status").open(encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return float(parts[1]) / 1024.0
    except OSError:
        return None
    return None


def _chunked(items: list[Path], chunk_size: int) -> list[list[Path]]:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def _spool_group_payload(
    *,
    path: Path,
    tags: list[str],
    trajectory_group: Any,
    worker_metrics: dict[str, Any],
    sampling_client_step: int,
) -> int:
    payload = {
        "tags": tags,
        "trajectory_group": trajectory_group,
        "worker_metrics": worker_metrics,
        "sampling_client_step": sampling_client_step,
        "uniform_rewards": len(set(trajectory_group.get_total_rewards())) <= 1,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with tmp_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, path)
    return path.stat().st_size


def _load_group_payload(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return pickle.load(handle)


async def _spooled_train_step_and_get_sampling_client(
    *,
    config: Any,
    i_batch: int,
    training_client: Any,
    checkpoint_mgr: Any,
    kl_reference_client: Any,
    tokenizer: Any,
    group_files: list[Path],
    ram_spool_minibatch_groups: int,
    ppo_clip_low_threshold: float,
    ppo_clip_high_threshold: float,
) -> tuple[Any, dict[str, Any]]:
    if getattr(config, "compute_post_kl", False):
        raise ValueError("compute_post_kl=true is not supported with ram_spool_enabled=true.")

    metrics: dict[str, Any] = {}
    datum_stats = _DatumStatsAccumulator()
    kl_ppo_stats = _KLPPOStatsAccumulator(
        clip_low=ppo_clip_low_threshold,
        clip_high=ppo_clip_high_threshold,
    )
    kl_policy_base_means = _MetricMeanAccumulator()

    num_substeps = min(max(1, int(config.num_substeps)), len(group_files))
    substep_file_batches = train.split_list(group_files, num_substeps)
    adam_params = train.tinker.AdamParams(
        learning_rate=config.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8
    )

    for i_substep, substep_files in enumerate(substep_file_batches):
        substep_had_data = False
        for i_chunk, chunk_files in enumerate(_chunked(substep_files, ram_spool_minibatch_groups)):
            payloads = [_load_group_payload(path) for path in chunk_files]
            env_group_builders_P = [
                _TagsOnlyEnvGroupBuilder(payload["tags"]) for payload in payloads
            ]
            trajectory_groups_P = [payload["trajectory_group"] for payload in payloads]
            async with train.trace.scope_span(
                f"spool_prepare_substep_{i_substep}_chunk_{i_chunk}"
            ):
                data_D, prepare_metrics = await train.prepare_minibatch(
                    env_group_builders_P,
                    trajectory_groups_P,
                    tokenizer,
                    kl_reference_client,
                    kl_penalty_coef=config.kl_penalty_coef,
                    kl_discount_factor=config.kl_discount_factor,
                )
            if "kl_policy_base" in prepare_metrics:
                action_weight = 0.0
                for datum in data_D:
                    action_weight += float(
                        (datum.loss_fn_inputs["mask"].to_torch() > 0).sum().item()
                    )
                kl_policy_base_means.add(
                    "kl_policy_base", float(prepare_metrics["kl_policy_base"]), action_weight
                )
            datum_stats.add(data_D, trajectory_groups_P)
            if not data_D:
                del payloads, env_group_builders_P, trajectory_groups_P, data_D
                gc.collect()
                continue

            substep_had_data = True
            async with train.trace.scope_span(
                f"spool_train_fwd_bwd_substep_{i_substep}_chunk_{i_chunk}"
            ):
                fwd_bwd_future = await training_client.forward_backward_async(
                    [train._remove_mask(d) for d in data_D],
                    loss_fn=config.loss_fn,
                    loss_fn_config=config.loss_fn_config,
                )
                fwd_bwd_result = await fwd_bwd_future.result_async()
            training_logprobs_D = train._training_logprobs_from_fwd_bwd(fwd_bwd_result)
            kl_ppo_stats.add(data_D, training_logprobs_D)

            del (
                payloads,
                env_group_builders_P,
                trajectory_groups_P,
                data_D,
                training_logprobs_D,
                fwd_bwd_result,
            )
            gc.collect()

        if substep_had_data:
            async with train.trace.scope_span(f"spool_train_optim_substep_{i_substep}"):
                optim_future = await training_client.optim_step_async(adam_params)
                optim_result = await optim_future.result_async()
            if optim_result.metrics:
                metrics.update(optim_result.metrics)

    metrics.update(datum_stats.metrics())
    metrics.update(kl_ppo_stats.metrics())
    metrics.update(kl_policy_base_means.metrics())

    sampling_client, checkpoint_metrics = await train.save_checkpoint_and_get_sampling_client(
        training_client,
        checkpoint_mgr,
        i_batch + 1,
    )
    metrics.update(checkpoint_metrics)
    return sampling_client, metrics


def _install_spooled_sync_training_patch(
    *,
    ram_spool_dir: str,
    ram_spool_minibatch_groups: int,
    ram_spool_cleanup: bool,
    ppo_clip_low_threshold: float,
    ppo_clip_high_threshold: float,
) -> None:
    if ram_spool_minibatch_groups <= 0:
        raise ValueError(
            f"ram_spool_minibatch_groups must be positive, got {ram_spool_minibatch_groups}"
        )

    spool_root = Path(ram_spool_dir).expanduser().resolve()
    spool_root.mkdir(parents=True, exist_ok=True)

    @train.trace.scope
    async def do_sync_training_spooled(
        start_batch: int,
        end_batch: int,
        num_batches: int,
        config: Any,
        training_client: Any,
        kl_reference_client: Any,
        evaluators: list[Any],
        dataset: Any,
        ml_logger: Any,
        tokenizer: Any,
        error_counter: Any | None = None,
        strategy: Any | None = None,
        checkpoint_mgr: Any | None = None,
    ) -> None:
        if config.rollout_json_export:
            raise ValueError(
                "rollout_json_export=true is not supported with ram_spool_enabled=true."
            )
        if getattr(config, "compute_post_kl", False):
            raise ValueError("compute_post_kl=true is not supported with ram_spool_enabled=true.")
        if config.kl_penalty_coef > 0:
            raise ValueError(
                "kl_penalty_coef > 0 is not supported with ram_spool_enabled=true because "
                "KL-penalty advantages are normalized within each prepared chunk."
            )

        assert checkpoint_mgr is not None
        sampling_client, _ = await train.save_checkpoint_and_get_sampling_client(
            training_client, checkpoint_mgr, start_batch, start_batch
        )

        for i_batch in range(start_batch, end_batch):
            metrics: dict[str, Any] = {
                "progress/batch": i_batch,
                "optim/lr": config.learning_rate,
                "progress/done_frac": (i_batch + 1) / num_batches,
                "ram_spool/enabled": 1.0,
                "ram_spool/minibatch_groups": float(ram_spool_minibatch_groups),
            }
            batch_dir = spool_root / f"batch_{i_batch:06d}"
            if batch_dir.exists():
                shutil.rmtree(batch_dir)
            batch_dir.mkdir(parents=True, exist_ok=True)

            try:
                with train.trace.trace_iteration(step=i_batch) as window:
                    if config.eval_every > 0 and i_batch % config.eval_every == 0:
                        eval_metrics = await train.run_evaluations_parallel(
                            evaluators,
                            sampling_client,
                            config,
                            i_batch,
                            store=ml_logger.store,
                        )
                        metrics.update(eval_metrics)

                    env_group_builders_P = dataset.get_batch(i_batch)
                    spooled_files: list[Path] = []
                    training_files: list[Path] = []
                    first_success_file: Path | None = None
                    trajectory_metrics = _TrajectoryMetricsAccumulator()
                    sampling_metric_records: list[Any] = []
                    total_spool_bytes = 0

                    iter_dir = train.iteration_dir(config.log_path, i_batch)
                    async with train.trace.scope_span("sampling"):
                        with train._get_logtree_scope(
                            output_dir=iter_dir,
                            num_groups_to_log=config.num_groups_to_log,
                            f_name="train",
                            scope_name=f"RL Iteration {i_batch}",
                            iteration=i_batch,
                            store=ml_logger.store,
                        ):

                            async def run_one_group(
                                group_idx: int, builder: Any
                            ) -> _SpooledGroupRecord:
                                t_start = time.time()
                                result = await train.do_group_rollout_and_filter_constant_reward(
                                    sampling_client,
                                    builder,
                                    max_tokens=config.max_tokens,
                                    temperature=config.temperature,
                                    do_remove_constant_reward_groups=False,
                                    enable_logging=group_idx < config.num_groups_to_log,
                                    strategy=strategy,
                                )
                                worker_metrics = {
                                    "time/trajectory_group_worker_loop/total": (
                                        time.time() - t_start
                                    )
                                }
                                if error_counter is not None:
                                    error_counter.ingest(result)
                                if result is None:
                                    return _SpooledGroupRecord(
                                        path=None,
                                        spool_bytes=0,
                                        worker_metrics=worker_metrics,
                                        uniform_rewards=True,
                                        included_for_training=False,
                                    )
                                tags = builder.logging_tags()
                                path = batch_dir / f"group_{group_idx:04d}.pkl"
                                spool_bytes = _spool_group_payload(
                                    path=path,
                                    tags=tags,
                                    trajectory_group=result,
                                    worker_metrics=worker_metrics,
                                    sampling_client_step=i_batch,
                                )
                                uniform_rewards = len(set(result.get_total_rewards())) <= 1
                                included_for_training = (
                                    not config.remove_constant_reward_groups
                                    or not uniform_rewards
                                )
                                if included_for_training:
                                    trajectory_metrics.add(tags, result)
                                del result
                                gc.collect()
                                return _SpooledGroupRecord(
                                    path=path,
                                    spool_bytes=spool_bytes,
                                    worker_metrics=worker_metrics,
                                    uniform_rewards=uniform_rewards,
                                    included_for_training=included_for_training,
                                )

                            tasks = [
                                asyncio.create_task(
                                    run_one_group(i, builder),
                                    name=f"spooled_trajectory_group_worker_{i}",
                                )
                                for i, builder in enumerate(env_group_builders_P)
                            ]
                            pbar = train.tqdm(
                                total=len(tasks), desc=f"Sampling batch {i_batch}"
                            )
                            try:
                                for completed in asyncio.as_completed(tasks):
                                    record = await completed
                                    pbar.update(1)
                                    if record.path is None:
                                        continue
                                    total_spool_bytes += record.spool_bytes
                                    spooled_files.append(record.path)
                                    if first_success_file is None:
                                        first_success_file = record.path
                                    if record.included_for_training:
                                        training_files.append(record.path)
                                    sampling_metric_records.append(
                                        _SamplingMetricRecord(
                                            sampling_client_step=i_batch,
                                            metrics=record.worker_metrics,
                                        )
                                    )
                                    gc.collect()
                            except Exception:
                                for task in tasks:
                                    task.cancel()
                                await asyncio.gather(*tasks, return_exceptions=True)
                                raise
                            finally:
                                pbar.close()

                    if (
                        config.remove_constant_reward_groups
                        and not training_files
                        and first_success_file is not None
                    ):
                        training_files = [first_success_file]
                        payload = _load_group_payload(first_success_file)
                        trajectory_metrics.add(payload["tags"], payload["trajectory_group"])
                        del payload

                    batch_skipped = not training_files
                    metrics["ram_spool/spooled_groups"] = float(len(spooled_files))
                    metrics["ram_spool/training_groups"] = float(len(training_files))
                    metrics["ram_spool/spool_bytes_mb"] = total_spool_bytes / (1024.0 * 1024.0)
                    if rss_mb := _current_rss_mb():
                        metrics["ram_spool/rss_mb_after_sampling"] = rss_mb

                    if batch_skipped:
                        logger.warning(
                            "Batch %s: all groups failed or filtered, skipping batch", i_batch
                        )
                    else:
                        metrics.update(
                            train.compute_sampling_client_metrics(sampling_metric_records)
                        )
                        metrics.update(trajectory_metrics.metrics())
                        sampling_client, train_step_metrics = (
                            await _spooled_train_step_and_get_sampling_client(
                                config=config,
                                i_batch=i_batch,
                                training_client=training_client,
                                checkpoint_mgr=checkpoint_mgr,
                                kl_reference_client=kl_reference_client,
                                tokenizer=tokenizer,
                                group_files=training_files,
                                ram_spool_minibatch_groups=ram_spool_minibatch_groups,
                                ppo_clip_low_threshold=ppo_clip_low_threshold,
                                ppo_clip_high_threshold=ppo_clip_high_threshold,
                            )
                        )
                        metrics.update(train_step_metrics)
                        if checkpoint_mgr is not None:
                            await checkpoint_mgr.maybe_save_rolling_async(
                                step=i_batch + 1, loop_state={"batch": i_batch + 1}
                            )

                metrics.update(window.get_timing_metrics())
                if error_counter is not None:
                    metrics.update(error_counter.get_metrics())
                if rss_mb := _current_rss_mb():
                    metrics["ram_spool/rss_mb_after_batch"] = rss_mb
                window.save_timing(i_batch, store=ml_logger.store)
                if (
                    config.span_chart_every > 0
                    and i_batch % config.span_chart_every == 0
                    and iter_dir is not None
                ):
                    iter_dir.mkdir(parents=True, exist_ok=True)
                    train.trace.save_gantt_chart_html(
                        window, i_batch, iter_dir / "timing_gantt.html"
                    )
                ml_logger.log_metrics(metrics, step=i_batch)
            finally:
                if ram_spool_cleanup and batch_dir.exists():
                    shutil.rmtree(batch_dir)
                del metrics
                gc.collect()

    train.do_sync_training = do_sync_training_spooled
    train._synthetic_fs_spooled_sync_patch = True


@chz.chz
class CLIConfig:
    model_name: str = "Qwen/Qwen3.5-35B-A3B"
    lora_rank: int = 32
    renderer_name: str | None = None

    learning_rate: float = 4e-5
    batch_size: int = 16
    group_size: int = 4
    train_epochs: int = 1
    reshuffle_each_epoch: bool = True
    num_substeps: int = 1
    stream_minibatch_groups_per_batch: int = 0
    stream_minibatch_num_minibatches: int = 0
    ram_spool_enabled: bool = False
    ram_spool_dir: str = "/scr/asap7772/tinker_synthfs_spool"
    ram_spool_minibatch_groups: int = 4
    ram_spool_cleanup: bool = True
    loss_fn: str = "ppo"
    loss_fn_config_json: str = ""
    ppo_clip_low_threshold: float = 0.8
    ppo_clip_high_threshold: float = 1.2
    log_rl_diagnostics: bool = True
    seed: int = 2
    max_tokens: int = 4096
    model_context_window_tokens: int | None = 65536
    context_window_safety_tokens: int = 256
    eval_every: int = 0
    max_steps: int | None = 110
    save_every: int = 5
    ttl_seconds: int | None = 604800
    rolling_save_every: int = 1
    rolling_ttl_seconds: int = 604800
    load_checkpoint_path: str | None = None

    index_jsonl: str = "../data/tinker_synthetic_fs_alltrain/index.jsonl"
    reward_mode: str = "hybrid"

    answerer_backend: str = "gemini"
    answerer_model: str = "gemini-3.1-flash-lite-preview"
    answerer_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    answerer_api_key_env: str = "GEMINI_API_KEY"

    judge_backend: str = "gemini"
    judge_model: str = "gemini-3.1-flash-lite-preview"
    judge_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    judge_api_key_env: str = "GEMINI_API_KEY"

    max_turns: int = 32
    max_trajectory_tokens: int | None = 140000
    max_generation_tokens: int | None = None
    step_penalty: float = 0.0
    termination_penalty: float = 0.1
    raw_docs_penalty: float = 0.0
    empty_synthetic_penalty: float = 1.0
    synthetic_success_bonus: float = 0.0
    synthetic_usage_bonus: float = 0.0
    raw_usage_ratio_penalty: float = 0.0
    filesystem_maturity_scale: float = 0.5
    filesystem_coverage_weight: float = 0.35
    filesystem_expansion_weight: float = 0.3
    filesystem_organization_weight: float = 0.35
    filesystem_stop_weight: float = 0.0
    mature_stop_bonus: float = 0.0
    mature_stop_min_score: float = 0.8
    terminal_reward_clip_min: float = -1.0
    terminal_reward_clip_max: float = 3.0
    answerer_max_turns: int = 32
    answerer_workspace_mode: str = "synthetic_only"
    answerer_final_answer_max_tokens: int = 128
    answerer_retrieval_cost_scale: float = 0.15
    answerer_retrieval_cost_token_unit: float = 1000.0
    answerer_retrieval_cost_correct_only: bool = True
    answerer_synthetic_read_cost_scale: float = 0.10
    answerer_synthetic_read_cost_unit: float = 10.0
    terminal_answerer_repeats: int = 4
    answerability_delta_reward_scale: float = 0.5
    answerability_delta_min_abs: float = 0.25
    answerability_delta_allow_negative: bool = True
    answerability_probe_max_per_episode: int = 4
    answerability_probe_interval_turns: int = 8
    answerability_probe_min_maturity: float = 0.45
    answerability_probe_repeats: int = 4
    judge_max_output_tokens: int = 64
    log_step_details: bool = False
    log_compaction_summaries: bool = False
    retain_reward_tool_messages: bool = False
    trim_terminal_history_for_memory: bool = True
    return_empty_terminal_observation: bool = True
    clear_state_on_terminal_for_memory: bool = True
    disable_sample_trajectory_printing: bool = True
    num_groups_to_log: int = 0
    rollout_json_export: bool = False
    builder_compaction_enabled: bool = True
    builder_compaction_backend: str = "gemini"
    builder_compaction_model: str = "gemini-3.1-flash-lite-preview"
    builder_compaction_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    builder_compaction_api_key_env: str = "GEMINI_API_KEY"
    builder_compaction_trigger_tokens: int = 3000
    builder_compaction_keep_recent_turns: int = 1
    builder_compaction_max_output_tokens: int = 800
    builder_compaction_input_max_chars: int = 60000
    builder_executor_enabled: bool = True
    builder_batch_tools_enabled: bool = True
    builder_executor_backend: str = "vllm"
    builder_executor_model: str = "Qwen/Qwen3.5-35B-A3B"
    builder_executor_base_url: str = (
        "https://iris-lab-ws--lateral-vllm-qwen3-5-35b-a3b.modal.run/v1/chat/completions"
    )
    builder_executor_api_key_env: str = ""
    builder_executor_max_source_chars: int = 16000
    builder_executor_max_output_tokens: int = 512
    step_construction_action_bonus: float = 0.05
    step_filesystem_maturity_delta_scale: float = 0.5
    step_non_construction_turn_penalty: float = 0.005
    step_non_construction_streak_penalty: float = 0.0
    step_non_construction_streak_free: int = 3
    step_tool_error_penalty: float = 0.05

    excluded_qids_jsonl: str = ""
    eval_index_jsonl: str = ""
    eval_size: int = 0
    limit: int = 0

    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


async def cli_main(cli_config: CLIConfig) -> None:
    if cli_config.max_generation_tokens is not None:
        raise ValueError(
            "max_generation_tokens is deprecated in this pipeline. "
            "The builder should not be method-limited by a per-turn generation cap. "
            "Remove max_generation_tokens and, if needed, use answerer_final_answer_max_tokens "
            "to control only the answerer's final answer length."
        )
    if cli_config.answerer_workspace_mode != "synthetic_only":
        raise ValueError("This clean setup only supports answerer_workspace_mode=synthetic_only.")
    renderer_name = cli_config.renderer_name or model_info.get_recommended_renderer_name(
        cli_config.model_name
    )
    effective_model_context_window_tokens = _effective_model_context_window_tokens(
        model_name=cli_config.model_name,
        configured_context_window_tokens=cli_config.model_context_window_tokens,
    )
    _install_sampler_context_window_patch(effective_model_context_window_tokens)
    effective_max_trajectory_tokens = cli_config.max_trajectory_tokens
    if effective_model_context_window_tokens is not None:
        safety_tokens = max(0, cli_config.context_window_safety_tokens)
        context_safe_max_trajectory_tokens = (
            effective_model_context_window_tokens - cli_config.max_tokens - safety_tokens
        )
        if context_safe_max_trajectory_tokens <= 0:
            raise ValueError(
                "model_context_window_tokens must be larger than max_tokens plus "
                f"context_window_safety_tokens; got {effective_model_context_window_tokens} "
                f"<= {cli_config.max_tokens} + {safety_tokens}."
            )
        if effective_max_trajectory_tokens is None:
            effective_max_trajectory_tokens = context_safe_max_trajectory_tokens
        else:
            effective_max_trajectory_tokens = min(
                effective_max_trajectory_tokens,
                context_safe_max_trajectory_tokens,
            )
        if effective_max_trajectory_tokens != cli_config.max_trajectory_tokens:
            logger.info(
                "Using max_trajectory_tokens=%d after context-window cap "
                "(configured=%s, context_window=%d, max_tokens=%d, safety=%d).",
                effective_max_trajectory_tokens,
                cli_config.max_trajectory_tokens,
                effective_model_context_window_tokens,
                cli_config.max_tokens,
                safety_tokens,
            )
    builder = SyntheticFilesystemDatasetBuilder(
        index_jsonl=cli_config.index_jsonl,
        model_name_for_tokenizer=cli_config.model_name,
        batch_size=cli_config.batch_size,
        group_size=cli_config.group_size,
        train_epochs=cli_config.train_epochs,
        reshuffle_each_epoch=cli_config.reshuffle_each_epoch,
        renderer_name=renderer_name,
        reward_mode=cli_config.reward_mode,
        answerer_backend=cli_config.answerer_backend,
        answerer_model=cli_config.answerer_model,
        answerer_base_url=cli_config.answerer_base_url,
        answerer_api_key_env=cli_config.answerer_api_key_env,
        judge_backend=cli_config.judge_backend,
        judge_model=cli_config.judge_model,
        judge_base_url=cli_config.judge_base_url,
        judge_api_key_env=cli_config.judge_api_key_env,
        max_turns=cli_config.max_turns,
        max_trajectory_tokens=effective_max_trajectory_tokens,
        max_generation_tokens=cli_config.max_generation_tokens,
        step_penalty=cli_config.step_penalty,
        termination_penalty=cli_config.termination_penalty,
        raw_docs_penalty=cli_config.raw_docs_penalty,
        empty_synthetic_penalty=cli_config.empty_synthetic_penalty,
        synthetic_success_bonus=cli_config.synthetic_success_bonus,
        synthetic_usage_bonus=cli_config.synthetic_usage_bonus,
        raw_usage_ratio_penalty=cli_config.raw_usage_ratio_penalty,
        filesystem_maturity_scale=cli_config.filesystem_maturity_scale,
        filesystem_coverage_weight=cli_config.filesystem_coverage_weight,
        filesystem_expansion_weight=cli_config.filesystem_expansion_weight,
        filesystem_organization_weight=cli_config.filesystem_organization_weight,
        filesystem_stop_weight=cli_config.filesystem_stop_weight,
        mature_stop_bonus=cli_config.mature_stop_bonus,
        mature_stop_min_score=cli_config.mature_stop_min_score,
        terminal_reward_clip_min=cli_config.terminal_reward_clip_min,
        terminal_reward_clip_max=cli_config.terminal_reward_clip_max,
        answerer_max_turns=cli_config.answerer_max_turns,
        answerer_workspace_mode=cli_config.answerer_workspace_mode,
        answerer_final_answer_max_tokens=cli_config.answerer_final_answer_max_tokens,
        answerer_retrieval_cost_scale=cli_config.answerer_retrieval_cost_scale,
        answerer_retrieval_cost_token_unit=cli_config.answerer_retrieval_cost_token_unit,
        answerer_retrieval_cost_correct_only=cli_config.answerer_retrieval_cost_correct_only,
        answerer_synthetic_read_cost_scale=cli_config.answerer_synthetic_read_cost_scale,
        answerer_synthetic_read_cost_unit=cli_config.answerer_synthetic_read_cost_unit,
        terminal_answerer_repeats=cli_config.terminal_answerer_repeats,
        answerability_delta_reward_scale=cli_config.answerability_delta_reward_scale,
        answerability_delta_min_abs=cli_config.answerability_delta_min_abs,
        answerability_delta_allow_negative=cli_config.answerability_delta_allow_negative,
        answerability_probe_max_per_episode=cli_config.answerability_probe_max_per_episode,
        answerability_probe_interval_turns=cli_config.answerability_probe_interval_turns,
        answerability_probe_min_maturity=cli_config.answerability_probe_min_maturity,
        answerability_probe_repeats=cli_config.answerability_probe_repeats,
        judge_max_output_tokens=cli_config.judge_max_output_tokens,
        log_step_details=cli_config.log_step_details,
        log_compaction_summaries=cli_config.log_compaction_summaries,
        retain_reward_tool_messages=cli_config.retain_reward_tool_messages,
        trim_terminal_history_for_memory=cli_config.trim_terminal_history_for_memory,
        return_empty_terminal_observation=cli_config.return_empty_terminal_observation,
        clear_state_on_terminal_for_memory=cli_config.clear_state_on_terminal_for_memory,
        builder_compaction_enabled=cli_config.builder_compaction_enabled,
        builder_compaction_backend=cli_config.builder_compaction_backend,
        builder_compaction_model=cli_config.builder_compaction_model,
        builder_compaction_base_url=cli_config.builder_compaction_base_url,
        builder_compaction_api_key_env=cli_config.builder_compaction_api_key_env,
        builder_compaction_trigger_tokens=cli_config.builder_compaction_trigger_tokens,
        builder_compaction_keep_recent_turns=cli_config.builder_compaction_keep_recent_turns,
        builder_compaction_max_output_tokens=cli_config.builder_compaction_max_output_tokens,
        builder_compaction_input_max_chars=cli_config.builder_compaction_input_max_chars,
        builder_executor_enabled=cli_config.builder_executor_enabled,
        builder_batch_tools_enabled=cli_config.builder_batch_tools_enabled,
        builder_executor_backend=cli_config.builder_executor_backend,
        builder_executor_model=cli_config.builder_executor_model,
        builder_executor_base_url=cli_config.builder_executor_base_url,
        builder_executor_api_key_env=cli_config.builder_executor_api_key_env,
        builder_executor_max_source_chars=cli_config.builder_executor_max_source_chars,
        builder_executor_max_output_tokens=cli_config.builder_executor_max_output_tokens,
        step_construction_action_bonus=cli_config.step_construction_action_bonus,
        step_filesystem_maturity_delta_scale=cli_config.step_filesystem_maturity_delta_scale,
        step_non_construction_turn_penalty=cli_config.step_non_construction_turn_penalty,
        step_non_construction_streak_penalty=cli_config.step_non_construction_streak_penalty,
        step_non_construction_streak_free=cli_config.step_non_construction_streak_free,
        step_tool_error_penalty=cli_config.step_tool_error_penalty,
        excluded_qids_jsonl=cli_config.excluded_qids_jsonl,
        eval_index_jsonl=cli_config.eval_index_jsonl,
        seed=cli_config.seed,
        eval_size=cli_config.eval_size,
        limit=cli_config.limit,
    )

    model_name_short = cli_config.model_name.lower().replace("/", "-")
    date_and_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_name = (
        f"synthetic_fs_{model_name_short}_bs{cli_config.batch_size}_"
        f"gs{cli_config.group_size}_seed{cli_config.seed}_lr{cli_config.learning_rate}_"
        f"rank{cli_config.lora_rank}_{date_and_time}"
    )

    log_path = cli_config.log_path or f"/tmp/tinker-examples/rl_synthetic_fs/{run_name}"
    wandb_name = cli_config.wandb_name or run_name

    if not Path("/tmp").exists():
        raise ValueError("/tmp does not exist")

    if cli_config.disable_sample_trajectory_printing:
        train.print_group = lambda *args, **kwargs: None

    if cli_config.log_rl_diagnostics:
        _install_rl_diagnostics_patch(
            ppo_clip_low_threshold=cli_config.ppo_clip_low_threshold,
            ppo_clip_high_threshold=cli_config.ppo_clip_high_threshold,
        )

    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    stream_minibatch_config = None
    if cli_config.stream_minibatch_groups_per_batch or cli_config.stream_minibatch_num_minibatches:
        if cli_config.ram_spool_enabled:
            raise ValueError(
                "ram_spool_enabled=true replaces the sync training path and cannot be combined "
                "with stream_minibatch_groups_per_batch/stream_minibatch_num_minibatches."
            )
        if (
            cli_config.stream_minibatch_groups_per_batch <= 0
            or cli_config.stream_minibatch_num_minibatches <= 0
        ):
            raise ValueError(
                "stream_minibatch_groups_per_batch and stream_minibatch_num_minibatches "
                "must both be positive when either is set."
            )
        if cli_config.batch_size < cli_config.stream_minibatch_groups_per_batch:
            raise ValueError(
                "batch_size must be at least stream_minibatch_groups_per_batch; otherwise "
                "Tinker's synchronous streaming trainer will wait for groups that were never sampled."
            )
        stream_minibatch_config = train.StreamMinibatchConfig(
            groups_per_batch=cli_config.stream_minibatch_groups_per_batch,
            num_minibatches=cli_config.stream_minibatch_num_minibatches,
        )

    loss_fn_config = json.loads(cli_config.loss_fn_config_json) if cli_config.loss_fn_config_json else None
    if cli_config.loss_fn == "ppo" and loss_fn_config is None:
        loss_fn_config = {
            "clip_low_threshold": cli_config.ppo_clip_low_threshold,
            "clip_high_threshold": cli_config.ppo_clip_high_threshold,
        }

    if cli_config.ram_spool_enabled:
        if cli_config.rollout_json_export:
            raise ValueError(
                "rollout_json_export=true is not supported with ram_spool_enabled=true."
            )
        _install_spooled_sync_training_patch(
            ram_spool_dir=cli_config.ram_spool_dir,
            ram_spool_minibatch_groups=cli_config.ram_spool_minibatch_groups,
            ram_spool_cleanup=cli_config.ram_spool_cleanup,
            ppo_clip_low_threshold=cli_config.ppo_clip_low_threshold,
            ppo_clip_high_threshold=cli_config.ppo_clip_high_threshold,
        )

    config = train.Config(
        model_name=cli_config.model_name,
        renderer_name=renderer_name,
        log_path=log_path,
        dataset_builder=builder,
        learning_rate=cli_config.learning_rate,
        max_tokens=cli_config.max_tokens,
        eval_every=cli_config.eval_every,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        lora_rank=cli_config.lora_rank,
        loss_fn=cli_config.loss_fn,
        loss_fn_config=loss_fn_config,
        num_substeps=cli_config.num_substeps,
        stream_minibatch_config=stream_minibatch_config,
        max_steps=cli_config.max_steps,
        save_every=cli_config.save_every,
        ttl_seconds=cli_config.ttl_seconds,
        rolling_save_every=cli_config.rolling_save_every,
        rolling_ttl_seconds=cli_config.rolling_ttl_seconds,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        num_groups_to_log=cli_config.num_groups_to_log,
        rollout_json_export=cli_config.rollout_json_export,
    )
    await train.main(config)


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config))
