import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_map
from mlx_lm.generate import BatchGenerator
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tuner.callbacks import TrainingCallback
from tqdm import tqdm

from ..recurrent_patch import enable_memory_safe_recurrences, model_uses_recurrence
from .grpo_reward_functions import (
    RewardFunctions,
    r1_accuracy_reward_func,
    r1_count_xml,
    r1_int_reward_func,
    r1_soft_format_reward_func,
    r1_strict_format_reward_func,
)
from .sft_trainer import SFTTrainingArgs, average_gradients, grad_checkpoint


@dataclass
class GRPOTrainingArgs(SFTTrainingArgs):
    group_size: int = field(
        default=4,
        metadata={"help": "Number of responses per prompt."},
    )
    beta: float = field(default=0.1, metadata={"help": "KL penalty coefficient."})
    epsilon: float = field(
        default=1e-4, metadata={"help": "Lower importance-ratio clipping width."}
    )
    epsilon_high: float = field(
        default=None,
        metadata={
            "help": "For DAPO Upper-bound epsilon value for clipping. If not specified, it defaults to the same value as the lower-bound specified in argument epsilon."
        },
    )
    max_completion_length: int = field(
        default=512, metadata={"help": "Number of Generations."}
    )
    reference_model_path: str = field(
        default=None,
        metadata={
            "help": "Path to reference model weights. If None, uses the same model."
        },
    )
    temperature: float = field(
        default=0.8,
        metadata={
            "help": "Temperature for sampling. The higher the temperature, the more random the completions."
        },
    )
    top_p: float = field(
        default=0.95,
        metadata={"help": "Top-p sampling parameter."},
    )
    top_k: int = field(default=20, metadata={"help": "Top-k sampling parameter."})
    min_p: float = field(
        default=0.0, metadata={"help": "Minimum probability for sampling."}
    )
    grpo_loss_type: str = field(
        default="grpo",
        metadata={
            "help": "Type of loss to use for GRPO. Supported: 'grpo', 'bnpo', 'dr_grpo'."
        },
    )
    reward_weights: Optional[List[float]] = field(
        default=None,
        metadata={
            "help": "Weights for each reward function. Must match the number of reward functions. If `None`, all rewards are weighted equally with weight `1.0`."
        },
    )
    importance_sampling_level: str = field(
        default="token",
        metadata={
            "help": (
                "Importance-sampling level (defaults to 'token'). 'token' keeps "
                "one log-probability ratio per token, while 'sequence' averages "
                "the ratios across valid tokens."
            )
        },
    )


@mx.compile
def _select_token_logps(logits, targets, mask):
    """Select float32 log probabilities without materializing log-softmax."""
    # Mask before reductions: multiplying an invalid log probability by zero
    # afterwards still produces NaN. Only the small selected scores are retained.
    logits = mx.where(mask[..., None], logits, 0).astype(mx.float32)
    logits = logits - mx.stop_gradient(mx.max(logits, axis=-1, keepdims=True))
    targets = targets[..., None]
    selected = mx.take_along_axis(logits, targets, axis=-1).squeeze(-1)
    return mx.where(mask, selected - mx.logsumexp(logits, axis=-1), 0)


def _get_token_logps(model, inputs, mask):
    return _select_token_logps(model(inputs[:, :-1]), inputs[:, 1:], mask)


def get_per_token_logps(model: nn.Module, inputs, lengths):
    """Return unpadded next-token log probabilities (legacy helper API)."""
    lengths = lengths.tolist() if isinstance(lengths, mx.array) else lengths
    mask = mx.arange(inputs.shape[1] - 1)[None, :] < (mx.array(lengths)[:, None] - 1)
    logps = _get_token_logps(model, inputs, mask)
    return [logps[i, : max(int(length) - 1, 0)] for i, length in enumerate(lengths)]


def _prepare_grpo_inputs(batch, completions, batch_indices):
    """Right-pad prompt + completion sequences and mask completion targets."""
    if not completions:
        raise ValueError("No completions were generated.")
    if batch_indices is None or len(batch_indices) != len(completions):
        raise ValueError("batch_indices must identify the prompt for each completion.")
    sequences, prompt_lengths, completion_lengths = [], [], []
    for completion, prompt_idx in zip(completions, batch_indices):
        prompt = mx.array(batch[0][prompt_idx], dtype=mx.int32)
        if prompt.size == 0:
            raise ValueError("GRPO requires non-empty tokenized prompts.")
        sequences.append(mx.concatenate([prompt, completion.astype(mx.int32)]))
        prompt_lengths.append(prompt.size)
        completion_lengths.append(completion.size)
    # Keep a valid model input even when every completion is empty.
    width = max(2, max(seq.size for seq in sequences))
    inputs = mx.stack([mx.pad(seq, (0, width - seq.size)) for seq in sequences])
    starts = mx.array(prompt_lengths)[:, None] - 1
    lengths = mx.array(completion_lengths)
    positions = mx.arange(width - 1)[None, :]
    mask = (positions >= starts) & (positions < starts + lengths[:, None])
    return inputs, mask, lengths


def compute_log_importance_weights(
    log_ratio: mx.array,
    length_mask: mx.array,
    importance_sampling_level: str,
) -> mx.array:
    """Computes token- or sequence-level GRPO importance weights in log space.

    The returned value must retain its gradient path through ``log_ratio``. A
    graph-independent zero would have the same on-policy forward value but
    would remove the policy gradient entirely.

    Args:
        log_ratio: Per-token log probability ratio between the policy and old
            policy.
        length_mask: Mask selecting valid completion tokens.
        importance_sampling_level: Either ``"token"`` or ``"sequence"``.

    Returns:
        Log importance weights with a gradient path to ``log_ratio``.

    Raises:
        ValueError: If ``importance_sampling_level`` is unsupported.
    """
    if importance_sampling_level == "token":
        return log_ratio
    if importance_sampling_level == "sequence":
        sequence_log_ratio = mx.where(length_mask, log_ratio, 0).sum(
            axis=1
        ) / mx.maximum(length_mask.sum(axis=1), 1.0)
        return mx.expand_dims(sequence_log_ratio, axis=1)
    raise ValueError(
        f"Unknown importance sampling level: {importance_sampling_level}. "
        "Possible values are 'token' or 'sequence'."
    )


def generate_grpo(
    model: nn.Module,
    tokenizer,
    prompt_tokens,
    max_tokens: int,
    group_size: int,
    batch_size: int,
    end_token: str,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
):
    """Generate exact sampled token IDs, keeping decode/encode out of training."""
    if not prompt_tokens or min(max_tokens, group_size, batch_size) < 1:
        raise ValueError("Prompts and positive generation sizes are required.")
    sampler = make_sampler(temperature, top_p=top_p, min_p=min_p, top_k=top_k)
    # Do not mutate the tokenizer's global EOS configuration.
    stop_tokens = [[token] for token in tokenizer.eos_token_ids]
    if end_token:
        end_ids = tokenizer.encode(end_token, add_special_tokens=False)
        if end_ids and end_ids not in stop_tokens:
            stop_tokens.append(end_ids)
    was_training = model.training
    model.eval()
    try:
        completions, texts, batch_indices = [], [], []
        for start in range(0, len(prompt_tokens), batch_size):
            indices = [
                idx
                for idx in range(start, min(start + batch_size, len(prompt_tokens)))
                for _ in range(group_size)
            ]
            prompts = [
                (
                    prompt_tokens[idx].tolist()
                    if isinstance(prompt_tokens[idx], mx.array)
                    else list(prompt_tokens[idx])
                )
                for idx in indices
            ]
            generator = BatchGenerator(
                model,
                stop_tokens=stop_tokens,
                sampler=sampler,
                # Bound KV-cache concurrency independently of group expansion.
                completion_batch_size=batch_size,
                prefill_batch_size=batch_size,
            )
            try:
                uids = generator.insert(prompts, [max_tokens] * len(prompts))
                tokens = {uid: [] for uid in uids}
                while responses := generator.next_generated():
                    for response in responses:
                        # Include the sampled stop token in the policy objective.
                        tokens[response.uid].append(response.token)
                for uid in uids:
                    ids = tokens[uid]
                    completions.append(mx.array(ids, dtype=mx.int32))
                    # EOS is not reward text; custom answer delimiters are.
                    text_ids = (
                        ids[:-1] if ids and ids[-1] in tokenizer.eos_token_ids else ids
                    )
                    texts.append(tokenizer.decode(text_ids))
                batch_indices.extend(indices)
            finally:
                generator.close()
        return completions, texts, batch_indices
    finally:
        model.train(was_training)


def calculate_rewards_and_advantages(
    reward_funcs: List[RewardFunctions],
    expanded_prompts: List[str],
    all_completion_texts: List[str],
    expanded_answers: List[str],
    expanded_types: List,
    batch_indices: List[int],
    unique_prompt_indices: List[int],
    reward_weights: Optional[List[float]] = None,
):
    """Evaluate each reward once and normalize groups on the CPU.

    Reward callbacks already return Python values. NumPy avoids building a GPU
    graph of scalar updates, repeated searches, and synchronizations for them.
    None/NaN denotes an inapplicable reward; infinities are errors.
    """
    if not reward_funcs or not all_completion_texts:
        raise ValueError("At least one reward function and completion are required.")
    if reward_weights is not None and len(reward_weights) != len(reward_funcs):
        raise ValueError(
            "Number of reward weights must match number of reward functions"
        )
    weights = np.asarray(
        reward_weights if reward_weights is not None else [1.0] * len(reward_funcs),
        dtype=np.float64,
    )
    if not np.isfinite(weights).all():
        raise ValueError("Reward weights must be finite.")
    columns, metrics = [], {}
    for reward_func in reward_funcs:
        raw = reward_func(
            prompts=expanded_prompts,
            completions=all_completion_texts,
            answer=expanded_answers,
            types=expanded_types,
        )
        if raw is None:
            raw = [None] * len(all_completion_texts)
        if len(raw) != len(all_completion_texts):
            raise ValueError(
                f"{reward_func.__name__} must return one reward per completion."
            )
        values = np.asarray([np.nan if r is None else float(r) for r in raw])
        if np.isinf(values).any():
            raise ValueError(f"{reward_func.__name__} returned an infinite reward.")
        valid = values[~np.isnan(values)]
        name = reward_func.__name__
        metrics[f"{name}_mean"] = float(valid.mean()) if valid.size else 0.0
        metrics[f"{name}_std"] = float(valid.std()) if valid.size else 0.0
        metrics[f"{name}_coverage"] = valid.size / values.size
        columns.append(values)
    rewards = np.stack(columns, axis=1)
    missing = np.isnan(rewards).all(axis=1)
    if missing.any():
        idx = int(np.flatnonzero(missing)[0])
        raise RuntimeError(
            "All reward functions returned None or NaN for completion "
            f"{idx}. At least one valid reward is required."
        )
    rewards = (np.nan_to_num(rewards, nan=0.0) * weights).sum(axis=1)
    if not np.isfinite(rewards).all():
        raise ValueError("Weighted rewards must be finite.")
    groups = {idx: [] for idx in unique_prompt_indices}
    if len(batch_indices) != len(rewards):
        raise ValueError("batch_indices must match the number of rewards.")
    for i, idx in enumerate(batch_indices):
        groups[idx].append(i)
    advantages = np.zeros_like(rewards)
    means, stds = [], []
    for indices in groups.values():
        if not indices:
            raise ValueError("Each prompt group must contain at least one completion.")
        group = rewards[indices]
        mean, std = group.mean(), group.std()
        advantages[indices] = (group - mean) / (std + 1e-4)
        means.append(mean)
        stds.append(std)
    metrics.update(
        total_rewards_mean=rewards.mean(),
        total_rewards_std=rewards.std(),
        grouped_rewards_mean=np.mean(means),
        grouped_rewards_std=np.mean(stds),
    )
    if any(
        not np.isfinite(v) or abs(v) > np.finfo(np.float32).max
        for v in metrics.values()
    ):
        raise ValueError("Reward statistics must be representable in float32.")
    return mx.array(advantages, dtype=mx.float32), {
        key: mx.array(value, dtype=mx.float32) for key, value in metrics.items()
    }


def _rollout_rewards(batch, texts, indices, reward_funcs, reward_weights):
    _, _, prompts, answers, types = batch
    return calculate_rewards_and_advantages(
        reward_funcs=reward_funcs,
        expanded_prompts=[prompts[i] for i in indices],
        all_completion_texts=texts,
        expanded_answers=[answers[i] for i in indices],
        expanded_types=[types[i] if types is not None else None for i in indices],
        batch_indices=indices,
        unique_prompt_indices=sorted(set(indices)),
        reward_weights=reward_weights,
    )


# Prevent exp overflow from extreme policy/reference separation. Expose the
# fraction saturated in metrics so a divergent run is not silently hidden.
_MAX_KL_LOG_RATIO = 20.0


@mx.compile
def _grpo_objective(
    logps,
    ref_logps,
    mask,
    advantages,
    beta,
    epsilon,
    epsilon_high,
    max_tokens,
    importance_sampling_level,
    grpo_loss_type,
):
    """On-policy objective; the fixed reference is used only for KL."""
    logps = mx.where(mask, logps.astype(mx.float32), 0)
    ref_logps = mx.where(mask, mx.stop_gradient(ref_logps.astype(mx.float32)), 0)
    # Each rollout is used once, before updating its generating policy. Retain
    # this zero-valued gradient path; reference probabilities are NOT pi_old.
    log_ratio = logps - mx.stop_gradient(logps)
    log_weights = compute_log_importance_weights(
        log_ratio, mask, importance_sampling_level
    )
    ratio = mx.exp(log_weights)
    clipped = mx.clip(ratio, 1 - epsilon, 1 + epsilon_high)
    advantage = mx.stop_gradient(advantages.astype(mx.float32))[:, None]
    per_token_loss = -mx.minimum(ratio * advantage, clipped * advantage)
    delta = ref_logps - logps
    bounded_delta = mx.minimum(delta, _MAX_KL_LOG_RATIO)
    # expm1 avoids cancellation near zero, where the true KL is quadratic.
    kl = mx.maximum(mx.expm1(bounded_delta) - bounded_delta, 0)
    if beta != 0:
        per_token_loss = per_token_loss + beta * kl
    counts = mask.sum(axis=1)
    total = counts.sum()
    safe_total = mx.maximum(total, 1)
    token_loss = mx.where(mask, per_token_loss, 0).sum(axis=1)
    if grpo_loss_type == "grpo":
        loss = (token_loss / mx.maximum(counts, 1)).mean()
    elif grpo_loss_type == "bnpo":
        loss = token_loss.sum() / safe_total
    elif grpo_loss_type == "dr_grpo":
        loss = token_loss.sum() / (logps.shape[0] * max_tokens)
    else:
        raise ValueError(f"Unknown loss type: {grpo_loss_type}")
    low = (ratio < 1 - epsilon) & (advantage < 0)
    high = (ratio > 1 + epsilon_high) & (advantage > 0)
    metrics = {
        "kl": (mx.where(mask, kl, 0).sum(axis=1) / mx.maximum(counts, 1)).mean(),
        "kl_clip_ratio": ((delta > _MAX_KL_LOG_RATIO) & mask).sum() / safe_total,
        "clip_ratio_low": (low & mask).sum() / safe_total,
        "clip_ratio_high": (high & mask).sum() / safe_total,
        "clip_ratio_total": ((low | high) & mask).sum() / safe_total,
    }
    return loss, total, metrics


def grpo_loss(
    model,
    ref_model,
    batch,
    completions=None,
    completion_texts=None,
    batch_indices=None,
    advantages=None,
    reward_metrics=None,
    beta: float = 0.1,
    epsilon: float = 1e-4,
    epsilon_high: float = None,
    max_tokens: int = 64,
    importance_sampling_level: str = "token",
    grpo_loss_type: str = "grpo",
):
    """Score prompt-conditioned completions and compute a finite masked loss.

    With beta=0 or no reference model, reference scoring is skipped and KL is
    reported as zero. A rollout is consumed once, so importance weights have
    on-policy value one while still carrying the policy gradient.
    """
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive.")
    epsilon_high = epsilon if epsilon_high is None else epsilon_high
    inputs, mask, completion_lengths = _prepare_grpo_inputs(
        batch, completions, batch_indices
    )
    logps = _get_token_logps(model, inputs, mask)
    if beta != 0 and ref_model is not None:
        ref_logps = mx.stop_gradient(_get_token_logps(ref_model, inputs, mask))
    else:
        ref_logps = mx.stop_gradient(logps)
    loss, tokens, metrics = _grpo_objective(
        logps,
        ref_logps,
        mask,
        advantages,
        beta,
        epsilon,
        epsilon_high,
        max_tokens,
        importance_sampling_level,
        grpo_loss_type,
    )
    metrics.update(
        average_generated_tokens=completion_lengths.astype(mx.float32).mean(),
        max_generated_tokens=completion_lengths.max(),
        min_generated_tokens=completion_lengths.min(),
        hit_max_tokens_ratio=(completion_lengths >= max_tokens)
        .astype(mx.float32)
        .mean(),
    )
    metrics.update(reward_metrics or {})
    return loss, tokens, metrics


def iterate_grpo_batches(dataset, batch_size, max_seq_length, train=False):
    has_types = bool(dataset) and isinstance(dataset[0], tuple) and len(dataset[0]) == 5

    if (
        not dataset
        or not isinstance(dataset[0], tuple)
        or (not has_types and len(dataset[0]) != 4)
    ):
        raise ValueError(
            "Dataset must be list of (prompt_tokens, answer_tokens, prompt_str, answer_str[, type]) tuples"
        )

    if batch_size < 1:
        raise ValueError("batch_size must be positive.")

    def length_key(i):
        return len(dataset[i][0]) + len(dataset[i][1])

    idx = sorted(range(len(dataset)), key=length_key)

    if len(dataset) < batch_size:
        raise ValueError(
            f"Dataset must have at least batch_size={batch_size} "
            f"examples but only has {len(dataset)}."
        )

    world = mx.distributed.init()
    step, rank = world.size(), world.rank()
    if batch_size % step != 0:
        raise ValueError("The batch size must be divisible by the number of workers")

    def batch_index_generator():
        for i in range(0, len(idx) - batch_size + 1, batch_size):
            yield idx[i + rank : i + batch_size : step]

    while True:
        indices = (
            np.random.permutation(list(batch_index_generator()))
            if train
            else batch_index_generator()
        )

        for batch_idx in indices:
            current_batch = [dataset[j] for j in batch_idx]

            prompts_tokens = [item[0] for item in current_batch]
            answers_tokens = [item[1] for item in current_batch]
            prompts_text = [item[2] for item in current_batch]
            answers_text = [item[3] for item in current_batch]
            types = [item[4] for item in current_batch] if has_types else None

            yield prompts_tokens, answers_tokens, prompts_text, answers_text, types

        if not train:
            break


def evaluate_grpo(
    model: nn.Module,
    ref_model: Optional[nn.Module],
    dataset,
    tokenizer,
    batch_size,
    num_batches,
    beta: float,
    epsilon: float,
    epsilon_high: float,
    group_size: int,
    max_seq_length: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    reward_funcs: Optional[List[RewardFunctions]] = [
        r1_accuracy_reward_func,
        r1_int_reward_func,
        r1_strict_format_reward_func,
        r1_soft_format_reward_func,
        r1_count_xml,
    ],
    reward_weights: Optional[List[float]] = None,
    loss_fn: callable = grpo_loss,
    iterate_batches: callable = iterate_grpo_batches,
    grpo_loss_type: str = "grpo",
    importance_sampling_level: str = "token",
    end_answer_token: str = "</answer>",
):
    was_training = model.training
    try:
        model.eval()
        if ref_model is not None:
            ref_model.eval()
        all_losses = 0
        ntokens = 0
        all_metrics = None

        index_iterator = iter(range(num_batches)) if num_batches != -1 else iter(int, 1)

        for _, batch in zip(
            index_iterator,
            iterate_batches(
                dataset=dataset,
                batch_size=batch_size,
                max_seq_length=max_seq_length,
            ),
        ):
            prompt_tokens, answer_tokens, prompt_text, answer_text, type_info = batch

            all_completions, all_completion_texts, batch_indices = generate_grpo(
                model=model,
                tokenizer=tokenizer,
                prompt_tokens=prompt_tokens,
                max_tokens=max_tokens,
                group_size=group_size,
                batch_size=batch_size,
                end_token=end_answer_token,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
            )

            advantages, reward_metrics = _rollout_rewards(
                batch, all_completion_texts, batch_indices, reward_funcs, reward_weights
            )

            losses, toks, metrics = loss_fn(
                model=model,
                ref_model=ref_model,
                batch=(
                    prompt_tokens,
                    answer_tokens,
                    prompt_text,
                    answer_text,
                    type_info,
                ),
                completions=all_completions,
                completion_texts=all_completion_texts,
                batch_indices=batch_indices,
                advantages=advantages,
                reward_metrics=reward_metrics,
                beta=beta,
                epsilon=epsilon,
                epsilon_high=epsilon_high,
                importance_sampling_level=importance_sampling_level,
                grpo_loss_type=grpo_loss_type,
                max_tokens=max_tokens,
            )

            del all_completions, all_completion_texts, batch_indices
            del advantages, reward_metrics

            all_losses += losses * toks
            ntokens += toks

            if all_metrics is None:
                all_metrics = {k: v * toks for k, v in metrics.items()}
            else:
                for k, v in metrics.items():
                    all_metrics[k] += v * toks

            mx.eval(all_losses, ntokens, *all_metrics.values())

        if all_metrics is None:
            raise ValueError("Evaluation requires at least one batch.")
        all_losses = mx.distributed.all_sum(all_losses, stream=mx.cpu)
        ntokens = mx.distributed.all_sum(ntokens, stream=mx.cpu)
        all_metrics = {k: mx.distributed.all_sum(v) for k, v in all_metrics.items()}

        avg_metrics = {
            k: (v / mx.maximum(ntokens, 1)).item() for k, v in all_metrics.items()
        }
        avg_loss = (all_losses / mx.maximum(ntokens, 1)).item()

        return avg_loss, ntokens, avg_metrics
    finally:
        model.train(was_training)


def _grpo_value_and_grad(loss_value_and_grad, model, chunk_size, **kwargs):
    """Accumulate completion microbatches without retaining all rollout graphs.

    Rewards/advantages are computed over complete groups before this function.
    Weight each microbatch by the loss's actual denominator so chunking does
    not change the objective for unequal completion lengths or empty rows.
    """
    completions = kwargs.pop("completions")
    texts = kwargs.pop("completion_texts")
    indices = kwargs.pop("batch_indices")
    advantages = kwargs.pop("advantages")
    sample_count = len(completions)
    token_count = sum(c.size for c in completions)
    loss_type = kwargs["grpo_loss_type"]
    total_loss, total_tokens, all_metrics, accumulated = 0, 0, None, None
    for start in range(0, sample_count, chunk_size):
        stop = min(start + chunk_size, sample_count)
        row_weight = (stop - start) / sample_count
        token_weight = sum(c.size for c in completions[start:stop]) / max(
            token_count, 1
        )
        weight = token_weight if loss_type == "bnpo" else row_weight
        (loss, tokens, metrics), grads = loss_value_and_grad(
            model,
            completions=completions[start:stop],
            completion_texts=texts[start:stop],
            batch_indices=indices[start:stop],
            advantages=advantages[start:stop],
            **kwargs,
        )
        grads = tree_map(lambda g: g * weight, grads)
        accumulated = (
            grads
            if accumulated is None
            else tree_map(lambda a, b: a + b, accumulated, grads)
        )
        total_loss = total_loss + loss * weight
        total_tokens = total_tokens + tokens
        weighted_metrics = {
            key: value
            * (
                token_weight
                if key.startswith("clip_ratio") or key == "kl_clip_ratio"
                else row_weight
            )
            for key, value in metrics.items()
        }
        if all_metrics is None:
            all_metrics = weighted_metrics
            all_metrics["max_generated_tokens"] = metrics["max_generated_tokens"]
            all_metrics["min_generated_tokens"] = metrics["min_generated_tokens"]
        else:
            for key, value in weighted_metrics.items():
                if key == "max_generated_tokens":
                    all_metrics[key] = mx.maximum(all_metrics[key], metrics[key])
                elif key == "min_generated_tokens":
                    all_metrics[key] = mx.minimum(all_metrics[key], metrics[key])
                else:
                    all_metrics[key] = all_metrics[key] + value
        del grads, loss, tokens, metrics, weighted_metrics
        # A real evaluation boundary releases each microbatch's activations
        # before building the next graph. Cache clearing cannot achieve this.
        mx.eval(total_loss, total_tokens, all_metrics, accumulated)
    return (total_loss, total_tokens, all_metrics), accumulated


def train_grpo(
    model: nn.Module,
    ref_model: Optional[nn.Module],
    tokenizer,
    optimizer,
    train_dataset,
    val_dataset: Optional[Any] = None,
    reward_funcs: Optional[List[RewardFunctions]] = [
        r1_accuracy_reward_func,
        r1_int_reward_func,
        r1_strict_format_reward_func,
        r1_soft_format_reward_func,
        r1_count_xml,
    ],
    args: GRPOTrainingArgs = GRPOTrainingArgs(),
    loss_fn: callable = grpo_loss,
    iterate_batches: callable = iterate_grpo_batches,
    training_callback: TrainingCallback = None,
    end_answer_token: str = "</answer>",
):
    if model_uses_recurrence(model):
        enable_memory_safe_recurrences(chunk_size=args.recurrence_chunk_size)
    world = mx.distributed.init()
    world_size = world.size()
    rank = world.rank()
    if world_size > 1:
        tqdm.write(f"Node {rank} of {world_size}")

    if args.grad_checkpoint:
        grad_checkpoint(model.layers[0])

    grad_accum_steps = args.gradient_accumulation_steps
    if grad_accum_steps < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1")

    state = [model.state, optimizer.state, mx.random.state]

    def step(
        batch,
        all_completions,
        all_completion_texts,
        batch_indices,
        advantages,
        reward_metrics,
        prev_grad,
        do_update,
        accumulation_count,
    ):
        prompt_tokens, answer_tokens, prompt_text, answer_text, type_info = batch

        if loss_fn is grpo_loss:
            # Group size no longer multiplies peak scoring/backprop memory.
            compute_grad = lambda model, **kwargs: _grpo_value_and_grad(
                loss_value_and_grad, model, max(len(prompt_tokens), 1), **kwargs
            )
        else:
            compute_grad = loss_value_and_grad
        (lvalue, toks, metrics), grad = compute_grad(
            model,
            batch=(prompt_tokens, answer_tokens, prompt_text, answer_text, type_info),
            completions=all_completions,
            completion_texts=all_completion_texts,
            batch_indices=batch_indices,
            advantages=advantages,
            reward_metrics=reward_metrics,
            beta=args.beta,
            epsilon=args.epsilon,
            epsilon_high=args.epsilon_high,
            ref_model=ref_model,
            grpo_loss_type=args.grpo_loss_type,
            importance_sampling_level=args.importance_sampling_level,
            max_tokens=args.max_completion_length,
        )

        del all_completions, all_completion_texts, batch_indices
        del advantages, reward_metrics

        if prev_grad is not None:
            grad = tree_map(lambda x, y: x + y, grad, prev_grad)

        # Validate before mutating optimizer state; finite loss alone does not
        # guarantee finite half-precision gradients.
        finite = mx.isfinite(lvalue) & mx.all(
            mx.stack([mx.all(mx.isfinite(value)) for _, value in tree_flatten(grad)])
        )
        if not finite.item():
            raise FloatingPointError(
                "Non-finite GRPO loss or gradients; optimizer update aborted."
            )

        if do_update:
            grad = average_gradients(grad)
            if grad_accum_steps > 1:
                grad = tree_map(lambda x: x / accumulation_count, grad)
            optimizer.update(model, grad)
            grad = None

        return lvalue, toks, metrics, grad

    if ref_model is not None:
        ref_model.eval()
    loss_value_and_grad = nn.value_and_grad(model, loss_fn)

    model.train()
    losses = 0
    n_tokens = 0
    steps = 0
    trained_tokens = 0
    accumulated_metrics = {
        "total_rewards_mean": 0,
        "total_rewards_std": 0,
        "grouped_rewards_mean": 0,
        "grouped_rewards_std": 0,
        "kl": 0,
        "kl_clip_ratio": 0,
        "average_generated_tokens": 0,
        "max_generated_tokens": 0,
        "min_generated_tokens": 0,
        "hit_max_tokens_ratio": 0,
        "clip_ratio_low": 0,
        "clip_ratio_high": 0,
        "clip_ratio_total": 0,
    }
    grad_accum = None
    for reward_func in reward_funcs:
        func_name = reward_func.__name__
        accumulated_metrics[f"{func_name}_mean"] = 0
        accumulated_metrics[f"{func_name}_std"] = 0
        accumulated_metrics[f"{func_name}_coverage"] = 0

    start = time.perf_counter()
    pbar = tqdm(range(1, args.iters + 1), desc="Training", disable=rank != 0)
    batches = iter(
        iterate_batches(
            dataset=train_dataset,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
            train=True,
        )
    )
    for it in pbar:
        batch = next(batches)

        if (
            val_dataset is not None
            and len(val_dataset) > 0
            and (it == 1 or it % args.steps_per_eval == 0 or it == args.iters)
        ):
            stop = time.perf_counter()
            val_loss, val_ntokens, val_metrics = evaluate_grpo(
                model=model,
                dataset=val_dataset,
                loss_fn=loss_fn,
                ref_model=ref_model,
                reward_funcs=reward_funcs,
                reward_weights=args.reward_weights,
                importance_sampling_level=args.importance_sampling_level,
                tokenizer=tokenizer,
                group_size=args.group_size,
                batch_size=args.batch_size,
                num_batches=args.val_batches,
                max_seq_length=args.max_seq_length,
                max_tokens=args.max_completion_length,
                beta=args.beta,
                epsilon=args.epsilon,
                epsilon_high=args.epsilon_high,
                iterate_batches=iterate_batches,
                grpo_loss_type=args.grpo_loss_type,
                end_answer_token=end_answer_token,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                min_p=args.min_p,
            )
            val_time = time.perf_counter() - stop
            if rank == 0:
                tqdm.write(
                    f"Iter {it}: "
                    f"Val loss {val_loss:.3f}, "
                    f"Val took {val_time:.3f}s"
                )

            if training_callback is not None:
                val_info = {
                    "iteration": it,
                    "val_loss": val_loss,
                    "val_time": val_time,
                }
                training_callback.on_val_loss_report(val_info)

            model.train()
            start = time.perf_counter()

        prompt_tokens, answer_tokens, prompt_text, answer_text, type_info = batch

        all_completions, all_completion_texts, batch_indices = generate_grpo(
            model=model,
            tokenizer=tokenizer,
            prompt_tokens=prompt_tokens,
            max_tokens=args.max_completion_length,
            group_size=args.group_size,
            batch_size=args.batch_size,
            end_token=end_answer_token,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
        )

        advantages, reward_metrics = _rollout_rewards(
            batch,
            all_completion_texts,
            batch_indices,
            reward_funcs,
            args.reward_weights,
        )

        lvalue, toks, metrics, grad_accum = step(
            batch,
            all_completions,
            all_completion_texts,
            batch_indices,
            advantages,
            reward_metrics,
            grad_accum,
            it % grad_accum_steps == 0 or it == args.iters,
            (it - 1) % grad_accum_steps + 1,
        )
        losses += lvalue
        n_tokens += toks
        steps += 1

        for k, v in metrics.items():
            accumulated_metrics[k] += v

        _acc = [v for v in accumulated_metrics.values() if isinstance(v, mx.array)]
        mx.eval(state, losses, n_tokens, grad_accum, *_acc)

        if it % args.steps_per_report == 0 or it == args.iters:
            stop = time.perf_counter()

            train_loss = mx.distributed.all_sum(losses).item() / (steps * world_size)
            reduced_metrics = {
                k: mx.distributed.all_sum(mx.array(v)) / (steps * world_size)
                for k, v in accumulated_metrics.items()
            }
            mx.eval(reduced_metrics)
            avg_metrics = {k: v.item() for k, v in reduced_metrics.items()}
            n_tokens = mx.distributed.all_sum(n_tokens).item()
            learning_rate = optimizer.learning_rate.item()
            it_sec = steps / (stop - start)
            tokens_sec = float(n_tokens) / (stop - start)
            trained_tokens += n_tokens
            peak_mem = mx.get_peak_memory() / 1e9

            if rank == 0:
                pbar.set_postfix(
                    {
                        "loss": f"{train_loss:.3f}",
                        "it/s": f"{it_sec:.3f}",
                    }
                )
                reward_metrics_str = ""
                for reward_func in reward_funcs:
                    func_name = reward_func.__name__
                    mean_key = f"{func_name}_mean"
                    std_key = f"{func_name}_std"
                    cov_key = f"{func_name}_coverage"

                    if mean_key in avg_metrics:
                        display_name = func_name.replace("_reward_func", "").replace(
                            "r1_", ""
                        )
                        reward_metrics_str += (
                            f"  • {display_name}: "
                            f"μ={avg_metrics[mean_key]:.3f}, "
                            f"σ={avg_metrics[std_key]:.3f}, "
                            f"cov={avg_metrics[cov_key]:.2%}\n"
                        )
                tqdm.write(
                    f"\n{'='*80}\n"
                    f"Iter {it}:\n"
                    f"{'-'*80}\n"
                    f"Loss: {train_loss:.3f}\n"
                    f"Total Rewards:  μ={avg_metrics['total_rewards_mean']:.3f}, "
                    f"σ={avg_metrics['total_rewards_std']:.3f}\n"
                    f"Group Rewards:  μ={avg_metrics['grouped_rewards_mean']:.3f}, "
                    f"σ={avg_metrics['grouped_rewards_std']:.3f}\n"
                    f"KL Divergence: {avg_metrics['kl']:.12f}\n"
                    f"KL saturation: {avg_metrics['kl_clip_ratio']:.2%}\n"
                    f"{'-'*80}\n"
                    f"Generation Stats:\n"
                    f"  • Avg tokens: {avg_metrics['average_generated_tokens']:.1f}\n"
                    f"  • Min tokens: {avg_metrics['min_generated_tokens']:.0f}\n"
                    f"  • Max tokens: {avg_metrics['max_generated_tokens']:.0f} "
                    f"(limit: {args.max_completion_length})\n"
                    f"  • Hit limit: {avg_metrics['hit_max_tokens_ratio']:.1%}\n"
                    f"{'-'*80}\n"
                    f"Individual Reward Functions:\n"
                    f"{reward_metrics_str}"
                    f"{'-'*80}\n"
                    f"Clipping:  low={avg_metrics['clip_ratio_low']:.3f}, "
                    f"high={avg_metrics['clip_ratio_high']:.3f}, "
                    f"total={avg_metrics['clip_ratio_total']:.3f}\n"
                    f"Learning Rate: {learning_rate:.4e}\n"
                    f"Speed: {it_sec:.3f} it/s, {tokens_sec:.1f} tok/s\n"
                    f"Memory: {peak_mem:.3f}GB\n"
                    f"{'='*80}\n"
                )

            if training_callback is not None:
                train_info = {
                    "iteration": it,
                    "train_loss": train_loss,
                    **{f"train_{k}": v for k, v in avg_metrics.items()},
                    "learning_rate": learning_rate,
                    "iterations_per_second": it_sec,
                    "tokens_per_second": tokens_sec,
                    "trained_tokens": trained_tokens,
                    "peak_memory": peak_mem,
                }
                training_callback.on_train_loss_report(train_info)

            losses = 0
            n_tokens = 0
            steps = 0
            accumulated_metrics = {k: 0 for k in accumulated_metrics}
            start = time.perf_counter()

        if it % args.steps_per_save == 0:
            adapter_weights = dict(tree_flatten(model.trainable_parameters()))
            mx.save_safetensors(str(args.adapter_file), adapter_weights)
            checkpoint = (
                Path(args.adapter_file).parent / f"{it:07d}_adapters.safetensors"
            )
            mx.save_safetensors(str(checkpoint), adapter_weights)
            tqdm.write(
                f"\n"
                f"Iter {it}: Saved adapter weights to "
                f"{args.adapter_file} and {checkpoint}."
            )

    adapter_weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(args.adapter_file), adapter_weights)
    tqdm.write(f"Saved final weights to {args.adapter_file}.")
