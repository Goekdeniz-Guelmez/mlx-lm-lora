"""Reward evaluation and bounded scoring shared by online policy trainers."""

import mlx.core as mx
import numpy as np
from mlx.utils import tree_map


def evaluate_rewards(
    reward_funcs,
    prompts,
    completion_texts,
    answers,
    types,
    reward_weights=None,
):
    """Call each reward once and aggregate Python values on the CPU.

    None/NaN denotes an inapplicable reward; every completion needs at least
    one applicable reward. NumPy avoids GPU scalar updates and synchronization.
    Leave algorithm-specific normalization and MLX conversion to the caller.
    """
    if not reward_funcs or not completion_texts:
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
            prompts=prompts,
            completions=completion_texts,
            answer=answers,
            types=types,
        )
        if raw is None:
            raw = [None] * len(completion_texts)
        if len(raw) != len(completion_texts):
            raise ValueError(
                f"{reward_func.__name__} must return one reward per completion."
            )
        values = np.asarray(
            [np.nan if value is None else float(value) for value in raw]
        )
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
    return rewards, metrics


def rollout_microbatches(
    compute,
    model,
    chunk_size,
    *,
    sample_keys,
    with_grad=False,
    token_weighted_loss=False,
    token_weighted_metrics=False,
    **kwargs,
):
    """Score slices, preserving denominators and releasing each chunk's graph.

    Sample fields are sliced together; other loss arguments stay unchanged.
    Most losses/metrics average over rows. Token-normalized losses and clipping
    fractions instead use completion-token counts, including empty rows.
    """
    samples = {key: kwargs.pop(key) for key in sample_keys}
    completions = samples["completions"]
    sample_count = len(completions)
    token_weighted = token_weighted_loss or token_weighted_metrics
    token_count = sum(c.size for c in completions) if token_weighted else 0
    total_loss, total_tokens, all_metrics, accumulated = 0, 0, None, None
    for start in range(0, sample_count, chunk_size):
        stop = min(start + chunk_size, sample_count)
        row_weight = (stop - start) / sample_count
        token_weight = (
            sum(c.size for c in completions[start:stop]) / max(token_count, 1)
            if token_weighted
            else row_weight
        )
        weight = token_weight if token_weighted_loss else row_weight
        result = compute(
            model,
            **{key: values[start:stop] for key, values in samples.items()},
            **kwargs,
        )
        if with_grad:
            (loss, tokens, metrics), grads = result
            grads = tree_map(lambda value, weight=weight: value * weight, grads)
            accumulated = (
                grads
                if accumulated is None
                else tree_map(lambda left, right: left + right, accumulated, grads)
            )
            del grads
        else:
            loss, tokens, metrics = result
        del result
        total_loss = total_loss + loss * weight
        total_tokens = total_tokens + tokens
        weighted_metrics = {
            key: value
            * (
                token_weight
                if token_weighted_metrics
                and (key.startswith("clip_ratio") or key == "kl_clip_ratio")
                else row_weight
            )
            for key, value in metrics.items()
            if key not in {"max_generated_tokens", "min_generated_tokens"}
        }
        if all_metrics is None:
            all_metrics = weighted_metrics
            all_metrics["max_generated_tokens"] = metrics["max_generated_tokens"]
            all_metrics["min_generated_tokens"] = metrics["min_generated_tokens"]
        else:
            for key, value in weighted_metrics.items():
                all_metrics[key] = all_metrics[key] + value
            all_metrics["max_generated_tokens"] = mx.maximum(
                all_metrics["max_generated_tokens"], metrics["max_generated_tokens"]
            )
            all_metrics["min_generated_tokens"] = mx.minimum(
                all_metrics["min_generated_tokens"], metrics["min_generated_tokens"]
            )
        del loss, tokens, metrics, weighted_metrics
        # Evaluation releases activations before building the next graph;
        # clearing the allocator cache cannot replace this boundary.
        mx.eval(total_loss, total_tokens, all_metrics, accumulated)
    result = total_loss, total_tokens, all_metrics
    return (result, accumulated) if with_grad else result
