"""Shared DPO/CPO scoring and preference evaluation helpers."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_map

from .long_context import iter_cached_sft_chunks
from .sft_trainer import reset_prompt_cache


def get_token_scores(model, x, mask, cache=None):
    inputs, targets = x[:, :-1], x[:, 1:]
    logits = model(inputs, cache=cache).astype(mx.float32)
    return -nn.losses.cross_entropy(logits, targets) * mask[:, :-1]


def compute_score(scores, mask, loss_type):
    token_count = mask.sum(-1)
    return scores.sum(-1) / token_count if loss_type == "ipo" else scores.sum(-1)


def iterate_preference_batches(dataset, batch_size, max_seq_length, train=False):
    """Yield length-sorted, dynamically padded DPO/CPO sequence pairs."""
    idx = sorted(range(len(dataset)), key=lambda idx: len(dataset[idx]["chosen"]))
    step = mx.distributed.init().size()
    if batch_size % step != 0:
        raise ValueError("Batch size must be divisible by workers")

    batch_idx = [
        idx[i : i + batch_size : step]
        for i in range(0, len(idx) - batch_size + 1, batch_size)
    ]
    while True:
        indices = (
            np.random.permutation(len(batch_idx)) if train else range(len(batch_idx))
        )
        for i in indices:
            batch = [dataset[j] for j in batch_idx[i]]
            chosen_lengths = [len(x["chosen"]) for x in batch]
            rejected_lengths = [len(x["rejected"]) for x in batch]
            max_length = min(
                max(max(chosen_lengths), max(rejected_lengths)), max_seq_length
            )
            shape = (batch_size // step, max_length)
            chosen_arr = np.zeros(shape, np.int32)
            rejected_arr = np.zeros(shape, np.int32)
            chosen_masks = np.zeros(shape, np.float32)
            rejected_masks = np.zeros(shape, np.float32)

            for j in range(batch_size // step):
                chosen_length = min(chosen_lengths[j], max_seq_length)
                rejected_length = min(rejected_lengths[j], max_seq_length)
                chosen_arr[j, :chosen_length] = batch[j]["chosen"][:chosen_length]
                rejected_arr[j, :rejected_length] = batch[j]["rejected"][
                    :rejected_length
                ]
                chosen_masks[j, :chosen_length] = 1.0
                rejected_masks[j, :rejected_length] = 1.0

            yield (
                mx.array(chosen_arr),
                mx.array(rejected_arr),
                mx.array(chosen_masks),
                mx.array(rejected_masks),
            )
        if not train:
            break


def compute_scores_chunked(model, cache, tokens, masks, seq_step_size):
    """Sum scores without retaining a full-sequence gradient graph."""
    score_sum = mx.zeros((tokens.shape[0],))
    if cache is not None:
        reset_prompt_cache(cache)
    for start, end in iter_cached_sft_chunks(tokens.shape[1], seq_step_size):
        score_sum += get_token_scores(
            model, tokens[:, start:end], masks[:, start:end], cache=cache
        ).sum(-1)
    return score_sum


def accumulate_score_gradients(
    model, cache, tokens, masks, weights, seq_step_size, grad_accum=None
):
    """Backpropagate weighted scores one cached chunk at a time."""
    reset_prompt_cache(cache)
    for start, end in iter_cached_sft_chunks(tokens.shape[1], seq_step_size):
        chunk = tokens[:, start:end]
        chunk_mask = masks[:, start:end]

        def local_loss_fn(model):
            local_sum = get_token_scores(model, chunk, chunk_mask, cache=cache).sum(-1)
            return (local_sum * weights).sum()

        grad = mx.grad(local_loss_fn)(model)
        grad_accum = (
            grad
            if grad_accum is None
            else tree_map(lambda x, y: x + y, grad_accum, grad)
        )
        mx.eval(grad_accum)
    return grad_accum


def evaluate_preference_batches(batches, num_batches, loss_fn, *, weight_rewards):
    """Reduce token-weighted losses/metrics across preference batches.

    ORPO token-weights rewards; DPO/CPO retain their batch reward sums.
    """
    all_losses = 0
    all_rewards = mx.zeros((2,))
    all_metrics = None
    ntokens = 0
    indices = iter(range(num_batches)) if num_batches != -1 else iter(int, 1)
    for _, batch in zip(indices, batches):
        loss, reward, toks, metrics = loss_fn(*batch)
        all_losses += loss * toks
        all_rewards += reward * toks if weight_rewards else reward
        ntokens += toks
        if all_metrics is None:
            all_metrics = {k: v * toks for k, v in metrics.items()}
        else:
            for k, v in metrics.items():
                all_metrics[k] += v * toks
        mx.eval(all_losses, all_rewards, ntokens, *all_metrics.values())

    all_losses = mx.distributed.all_sum(all_losses)
    all_rewards = mx.distributed.all_sum(all_rewards)
    ntokens = mx.distributed.all_sum(ntokens)
    all_metrics = {k: mx.distributed.all_sum(v) for k, v in all_metrics.items()}
    avg_metrics = {k: (v / ntokens).item() for k, v in all_metrics.items()}
    return (
        (all_losses / ntokens).item(),
        (all_rewards / ntokens).tolist(),
        ntokens,
        avg_metrics,
    )
