"""Final Token Preference Optimization (FTPO) for Antidoom datasets."""

import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.nn.utils import average_gradients
from mlx.utils import tree_flatten, tree_map
from mlx_lm.tuner.callbacks import TrainingCallback
from tqdm import tqdm

from .sft_trainer import SFTTrainingArgs, grad_checkpoint


@dataclass
class FTPOTrainingArgs(SFTTrainingArgs):
    lambda_mse_target: float = field(default=0.05)
    tau_mse_target: float = field(default=1.0)
    lambda_mse: float = field(default=0.4)
    clip_epsilon_logits: float = field(default=2.0)


def ftpo_loss(
    policy_logits: mx.array,
    reference_logits: mx.array,
    chosen_ids: mx.array,
    chosen_mask: mx.array,
    rejected_ids: mx.array,
    lambda_mse_target: float = 0.05,
    tau_mse_target: float = 1.0,
    lambda_mse: float = 0.4,
    clip_epsilon_logits: float = 2.0,
):
    """Liquid AI's FTPO objective over the next-token distribution."""
    if clip_epsilon_logits <= 0:
        raise ValueError("clip_epsilon_logits must be positive")

    chosen_logits = mx.take_along_axis(policy_logits, chosen_ids, axis=-1)
    rejected_logits = mx.take_along_axis(policy_logits, rejected_ids[:, None], axis=-1)
    delta = chosen_logits - rejected_logits
    weights = (
        mx.clip((clip_epsilon_logits - delta) / clip_epsilon_logits, 0.0, 1.0)
        * chosen_mask
    )
    preference = nn.softplus(clip_epsilon_logits - delta)
    chosen_counts = mx.maximum(chosen_mask.sum(-1), 1.0)
    pref_loss = mx.mean((preference * weights).sum(-1) / chosen_counts)

    diff = policy_logits - mx.stop_gradient(reference_logits)
    diff_sq = diff * diff
    chosen_diff_sq = mx.take_along_axis(diff_sq, chosen_ids, axis=-1)
    rejected_diff_sq = mx.take_along_axis(diff_sq, rejected_ids[:, None], axis=-1)
    target_count = chosen_mask.sum() + policy_logits.shape[0]
    non_target_count = policy_logits.size - target_count
    non_target_sq = (
        diff_sq.sum() - (chosen_diff_sq * chosen_mask).sum() - rejected_diff_sq.sum()
    )
    mse_elem = non_target_sq / mx.maximum(non_target_count, 1.0)
    excess = mx.maximum(mx.abs(diff) - tau_mse_target, 0.0)
    excess_sq = excess * excess
    chosen_excess_sq = mx.take_along_axis(excess_sq, chosen_ids, axis=-1)
    rejected_excess_sq = mx.take_along_axis(excess_sq, rejected_ids[:, None], axis=-1)
    mse_target = (
        (chosen_excess_sq * chosen_mask).sum() + rejected_excess_sq.sum()
    ) / mx.maximum(target_count, 1.0)
    loss = pref_loss + lambda_mse * mse_elem + lambda_mse_target * mse_target

    active = chosen_mask.astype(mx.bool_)
    denom = mx.maximum(chosen_mask.sum(), 1.0)
    metrics = {
        "pref_loss": pref_loss,
        "mse_elem": mse_elem,
        "mse_tgt_tokenwise": mse_target,
        "chosen_win": (((delta > 0) & active).astype(mx.float32).sum() / denom),
        "margin_win": (
            ((delta >= clip_epsilon_logits) & active).astype(mx.float32).sum() / denom
        ),
        "mean_delta": (delta * chosen_mask).sum() / denom,
        "active_weight": weights.sum() / denom,
    }
    return loss, mx.array(policy_logits.shape[0]), metrics


def iterate_ftpo_batches(dataset, batch_size, max_seq_length, train=False):
    if len(dataset) < batch_size:
        raise ValueError(
            f"FTPO dataset must contain at least batch_size={batch_size} rows"
        )
    order = sorted(range(len(dataset)), key=lambda i: len(dataset[i]["prompt_ids"]))
    batches = [
        order[i : i + batch_size]
        for i in range(0, len(order) - batch_size + 1, batch_size)
    ]
    while True:
        batch_order = (
            np.random.permutation(len(batches)) if train else range(len(batches))
        )
        for batch_index in batch_order:
            rows = [dataset[i] for i in batches[batch_index]]
            lengths = [len(row["prompt_ids"]) for row in rows]
            if any(length == 0 or length > max_seq_length for length in lengths):
                raise ValueError(
                    "FTPO prompts must be non-empty and fit max_seq_length"
                )
            width = max(lengths)
            max_chosen = max(len(row["chosen_ids"]) for row in rows)
            prompts = np.zeros((batch_size, width), dtype=np.int32)
            chosen = np.zeros((batch_size, max_chosen), dtype=np.int32)
            chosen_mask = np.zeros((batch_size, max_chosen), dtype=np.float32)
            rejected = np.zeros((batch_size,), dtype=np.int32)
            for j, row in enumerate(rows):
                prompts[j, : lengths[j]] = row["prompt_ids"]
                count = len(row["chosen_ids"])
                if count == 0:
                    raise ValueError("Every FTPO row needs at least one chosen token")
                chosen[j, :count] = row["chosen_ids"]
                chosen_mask[j, :count] = 1.0
                rejected[j] = row["rejected_token_id"]
            yield tuple(
                map(
                    mx.array,
                    (prompts, np.asarray(lengths), chosen, chosen_mask, rejected),
                )
            )
        if not train:
            break


def _last_logits(model, prompts, lengths):
    logits = model(prompts).astype(mx.float32)
    indices = (lengths - 1).astype(mx.int32)[:, None, None]
    indices = mx.broadcast_to(indices, (logits.shape[0], 1, logits.shape[-1]))
    return mx.take_along_axis(logits, indices, axis=1).squeeze(1)


def _batch_loss(model, ref_model, batch, args):
    prompts, lengths, chosen, chosen_mask, rejected = batch
    policy_logits = _last_logits(model, prompts, lengths)
    reference_logits = mx.stop_gradient(_last_logits(ref_model, prompts, lengths))
    return ftpo_loss(
        policy_logits,
        reference_logits,
        chosen,
        chosen_mask,
        rejected,
        args.lambda_mse_target,
        args.tau_mse_target,
        args.lambda_mse,
        args.clip_epsilon_logits,
    )


def evaluate_ftpo(
    model, ref_model, dataset, batch_size, num_batches, max_seq_length, args
):
    model.eval()
    totals = None
    total_loss = mx.array(0.0)
    count = 0
    iterator = iterate_ftpo_batches(dataset, batch_size, max_seq_length)
    for i, batch in enumerate(iterator):
        if num_batches != -1 and i >= num_batches:
            break
        loss, _, metrics = _batch_loss(model, ref_model, batch, args)
        total_loss += loss
        totals = (
            metrics
            if totals is None
            else {k: totals[k] + v for k, v in metrics.items()}
        )
        count += 1
    if count == 0:
        raise ValueError("No FTPO evaluation batches available")
    mx.eval(total_loss, *totals.values())
    return total_loss.item() / count, {k: v / count for k, v in totals.items()}


def train_ftpo(
    model,
    ref_model,
    optimizer,
    train_dataset,
    val_dataset: Optional[Any] = None,
    args: FTPOTrainingArgs = FTPOTrainingArgs(),
    training_callback: TrainingCallback = None,
):
    if ref_model is None:
        raise ValueError("FTPO requires a frozen reference model")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1")
    if args.grad_checkpoint:
        grad_checkpoint(model.layers[0])

    world = mx.distributed.init()
    state = [model.state, optimizer.state, mx.random.state]
    value_and_grad = nn.value_and_grad(
        model, lambda m, batch: _batch_loss(m, ref_model, batch, args)
    )

    @partial(mx.compile, inputs=state + [ref_model.state], outputs=state)
    def step(batch, previous_grad, update):
        (loss, samples, metrics), grad = value_and_grad(model, batch)
        if previous_grad is not None:
            grad = tree_map(lambda x, y: x + y, grad, previous_grad)
        if update:
            grad = average_gradients(grad)
            if args.gradient_accumulation_steps > 1:
                grad = tree_map(lambda x: x / args.gradient_accumulation_steps, grad)
            optimizer.update(model, grad)
            grad = None
        return loss, samples, metrics, grad

    model.train()
    iterator = iterate_ftpo_batches(
        train_dataset, args.batch_size, args.max_seq_length, train=True
    )
    accumulated_grad = None
    loss_sum = mx.array(0.0)
    metric_sums = None
    report_steps = 0
    start = time.perf_counter()
    rank = world.rank()
    pbar = tqdm(range(1, args.iters + 1), desc="FTPO", disable=rank != 0)
    for iteration in pbar:
        if val_dataset and (iteration == 1 or iteration % args.steps_per_eval == 0):
            val_loss, val_metrics = evaluate_ftpo(
                model,
                ref_model,
                val_dataset,
                args.batch_size,
                args.val_batches,
                args.max_seq_length,
                args,
            )
            if training_callback:
                training_callback.on_val_loss_report(
                    {
                        "iteration": iteration,
                        "val_loss": val_loss,
                        **{f"val_{k}": v for k, v in val_metrics.items()},
                    }
                )
            model.train()
        loss, _, metrics, accumulated_grad = step(
            next(iterator),
            accumulated_grad,
            iteration % args.gradient_accumulation_steps == 0,
        )
        loss_sum += loss
        metric_sums = (
            metrics
            if metric_sums is None
            else {k: metric_sums[k] + v for k, v in metrics.items()}
        )
        report_steps += 1
        mx.eval(state, loss_sum, accumulated_grad, *metric_sums.values())
        if iteration % args.steps_per_report == 0 or iteration == args.iters:
            elapsed = time.perf_counter() - start
            avg_loss = mx.distributed.all_sum(loss_sum).item() / (
                report_steps * world.size()
            )
            avg_metrics = {
                k: mx.distributed.all_sum(v).item() / (report_steps * world.size())
                for k, v in metric_sums.items()
            }
            if rank == 0:
                tqdm.write(
                    f"Iter {iteration}: loss {avg_loss:.4f}, chosen_win {avg_metrics['chosen_win']:.3f}, margin_win {avg_metrics['margin_win']:.3f}, it/s {report_steps / elapsed:.3f}"
                )
            if training_callback:
                training_callback.on_train_loss_report(
                    {
                        "iteration": iteration,
                        "train_loss": avg_loss,
                        **{f"train_{k}": v for k, v in avg_metrics.items()},
                    }
                )
            loss_sum, metric_sums, report_steps, start = (
                mx.array(0.0),
                None,
                0,
                time.perf_counter(),
            )
        if iteration % args.steps_per_save == 0:
            mx.save_safetensors(
                str(args.adapter_file), dict(tree_flatten(model.trainable_parameters()))
            )

    mx.save_safetensors(
        str(args.adapter_file), dict(tree_flatten(model.trainable_parameters()))
    )
