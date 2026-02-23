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
from mlx_lm.models.cache import KVCache, make_prompt_cache
from mlx_lm.tuner.callbacks import TrainingCallback
from tqdm import tqdm

from .sft_trainer import SFTTrainingArgs, grad_checkpoint


def reset_prompt_cache(cache):
    for e, c in enumerate(cache):
        if isinstance(c, KVCache):
            cache[e] = KVCache()
        else:
            raise ValueError("Unsupported cache")


@dataclass
class ORPOTrainingArgs(SFTTrainingArgs):
    beta: float = field(
        default=0.1, metadata={"help": "Temperature parameter for ORPO training."}
    )
    reward_scaling: float = field(
        default=1.0,
        metadata={"help": "Reward scaling factor for ORPO training, not implemented."},
    )


def get_logps(model, tokens, mask, cache=None):
    inputs = tokens[:, :-1]
    targets = tokens[:, 1:]
    logits = model(inputs, cache=cache)
    # Clip log_probs to avoid -inf and NaN stability issues
    log_probs = -nn.losses.cross_entropy(logits, targets, reduction="none")
    log_probs = mx.clip(log_probs, -1000.0, 0.0)
    
    mask = mask[:, :-1]
    seq_lengths = mask.sum(-1)
    logp_sum = (log_probs * mask).sum(-1)
    safe_seq_lengths = mx.where(seq_lengths > 0, seq_lengths, mx.array(1.0))
    logp_seq_avg = mx.where(seq_lengths > 0, logp_sum / safe_seq_lengths, mx.array(0.0))
    mask_sum = mask.sum()
    safe_mask_sum = mx.where(mask_sum > 0, mask_sum, mx.array(1.0))
    logits_mean = mx.where(mask_sum > 0, logits.sum() / safe_mask_sum, mx.array(0.0))
    return logp_seq_avg, logits_mean


def orpo_loss(
    chosen_logps,
    chosen_logits_mean,
    rejected_logps,
    rejected_logits_mean,
    chosen_masks,
    rejected_masks,
    preference_scores,
    beta: float = 0.1,
):
    chosen_logps = chosen_logps * preference_scores

    # Stable log-odds computation
    # Ensure no NaN from inf - inf
    chosen_logps = mx.nan_to_num(chosen_logps, nan=0.0, posinf=0.0, neginf=-1000.0)
    rejected_logps = mx.nan_to_num(rejected_logps, nan=0.0, posinf=0.0, neginf=-1000.0)
    
    log_odds = chosen_logps - rejected_logps
    ratio = nn.log_sigmoid(log_odds)
    loss = -beta * ratio

    # Reward estimation
    chosen_reward = beta * chosen_logps
    rejected_reward = beta * rejected_logps
    reward = mx.stack([mx.mean(chosen_reward), mx.mean(rejected_reward)])

    num_tokens = chosen_masks.sum() + rejected_masks.sum()

    metrics = {
        "accuracies": mx.mean((chosen_reward > rejected_reward).astype(mx.float32)),
        "margins": mx.mean(chosen_reward - rejected_reward),
        "policy_chosen_logps": mx.mean(chosen_logps),
        "policy_rejected_logps": mx.mean(rejected_logps),
        "chosen_logits_mean": chosen_logits_mean,
        "rejected_logits_mean": rejected_logits_mean,
    }

    mx.clear_cache()
    return mx.mean(loss), reward, num_tokens, metrics


def iterate_orpo_batches(dataset, batch_size, max_seq_length, train=False):
    """Batch iterator for ORPO with preference scores"""
    idx = sorted(range(len(dataset)), key=lambda idx: len(dataset[idx]["chosen"]))

    if len(dataset) < batch_size:
        raise ValueError(
            f"Dataset must have at least batch_size={batch_size}"
            f" examples but only has {len(dataset)}."
        )

    step = mx.distributed.init().size()
    if batch_size % step != 0:
        raise ValueError("Batch size must be divisible by number of workers")

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
            pad_to = 8
            max_length_in_batch = pad_to * ((max_length + pad_to - 1) // pad_to)

            batch_size_per_device = batch_size // step
            chosen_arr = np.zeros(
                (batch_size_per_device, max_length_in_batch), np.int32
            )
            rejected_arr = np.zeros(
                (batch_size_per_device, max_length_in_batch), np.int32
            )
            chosen_masks = np.zeros(
                (batch_size_per_device, max_length_in_batch), np.float32
            )
            rejected_masks = np.zeros(
                (batch_size_per_device, max_length_in_batch), np.float32
            )

            preference_scores = np.array(
                [x.get("preference_score", 1.0) for x in batch], np.float32
            )

            for j in range(batch_size_per_device):
                chosen_length = min(chosen_lengths[j], max_length_in_batch)
                rejected_length = min(rejected_lengths[j], max_length_in_batch)

                chosen_arr[j, :chosen_length] = batch[j]["chosen"][:chosen_length]
                chosen_masks[j, :chosen_length] = 1.0
                rejected_arr[j, :rejected_length] = batch[j]["rejected"][
                    :rejected_length
                ]
                rejected_masks[j, :rejected_length] = 1.0

            yield (
                mx.array(chosen_arr),
                mx.array(rejected_arr),
                mx.array(chosen_masks),
                mx.array(rejected_masks),
                mx.array(preference_scores),
            )

        if not train:
            break


def evaluate_orpo(
    model, dataset, batch_size, num_batches, beta: float, max_seq_length=2048
):
    model.eval()
    all_losses = 0
    all_rewards = mx.zeros((2,))
    all_metrics = None
    ntokens = 0

    index_iterator = iter(range(num_batches)) if num_batches != -1 else iter(int, 1)
    for _, batch in zip(
        index_iterator,
        iterate_orpo_batches(
            dataset=dataset,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
        ),
    ):
        chosen, rejected, chosen_masks, rejected_masks, preference_scores = batch

        chosen_logps, chosen_logits_mean = get_logps(model, chosen, chosen_masks)
        rejected_logps, rejected_logits_mean = get_logps(
            model, rejected, rejected_masks
        )

        lvalue, reward, toks, metrics = orpo_loss(
            chosen_logps,
            chosen_logits_mean,
            rejected_logps,
            rejected_logits_mean,
            chosen_masks=chosen_masks,
            rejected_masks=rejected_masks,
            preference_scores=preference_scores,
            beta=beta,
        )
        all_losses += lvalue * toks
        all_rewards += reward * toks
        ntokens += toks

        if all_metrics is None:
            all_metrics = {k: v * toks for k, v in metrics.items()}
        else:
            for k, v in metrics.items():
                all_metrics[k] += v * toks

    mx.eval(all_losses, all_rewards, ntokens)
    all_losses = mx.distributed.all_sum(all_losses)
    all_rewards = mx.distributed.all_sum(all_rewards)
    ntokens = mx.distributed.all_sum(ntokens)
    all_metrics = {k: mx.distributed.all_sum(v) for k, v in all_metrics.items()}

    avg_metrics = {k: (v / ntokens).item() for k, v in all_metrics.items()}
    avg_rewards = (all_rewards / ntokens).tolist()
    avg_loss = (all_losses / ntokens).item()

    return avg_loss, avg_rewards, ntokens, avg_metrics


def train_orpo(
    model,
    optimizer,
    train_dataset,
    val_dataset: Optional[Any] = None,
    loss: callable = orpo_loss,
    args: ORPOTrainingArgs = ORPOTrainingArgs(),
    training_callback: TrainingCallback = None,
):
    mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
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

    efficient = True if args.seq_step_size is not None else False
    if efficient:
        cache = make_prompt_cache(model)
        seq_step_size = args.seq_step_size

    state = [model.state, optimizer.state, mx.random.state]

    def loss_wrapper(
        chosen_logps,
        chosen_logits_mean,
        rejected_logps,
        rejected_logits_mean,
        chosen_masks,
        rejected_masks,
        preference_scores,
    ):
        return loss(
            chosen_logps=chosen_logps,
            chosen_logits_mean=chosen_logits_mean,
            rejected_logps=rejected_logps,
            rejected_logits_mean=rejected_logits_mean,
            chosen_masks=chosen_masks,
            rejected_masks=rejected_masks,
            preference_scores=preference_scores,
            beta=args.beta,
        )

    loss_value_and_grad = nn.value_and_grad(model, loss_wrapper)

    @partial(mx.compile, inputs=state, outputs=state)
    def step(batch, prev_grad, do_update):
        chosen, rejected, chosen_masks, rejected_masks, preference_scores = batch

        chosen_logps, chosen_logits_mean = get_logps(model, chosen, chosen_masks)
        rejected_logps, rejected_logits_mean = get_logps(
            model, rejected, rejected_masks
        )

        (lvalue, reward, toks, metrics), grad = loss_value_and_grad(
            chosen_logps,
            chosen_logits_mean,
            rejected_logps,
            rejected_logits_mean,
            chosen_masks,
            rejected_masks,
            preference_scores=preference_scores,
        )

        if prev_grad is not None:
            grad = tree_map(lambda x, y: x + y, grad, prev_grad)

        if do_update:
            grad = average_gradients(grad)
            if grad_accum_steps > 1:
                grad = tree_map(lambda x: x / grad_accum_steps, grad)
            optimizer.update(model, grad)
            grad = None

        return lvalue, reward, toks, metrics, grad

    def seq_split_step(batch, prev_grad, do_update):
        chosen, rejected, chosen_masks, rejected_masks, preference_scores = batch
        batch_size = chosen.shape[0]

        def compute_logps_chunked(tokens, masks):
            seq_length = tokens.shape[1]
            logp_sum = mx.zeros((batch_size,))
            logits_mean_sum = mx.array(0.0)
            token_count = mx.array(0.0)

            reset_prompt_cache(cache)

            for s in range(0, seq_length, seq_step_size):
                end = min(s + seq_step_size, seq_length)
                if 0 < (seq_length - end) < 2:
                    end = seq_length

                chunk = tokens[:, s:end]
                chunk_mask = masks[:, s:end]

                chunk_avg, chunk_logits_mean = get_logps(model, chunk, chunk_mask, cache)
                
                chunk_input_mask = chunk_mask[:, :-1]
                chunk_lens = chunk_input_mask.sum(-1)
                
                logp_sum += chunk_avg * chunk_lens
                
                valid_toks = chunk_input_mask.sum()
                logits_mean_sum += chunk_logits_mean * valid_toks
                token_count += valid_toks

                if end >= seq_length:
                    break
            
            # Safe division for logits mean
            final_logits_mean = logits_mean_sum / (token_count + 1e-9)
            return logp_sum, final_logits_mean

        # 1. Forward Pass (No Grad)
        c_logp_sum, c_logits_mean = compute_logps_chunked(chosen, chosen_masks)
        r_logp_sum, r_logits_mean = compute_logps_chunked(rejected, rejected_masks)

        c_lens = chosen_masks[:, :-1].sum(-1)
        r_lens = rejected_masks[:, :-1].sum(-1)
        c_lens_safe = mx.where(c_lens > 0, c_lens, mx.array(1.0))
        r_lens_safe = mx.where(r_lens > 0, r_lens, mx.array(1.0))

        c_avg = mx.where(c_lens > 0, c_logp_sum / c_lens_safe, mx.array(0.0))
        r_avg = mx.where(r_lens > 0, r_logp_sum / r_lens_safe, mx.array(0.0))

        # 2. Compute ORPO Gradients Weights
        def internal_loss_fn(c, r):
            return loss_wrapper(
                c,
                c_logits_mean,
                r,
                r_logits_mean,
                chosen_masks,
                rejected_masks,
                preference_scores,
            )[0]

        # Get full metrics for reporting
        (lvalue, reward, toks, metrics) = loss_wrapper(
            c_avg,
            c_logits_mean,
            r_avg,
            r_logits_mean,
            chosen_masks,
            rejected_masks,
            preference_scores,
        )

        (g_c_avg, g_r_avg) = mx.grad(internal_loss_fn, argnums=[0, 1])(c_avg, r_avg)

        w_c = mx.where(c_lens > 0, g_c_avg / c_lens_safe, mx.array(0.0))
        w_r = mx.where(r_lens > 0, g_r_avg / r_lens_safe, mx.array(0.0))

        # 3. Backward chunks
        seq_grad_accum = None

        def accum_chunk_grads(tokens, masks, weights):
            nonlocal seq_grad_accum
            seq_length = tokens.shape[1]
            reset_prompt_cache(cache)

            def chunk_loss_fn(chunk, chunk_mask, weights):
                chunk_avg, _ = get_logps(model, chunk, chunk_mask, cache)
                chunk_lens = chunk_mask[:, :-1].sum(-1)
                chunk_sum = chunk_avg * chunk_lens
                return (chunk_sum * weights).sum()
            
            chunk_value_and_grad = nn.value_and_grad(model, chunk_loss_fn)

            for s in range(0, seq_length, seq_step_size):
                end = min(s + seq_step_size, seq_length)
                if 0 < (seq_length - end) < 2:
                    end = seq_length

                chunk = tokens[:, s:end]
                chunk_mask = masks[:, s:end]

                _, grad = chunk_value_and_grad(chunk, chunk_mask, weights)

                if seq_grad_accum is None:
                    seq_grad_accum = grad
                else:
                    seq_grad_accum = tree_map(lambda x, y: x + y, seq_grad_accum, grad)
                
                mx.eval(seq_grad_accum)

                if end >= seq_length:
                    break

        accum_chunk_grads(chosen, chosen_masks, w_c)
        accum_chunk_grads(rejected, rejected_masks, w_r)

        if prev_grad is not None:
            seq_grad_accum = tree_map(lambda x, y: x + y, seq_grad_accum, prev_grad)

        if do_update:
            seq_grad_accum = average_gradients(seq_grad_accum)
            if grad_accum_steps > 1:
                seq_grad_accum = tree_map(lambda x: x / grad_accum_steps, seq_grad_accum)
            optimizer.update(model, seq_grad_accum)
            seq_grad_accum = None

        return lvalue, reward, toks, metrics, seq_grad_accum

    model.train()
    seq_step_size = args.seq_step_size or args.max_seq_length
    losses = 0
    rewards = mx.zeros((2,))
    n_tokens = 0
    steps = 0
    trained_tokens = 0
    accumulated_metrics = {
        "accuracies": 0,
        "margins": 0,
        "policy_rejected_logps": 0,
        "policy_chosen_logps": 0,
        "rejected_logits_mean": 0,
        "chosen_logits_mean": 0,
    }
    grad_accum = None

    start = time.perf_counter()
    pbar = tqdm(range(1, args.iters + 1), desc="Training", disable=rank != 0)
    for it in pbar:
        batch = next(
            iterate_orpo_batches(
                train_dataset,
                args.batch_size,
                args.max_seq_length,
                train=True,
            )
        )

        if (
            val_dataset is not None
            and len(val_dataset) > 0
            and (it == 1 or it % args.steps_per_eval == 0 or it == args.iters)
        ):
            stop = time.perf_counter()
            val_loss, val_rewards, val_ntokens, val_metrics = evaluate_orpo(
                model=model,
                dataset=val_dataset,
                batch_size=args.batch_size,
                num_batches=args.val_batches,
                max_seq_length=args.max_seq_length,
                beta=args.beta,
            )
            val_time = time.perf_counter() - stop
            if rank == 0:
                tqdm.write(
                    f"Iter {it}: "
                    f"Val loss {val_loss:.3f}, "
                    f"Val chosen reward {val_rewards[0]:.3f}, "
                    f"Val rejected reward {val_rewards[1]:.3f}, "
                    f"Val accuracy {val_metrics['accuracies']:.3f}, "
                    f"Val margin {val_metrics['margins']:.3f}, "
                    f"Val took {val_time:.3f}s",
                )

            if training_callback is not None:
                training_callback.on_val_loss_report(
                    {
                        "iteration": it,
                        "val_loss": val_loss,
                        "val_chosen_reward": val_rewards[0],
                        "val_rejected_reward": val_rewards[1],
                        **{f"val_{k}": v for k, v in val_metrics.items()},
                        "val_time": val_time,
                    }
                )

            start = time.perf_counter()

        # Training step
        if efficient and batch[0].shape[1] > seq_step_size:
            lvalue, reward, toks, metrics, grad_accum = seq_split_step(
                batch,
                grad_accum,
                it % grad_accum_steps == 0,
            )
        else:
            lvalue, reward, toks, metrics, grad_accum = step(
                batch,
                grad_accum,
                it % grad_accum_steps == 0,
            )
            
        losses += lvalue
        rewards += reward
        n_tokens += toks
        steps += 1

        for k, v in metrics.items():
            accumulated_metrics[k] += v

        mx.eval(state, losses, rewards, n_tokens, grad_accum)

        if it % args.steps_per_report == 0 or it == args.iters:
            stop = time.perf_counter()

            train_loss = mx.distributed.all_sum(losses).item() / (steps * world_size)
            train_rewards = [
                r / (steps * world_size)
                for r in mx.distributed.all_sum(rewards).tolist()
            ]
            avg_metrics = {
                k: v / (steps * world_size) for k, v in accumulated_metrics.items()
            }
            n_tokens = mx.distributed.all_sum(n_tokens).item()
            learning_rate = optimizer.learning_rate.item()
            it_sec = args.steps_per_report / (stop - start)
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
                tqdm.write(
                    f"\nIter {it}: "
                    f"loss {train_loss:.3f}, "
                    f"chosen_r {train_rewards[0]:.3f}, "
                    f"rejected_r {train_rewards[1]:.3f}, "
                    f"acc {avg_metrics['accuracies']:.3f}, "
                    f"margin {avg_metrics['margins']:.3f}, "
                    f"lr {learning_rate:.3e}, "
                    f"it/s {it_sec:.3f}, "
                    f"tok/s {tokens_sec:.3f}, "
                    f"peak_mem {peak_mem:.3f}GB"
                )

            if training_callback is not None:
                train_info = {
                    "iteration": it,
                    "train_loss": train_loss,
                    "train_chosen_reward": train_rewards[0],
                    "train_rejected_reward": train_rewards[1],
                    **{f"train_{k}": v for k, v in avg_metrics.items()},
                    "learning_rate": learning_rate,
                    "iterations_per_second": it_sec,
                    "tokens_per_second": tokens_sec,
                    "trained_tokens": trained_tokens,
                    "peak_memory": peak_mem,
                }
                training_callback.on_train_loss_report(train_info)

            losses = 0
            rewards = mx.zeros((2,))
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
                f"Iter {it}: Saved adapter weights to "
                f"{args.adapter_file} and {checkpoint}."
            )

    adapter_weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(args.adapter_file), adapter_weights)
    tqdm.write(f"Saved final weights to {args.adapter_file}.")
