import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.utils import average_gradients
from mlx.utils import tree_flatten, tree_map
from mlx_lm.tuner.callbacks import TrainingCallback
from tqdm import tqdm

from ..recurrent_patch import enable_memory_safe_recurrences, model_uses_recurrence
from .judge import LLMPPOJudge
from .online_dpo_trainer import (
    _online_token_logps,
    _pad_online_sequences,
    _validate_micro_batch_size,
    generate_for_online_dpo,
    iterate_online_dpo_batches,
)
from .sft_trainer import SFTTrainingArgs, grad_checkpoint


@dataclass
class RLHFReinforceTrainingArgs(SFTTrainingArgs):
    beta: float = field(
        default=0.1, metadata={"help": "KL penalty coefficient for RLHF training."}
    )
    judge: str = field(default=None, metadata={"help": "Path to reward model weights."})
    reference_model_path: str = field(
        default=None, metadata={"help": "Path to reference model weights."}
    )
    max_completion_length: int = field(
        default=128, metadata={"help": "Max tokens to generate per prompt."}
    )
    micro_batch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Maximum number of sampled trajectories scored per microbatch."
        },
    )


def compute_kl_penalty(logits_policy, logits_ref, masks):
    policy_probs = nn.softmax(logits_policy, axis=-1)
    ref_probs = nn.softmax(logits_ref, axis=-1)

    kl_div = policy_probs * (mx.log(policy_probs) - mx.log(ref_probs))
    kl_div = mx.sum(kl_div, axis=-1)
    return mx.sum(kl_div * masks, axis=-1)


def rlhf_reinforce_loss(
    policy_logits: mx.array,
    ref_logits: mx.array,
    rewards: mx.array,
    masks: mx.array,
    beta: float,
    targets: mx.array = None,
):
    """
    KL-regularized REINFORCE loss for RLHF.

    Computes per-token log-probs for the sampled trajectory,
    applies a KL penalty against a reference model, and uses
    (reward - beta * KL) as the advantage signal.
    """
    # Compute log probabilities for actual tokens
    labels = mx.argmax(policy_logits, axis=-1) if targets is None else targets
    policy_log_probs = -nn.losses.cross_entropy(policy_logits, labels, reduction="none")
    ref_log_probs = -nn.losses.cross_entropy(ref_logits, labels, reduction="none")

    # Compute KL divergence per token
    kl_div = policy_log_probs - ref_log_probs

    # Sum KL over sequence and average over batch
    kl_penalty = (kl_div * masks).sum(axis=-1)

    # Policy gradient loss
    advantages = rewards - beta * kl_penalty
    loss = -advantages * (policy_log_probs * masks).sum(axis=-1)

    # Normalize by token count
    token_count = masks.sum()
    loss = loss.sum() / token_count

    # Compute metrics
    metrics = {
        "rewards": mx.mean(rewards),
        "kl_penalty": mx.mean(kl_penalty),
        "advantages": mx.mean(advantages),
        "policy_logps": mx.mean(policy_log_probs),
        "ref_logps": mx.mean(ref_log_probs),
    }

    return loss, token_count, metrics


def _rlhf_reinforce_logp_loss(
    policy_log_probs: mx.array,
    ref_log_probs: mx.array,
    rewards: mx.array,
    masks: mx.array,
    beta: float,
):
    """Memory-bounded RLHF objective over already-selected trajectory logps."""
    policy_log_probs = mx.where(masks, policy_log_probs.astype(mx.float32), 0)
    ref_log_probs = mx.where(
        masks, mx.stop_gradient(ref_log_probs.astype(mx.float32)), 0
    )
    kl_penalty = (policy_log_probs - ref_log_probs).sum(axis=-1)
    advantages = rewards.astype(mx.float32) - beta * kl_penalty
    loss = -(advantages * policy_log_probs.sum(axis=-1)).sum()
    token_count = masks.sum()
    loss = loss / mx.maximum(token_count, 1)
    metrics = {
        "rewards": mx.mean(rewards),
        "kl_penalty": mx.mean(kl_penalty),
        "advantages": mx.mean(advantages),
        "policy_logps": mx.sum(policy_log_probs) / mx.maximum(token_count, 1),
        "ref_logps": mx.sum(ref_log_probs) / mx.maximum(token_count, 1),
    }
    return loss, token_count, metrics


def get_model_logits(model, tokens, masks):
    inputs = tokens[:, :-1]
    targets = tokens[:, 1:]
    target_masks = masks[:, 1:]
    return model(inputs), targets, target_masks


def _rlhf_loss_from_sequences(
    model, ref_model, sequences, rewards, beta, loss_fn
):
    tokens, target_masks = _pad_online_sequences(sequences)
    policy_log_probs = _online_token_logps(model, tokens, target_masks)
    if ref_model is None:
        ref_log_probs = mx.stop_gradient(policy_log_probs)
    else:
        ref_log_probs = mx.stop_gradient(
            _online_token_logps(ref_model, tokens, target_masks)
        )
    if loss_fn is rlhf_reinforce_loss:
        return _rlhf_reinforce_logp_loss(
            policy_log_probs, ref_log_probs, rewards, target_masks, beta
        )

    full_masks = mx.concatenate(
        [mx.ones((tokens.shape[0], 1), dtype=target_masks.dtype), target_masks],
        axis=1,
    )
    policy_logits, targets, target_masks = get_model_logits(model, tokens, full_masks)
    ref_logits = (
        get_model_logits(ref_model, tokens, full_masks)[0]
        if ref_model is not None
        else mx.stop_gradient(policy_logits)
    )
    return loss_fn(
        policy_logits=policy_logits,
        ref_logits=ref_logits,
        rewards=rewards,
        masks=target_masks,
        beta=beta,
    )


def _rlhf_value_and_grad(
    loss_value_and_grad, model, sequences, rewards, micro_batch_size
):
    """Accumulate RLHF gradients over trajectory microbatches."""
    if not sequences:
        raise ValueError("RLHF requires at least one sampled trajectory")
    total = len(sequences)
    token_counts = [max(int(sequence.size) - 1, 0) for sequence in sequences]
    total_tokens = max(sum(token_counts), 1)
    total_loss, counted_tokens, all_metrics, accumulated = 0, 0, None, None
    for start in range(0, total, micro_batch_size):
        stop = min(start + micro_batch_size, total)
        row_weight = (stop - start) / total
        token_weight = sum(token_counts[start:stop]) / total_tokens
        result = loss_value_and_grad(
            model, sequences[start:stop], rewards[start:stop]
        )
        (loss, tokens, metrics), grads = result
        grads = tree_map(lambda value, weight=token_weight: value * weight, grads)
        accumulated = (
            grads
            if accumulated is None
            else tree_map(lambda left, right: left + right, accumulated, grads)
        )
        total_loss = total_loss + loss * token_weight
        counted_tokens = counted_tokens + tokens
        weighted_metrics = {key: value * row_weight for key, value in metrics.items()}
        if all_metrics is None:
            all_metrics = weighted_metrics
        else:
            for key, value in weighted_metrics.items():
                all_metrics[key] = all_metrics[key] + value
        del result, loss, tokens, metrics, weighted_metrics, grads
        mx.eval(total_loss, counted_tokens, all_metrics, accumulated)
    return (total_loss, counted_tokens, all_metrics), accumulated


def evaluate_rlhf_reinforce(
    model,
    ref_model,
    dataset,
    batch_size,
    num_batches,
    beta: float,
    max_seq_length,
    judge_config,
    loss_fn: callable = rlhf_reinforce_loss,
    judge_model: mx.array = None,
    judge_tokenizer: mx.array = None,
    tokenizer=None,
    max_tokens: int = 512,
    micro_batch_size: Optional[int] = None,
):
    model.eval()
    if ref_model is not None:
        ref_model.eval()
    all_losses = 0
    all_metrics = None
    ntokens = 0
    micro_batch_size = _validate_micro_batch_size(
        micro_batch_size, batch_size * 2
    )

    index_iterator = iter(range(num_batches)) if num_batches != -1 else iter(int, 1)

    for _, batch in zip(
        index_iterator,
        iterate_online_dpo_batches(
            dataset=dataset,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
        ),
    ):
        prompts, prompt_texts = batch

        # Generate completions
        completions = generate_for_online_dpo(
            model,
            tokenizer,
            prompts,
            max_tokens=max_tokens,
            batch_size=max(1, micro_batch_size // 2),
        )

        judger = LLMPPOJudge(
            model=judge_model,
            tokenizer=judge_tokenizer,
            system_prompt=(judge_config or {}).get("system_prompt", None),
        )
        rewards = judger.judge(prompt_texts, completions=completions)

        all_tokens = []
        all_rewards = []

        for i, (prompt_text, completion_pair, reward_pair) in enumerate(
            zip(prompt_texts, completions, rewards)
        ):
            for j, (completion, reward) in enumerate(zip(completion_pair, reward_pair)):
                full_text = prompt_text + completion
                all_tokens.append(mx.array(tokenizer.encode(full_text), dtype=mx.int32))
                all_rewards.append(reward)
        batch_rewards = mx.array(all_rewards)
        total_rows = len(all_tokens)
        for start in range(0, total_rows, micro_batch_size):
            stop = min(start + micro_batch_size, total_rows)
            row_weight = (stop - start) / total_rows
            loss_value, toks, metrics = _rlhf_loss_from_sequences(
                model,
                ref_model,
                all_tokens[start:stop],
                batch_rewards[start:stop],
                beta,
                loss_fn,
            )
            all_losses += loss_value * toks
            ntokens += toks
            weighted_metrics = {key: value * row_weight for key, value in metrics.items()}
            if all_metrics is None:
                all_metrics = weighted_metrics
            else:
                for key, value in weighted_metrics.items():
                    all_metrics[key] += value
            del loss_value, toks, metrics, weighted_metrics
            mx.eval(all_losses, ntokens, *all_metrics.values())

    # Distributed reduction
    all_losses = mx.distributed.all_sum(all_losses)
    ntokens = mx.distributed.all_sum(ntokens)
    all_metrics = {k: mx.distributed.all_sum(v) for k, v in all_metrics.items()}

    # Compute averages
    avg_metrics = {k: (v / ntokens).item() for k, v in all_metrics.items()}
    avg_loss = (all_losses / ntokens).item()

    return avg_loss, [], ntokens, avg_metrics


def train_rlhf_reinforce(
    model,
    ref_model,
    tokenizer,
    optimizer,
    train_dataset,
    val_dataset: Optional[Any] = None,
    judge_config=None,
    args: RLHFReinforceTrainingArgs = RLHFReinforceTrainingArgs(),
    judge_model: mx.array = None,
    judge_tokenizer: mx.array = None,
    loss_fn: callable = rlhf_reinforce_loss,
    training_callback: TrainingCallback = None,
):
    if model_uses_recurrence(model):
        enable_memory_safe_recurrences(chunk_size=args.recurrence_chunk_size)
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

    state = [model.state, optimizer.state, mx.random.state]
    if ref_model is not None:
        ref_model.eval()
    micro_batch_size = _validate_micro_batch_size(
        args.micro_batch_size, args.batch_size * 2
    )

    def step(batch, prev_grad, do_update, accumulation_count):
        prompts, prompt_texts = batch

        # Generate completions for each prompt
        completions = generate_for_online_dpo(
            model,
            tokenizer,
            prompts,
            max_tokens=args.max_completion_length,
            batch_size=max(1, micro_batch_size // 2),
        )

        # Judge the completions
        judger = LLMPPOJudge(
            model=judge_model,
            tokenizer=judge_tokenizer,
            system_prompt=(judge_config or {}).get("system_prompt", None),
        )
        rewards = judger.judge(prompt_texts, completions=completions)

        all_tokens = []
        all_rewards = []

        for i, (prompt_text, completion_pair, reward_pair) in enumerate(
            zip(prompt_texts, completions, rewards)
        ):
            for j, (completion, reward) in enumerate(zip(completion_pair, reward_pair)):
                full_text = prompt_text + completion
                all_tokens.append(mx.array(tokenizer.encode(full_text), dtype=mx.int32))
                all_rewards.append(reward)
        batch_rewards = mx.array(all_rewards)
        (lvalue, toks, metrics), grad = _rlhf_value_and_grad(
            loss_value_and_grad,
            model,
            all_tokens,
            batch_rewards,
            micro_batch_size,
        )

        if prev_grad is not None:
            grad = tree_map(lambda x, y: x + y, grad, prev_grad)

        if do_update:
            grad = average_gradients(grad)
            if grad_accum_steps > 1:
                grad = tree_map(lambda x: x / accumulation_count, grad)
            optimizer.update(model, grad)
            grad = None

        return lvalue, batch_rewards, toks, metrics, grad

    def loss_wrapper(model, sequences, rewards):
        return _rlhf_loss_from_sequences(
            model, ref_model, sequences, rewards, args.beta, loss_fn
        )

    loss_value_and_grad = nn.value_and_grad(model, loss_wrapper)

    model.train()
    seq_step_size = args.seq_step_size or args.max_seq_length
    losses = 0
    n_tokens = 0
    steps = 0
    trained_tokens = 0
    accumulated_metrics = {
        "rewards": 0,
        "kl_penalty": 0,
        "advantages": 0,
        "policy_logps": 0,
        "ref_logps": 0,
    }
    grad_accum = None

    start = time.perf_counter()

    pbar = tqdm(range(1, args.iters + 1), desc="Training", disable=rank != 0)
    for it in pbar:
        batch = next(
            iterate_online_dpo_batches(
                dataset=train_dataset,
                batch_size=args.batch_size,
                max_seq_length=args.max_seq_length,
                train=True,
            )
        )

        if (
            val_dataset is not None
            and len(val_dataset) > 0
            and (it == 1 or it % args.steps_per_eval == 0 or it == args.iters)
        ):
            stop = time.perf_counter()
            val_loss, val_rewards, val_ntokens, val_metrics = evaluate_rlhf_reinforce(
                model=model,
                ref_model=ref_model,
                tokenizer=tokenizer,
                dataset=val_dataset,
                batch_size=args.batch_size,
                num_batches=args.val_batches,
                max_seq_length=args.max_seq_length,
                loss_fn=loss_fn,
                beta=args.beta,
                judge_model=judge_model,
                judge_tokenizer=judge_tokenizer,
                judge_config=judge_config,
                max_tokens=args.max_completion_length,
                micro_batch_size=micro_batch_size,
            )
            val_time = time.perf_counter() - stop
            if rank == 0:
                tqdm.write(
                    f"Iter {it}: "
                    f"Val loss {val_loss:.3f}, "
                    f"Val rewards {val_metrics['rewards']:.3f}, "
                    f"Val KL penalty {val_metrics['kl_penalty']:.3f}, "
                    f"Val advantages {val_metrics['advantages']:.3f}, "
                    f"Val took {val_time:.3f}s",
                )

            if training_callback is not None:
                training_callback.on_val_loss_report(
                    {
                        "iteration": it,
                        "val_loss": val_loss,
                        **{f"val_{k}": v for k, v in val_metrics.items()},
                        "val_time": val_time,
                    }
                )

            model.train()
            start = time.perf_counter()

        lvalue, rewards, toks, metrics, grad_accum = step(
            batch,
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
        mx.eval(state, losses, rewards, n_tokens, grad_accum, *_acc)

        if it % args.steps_per_report == 0 or it == args.iters:
            stop = time.perf_counter()

            train_loss = mx.distributed.all_sum(losses).item() / (steps * world_size)
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
                tqdm.write(
                    f"Iter {it}: Train loss {train_loss:.3f}, "
                    f"Rewards {avg_metrics['rewards']:.3f}, "
                    f"KL penalty {avg_metrics['kl_penalty']:.3f}, "
                    f"Learning Rate {learning_rate:.3e}, "
                    f"It/sec {it_sec:.3f}, "
                    f"Tokens/sec {tokens_sec:.3f}, "
                    f"Trained Tokens {trained_tokens}, "
                    f"Peak mem {peak_mem:.3f} GB",
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
            for k in accumulated_metrics:
                accumulated_metrics[k] = 0
            start = time.perf_counter()

        # Save adapter weights
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

    # Save final weights
    adapter_weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(args.adapter_file), adapter_weights)
    tqdm.write(f"Saved final weights to {args.adapter_file}.")
