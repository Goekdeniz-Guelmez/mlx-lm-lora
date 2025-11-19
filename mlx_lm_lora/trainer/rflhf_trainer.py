import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.utils import average_gradients
from mlx.utils import tree_flatten, tree_map
from mlx_lm.tuner.callbacks import TrainingCallback
from tqdm import tqdm

from .judge import LLMPPOJudge
from .online_dpo_trainer import (
    compute_score,
    generate_for_online_dpo,
    iterate_online_dpo_batches,
)
from .sft_trainer import SFTTrainingArgs, grad_checkpoint


@dataclass
class RLHFTrainingArgs(SFTTrainingArgs):
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


def compute_kl_penalty(logits_policy, logits_ref, masks):
    policy_probs = nn.softmax(logits_policy, axis=-1)
    ref_probs = nn.softmax(logits_ref, axis=-1)

    kl_div = policy_probs * (mx.log(policy_probs) - mx.log(ref_probs))
    kl_div = mx.sum(kl_div, axis=-1)
    return mx.sum(kl_div * masks, axis=-1)


def rlhf_loss(
    policy_logits: mx.array,
    ref_logits: mx.array,
    rewards: mx.array,
    masks: mx.array,
    beta: float,
):
    # Compute log probabilities for actual tokens
    labels = mx.argmax(policy_logits, axis=-1)
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

    mx.clear_cache()
    return loss, token_count, metrics


def get_model_logits(model, tokens, masks):
    inputs = tokens[:, :-1]
    targets = tokens[:, 1:]
    target_masks = masks[:, 1:]
    return model(inputs), targets, target_masks


def evaluate_rlhf(
    model,
    ref_model,
    dataset,
    batch_size,
    num_batches,
    beta: float,
    max_seq_length,
    judge_config,
    loss_fn: callable = rlhf_loss,
    judge_model: mx.array = None,
    judge_tokenizer: mx.array = None,
    tokenizer=None,
    max_tokens: int = 512,
):
    all_losses = 0
    all_metrics = None
    ntokens = 0

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
            model, tokenizer, prompts, max_tokens=max_tokens
        )

        judger = LLMPPOJudge(
            model=judge_model,
            tokenizer=judge_tokenizer,
            system_prompt=judge_config.get("system_prompt", None),
        )
        rewards = judger.judge(prompt_texts, completions=completions)

        # Process completions into tokens and masks
        all_tokens = []
        all_masks = []
        all_rewards = []

        for i, (prompt_text, completion_pair, reward_pair) in enumerate(
            zip(prompt_texts, completions, rewards)
        ):
            for j, (completion, reward) in enumerate(zip(completion_pair, reward_pair)):
                full_text = prompt_text + completion
                tokens = mx.array(tokenizer.encode(full_text))
                mask = mx.ones(len(tokens))

                all_tokens.append(tokens)
                all_masks.append(mask)
                all_rewards.append(reward)

        # Pad sequences to same length
        max_len = max(len(tokens) for tokens in all_tokens)
        padded_tokens = []
        padded_masks = []

        for tokens, mask in zip(all_tokens, all_masks):
            pad_len = max_len - len(tokens)
            if pad_len > 0:
                padded_tokens.append(
                    mx.concatenate([tokens, mx.zeros(pad_len, dtype=tokens.dtype)])
                )
                padded_masks.append(mx.concatenate([mask, mx.zeros(pad_len)]))
            else:
                padded_tokens.append(tokens)
                padded_masks.append(mask)

        batch_tokens = mx.stack(padded_tokens)
        batch_masks = mx.stack(padded_masks)
        batch_rewards = mx.array(all_rewards)

        # Get model logits
        policy_logits, targets, target_masks = get_model_logits(
            model, batch_tokens, batch_masks
        )

        if ref_model is not None:
            ref_logits, _, _ = get_model_logits(ref_model, batch_tokens, batch_masks)
        else:
            ref_logits = mx.zeros_like(policy_logits)

        # Compute loss
        loss_value, toks, metrics = loss_fn(
            policy_logits=policy_logits,
            ref_logits=ref_logits,
            rewards=batch_rewards,
            masks=target_masks,
            beta=beta,
        )

        all_losses += loss_value * toks
        ntokens += toks

        if all_metrics is None:
            all_metrics = {k: v * toks for k, v in metrics.items()}
        else:
            for k, v in metrics.items():
                all_metrics[k] += v * toks

    mx.eval(all_losses, ntokens)

    # Distributed reduction
    all_losses = mx.distributed.all_sum(all_losses)
    ntokens = mx.distributed.all_sum(ntokens)
    all_metrics = {k: mx.distributed.all_sum(v) for k, v in all_metrics.items()}

    # Compute averages
    avg_metrics = {k: (v / ntokens).item() for k, v in all_metrics.items()}
    avg_loss = (all_losses / ntokens).item()

    return avg_loss, [], ntokens, avg_metrics


def train_rlhf(
    model,
    ref_model,
    tokenizer,
    optimizer,
    train_dataset,
    val_dataset,
    judge_config,
    args: RLHFTrainingArgs = RLHFTrainingArgs(),
    judge_model: mx.array = None,
    judge_tokenizer: mx.array = None,
    loss_fn: callable = rlhf_loss,
    training_callback: TrainingCallback = None,
):
    tqdm.write(f"Starting RLHF training..., iters: {args.iters}")
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

    def step(batch, prev_grad, do_update):
        prompts, prompt_texts = batch

        # Generate completions for each prompt
        completions = generate_for_online_dpo(
            model, tokenizer, prompts, max_tokens=args.max_completion_length
        )

        # Judge the completions
        judger = LLMPPOJudge(
            model=judge_model,
            tokenizer=judge_tokenizer,
            system_prompt=judge_config.get("system_prompt", None),
        )
        rewards = judger.judge(prompt_texts, completions=completions)

        # Process completions into tokens and masks
        all_tokens = []
        all_masks = []
        all_rewards = []

        for i, (prompt_text, completion_pair, reward_pair) in enumerate(
            zip(prompt_texts, completions, rewards)
        ):
            for j, (completion, reward) in enumerate(zip(completion_pair, reward_pair)):
                full_text = prompt_text + completion
                tokens = mx.array(tokenizer.encode(full_text))
                mask = mx.ones(len(tokens))

                all_tokens.append(tokens)
                all_masks.append(mask)
                all_rewards.append(reward)

        # Pad sequences to same length
        max_len = max(len(tokens) for tokens in all_tokens)
        padded_tokens = []
        padded_masks = []

        for tokens, mask in zip(all_tokens, all_masks):
            pad_len = max_len - len(tokens)
            if pad_len > 0:
                padded_tokens.append(
                    mx.concatenate([tokens, mx.zeros(pad_len, dtype=tokens.dtype)])
                )
                padded_masks.append(mx.concatenate([mask, mx.zeros(pad_len)]))
            else:
                padded_tokens.append(tokens)
                padded_masks.append(mask)

        batch_tokens = mx.stack(padded_tokens)
        batch_masks = mx.stack(padded_masks)
        batch_rewards = mx.array(all_rewards)

        # Get model logits
        policy_logits, targets, target_masks = get_model_logits(
            model, batch_tokens, batch_masks
        )

        if ref_model is not None:
            ref_logits, _, _ = get_model_logits(ref_model, batch_tokens, batch_masks)
        else:
            ref_logits = mx.zeros_like(policy_logits)

        # Compute loss and gradients
        (lvalue, toks, metrics), grad = loss_value_and_grad(
            policy_logits, ref_logits, batch_rewards, target_masks
        )

        if prev_grad is not None:
            grad = tree_map(lambda x, y: x + y, grad, prev_grad)

        if do_update:
            grad = average_gradients(grad)
            if grad_accum_steps > 1:
                grad = tree_map(lambda x: x / grad_accum_steps, grad)
            optimizer.update(model, grad)
            grad = None

        return lvalue, batch_rewards, toks, metrics, grad

    def loss_wrapper(policy_logits, ref_logits, rewards, masks):
        return loss_fn(
            policy_logits=policy_logits,
            ref_logits=ref_logits,
            rewards=rewards,
            masks=masks,
            beta=args.beta,
        )

    loss_value_and_grad = nn.value_and_grad(model, loss_wrapper)

    model.train()
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

        if it == 1 or it % args.steps_per_eval == 0 or it == args.iters:
            stop = time.perf_counter()
            val_loss, val_rewards, val_ntokens, val_metrics = evaluate_rlhf(
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

            start = time.perf_counter()

        lvalue, rewards, toks, metrics, grad_accum = step(
            batch,
            grad_accum,
            it % grad_accum_steps == 0,
        )
        losses += lvalue
        n_tokens += toks
        steps += 1

        for k, v in metrics.items():
            accumulated_metrics[k] += v

        mx.eval(state, losses, rewards, n_tokens, grad_accum)

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
