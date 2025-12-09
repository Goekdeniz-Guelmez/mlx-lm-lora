import time
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.utils import average_gradients
from mlx.utils import tree_flatten, tree_map
from mlx_lm.tuner.callbacks import TrainingCallback
from tqdm import tqdm

from .dpo_trainer import get_token_scores
from .judge import HumanPairwiseJudge, LLMPairwiseJudge
from .online_dpo_trainer import (
    OnlineDPOTrainingArgs,
    compute_score,
    generate_for_online_dpo,
    iterate_online_dpo_batches,
)
from .sft_trainer import grad_checkpoint


@dataclass
class PPOTrainingArgs(OnlineDPOTrainingArgs):
    epsilon: float = field(
        default=0.2, metadata={"help": "The Epsilon for numerical stability."}
    )


def ppo_loss(
    policy_chosen_score: mx.array,
    policy_rejected_score: mx.array,
    reference_chosen_score: mx.array,
    reference_rejected_score: mx.array,
    chosen_masks: mx.array,
    rejected_masks: mx.array,
    beta: float = 0.1,
    epsilon: float = 0.2,
):
    # Compute log ratios for chosen and rejected sequences
    chosen_log_ratios = policy_chosen_score - reference_chosen_score
    rejected_log_ratios = policy_rejected_score - reference_rejected_score

    chosen_ratios = mx.exp(chosen_log_ratios)
    rejected_ratios = mx.exp(rejected_log_ratios)

    # Compute advantages (difference between chosen and rejected rewards)
    advantages = policy_chosen_score - policy_rejected_score

    # Normalize advantages
    advantage_mean = mx.mean(advantages)
    advantage_std = mx.sqrt(mx.var(advantages) + 1e-8)
    normalized_advantages = (advantages - advantage_mean) / advantage_std

    # PPO clipped objective for chosen sequences
    chosen_surr1 = chosen_ratios * normalized_advantages
    chosen_surr2 = (
        mx.clip(chosen_ratios, 1.0 - epsilon, 1.0 + epsilon) * normalized_advantages
    )
    chosen_policy_losses = -mx.minimum(chosen_surr1, chosen_surr2)

    # PPO clipped objective for rejected sequences (negative advantages)
    rejected_surr1 = rejected_ratios * (-normalized_advantages)
    rejected_surr2 = mx.clip(rejected_ratios, 1.0 - epsilon, 1.0 + epsilon) * (
        -normalized_advantages
    )
    rejected_policy_losses = -mx.minimum(rejected_surr1, rejected_surr2)

    # Combine losses
    policy_loss = mx.mean(chosen_policy_losses) + mx.mean(rejected_policy_losses)

    # KL penalty
    kl_penalty = beta * (mx.mean(chosen_log_ratios) + mx.mean(rejected_log_ratios))

    total_loss = policy_loss + kl_penalty

    # Calculate total tokens
    num_tokens = chosen_masks.sum() + rejected_masks.sum()

    # Rewards
    chosen_reward = beta * (policy_chosen_score - reference_chosen_score)
    rejected_reward = beta * (policy_rejected_score - reference_rejected_score)
    reward = mx.stack([mx.mean(chosen_reward), mx.mean(rejected_reward)])

    # Metrics
    metrics = {
        "policy_loss": policy_loss,
        "kl_penalty": kl_penalty,
        "advantages_mean": mx.mean(normalized_advantages),
        "ratios_mean": mx.mean(mx.concatenate([chosen_ratios, rejected_ratios])),
        "clip_fraction": mx.mean(
            (
                mx.abs(mx.concatenate([chosen_ratios, rejected_ratios]) - 1.0) > epsilon
            ).astype(mx.float32)
        ),
        "policy_chosen_logps": mx.mean(policy_chosen_score),
        "policy_rejected_logps": mx.mean(policy_rejected_score),
        "reference_chosen_logps": mx.mean(reference_chosen_score),
        "reference_rejected_logps": mx.mean(reference_rejected_score),
        "accuracies": mx.mean(
            (policy_chosen_score > policy_rejected_score).astype(mx.float32)
        ),
        "margins": mx.mean(policy_chosen_score - policy_rejected_score),
        "chosen_logits_mean": mx.mean(policy_chosen_score),
        "rejected_logits_mean": mx.mean(policy_rejected_score),
    }

    mx.clear_cache()
    return total_loss, reward, num_tokens, metrics


def evaluate_ppo(
    model,
    ref_model,
    dataset,
    batch_size,
    num_batches,
    beta: float,
    epsilon: float,
    max_seq_length,
    loss_type,
    judge_config,
    loss_fn: callable = ppo_loss,
    judge_model: mx.array = None,
    judge_tokenizer: mx.array = None,
    tokenizer=None,
    max_tokens: int = 512,
    temperature: float = 0.8,
):
    all_losses = 0
    all_rewards = mx.zeros((2,))
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

        completions = generate_for_online_dpo(
            model, tokenizer, prompts, temperature=temperature, max_tokens=max_tokens
        )

        if judge_model == "human":
            judger = HumanPairwiseJudge()
            judged = judger.judge(prompt_texts, completions=completions)
        else:
            judger = LLMPairwiseJudge(
                model=judge_model,
                tokenizer=judge_tokenizer,
                system_prompt=judge_config.get("system_prompt", None),
            )
            judged = judger.judge(prompt_texts, completions=completions)

        chosen = []
        rejected = []
        for i, (prompt_text, completion_pair, judgment) in enumerate(
            zip(prompt_texts, completions, judged)
        ):
            if judgment == 0:
                chosen.append(prompt_text + completion_pair[0])
                rejected.append(prompt_text + completion_pair[1])
            else:
                chosen.append(prompt_text + completion_pair[1])
                rejected.append(prompt_text + completion_pair[0])

        chosen_tokens = [mx.array(tokenizer.encode(text)) for text in chosen]
        rejected_tokens = [mx.array(tokenizer.encode(text)) for text in rejected]

        chosen_masks = [mx.ones(len(tokens)) for tokens in chosen_tokens]
        rejected_masks = [mx.ones(len(tokens)) for tokens in rejected_tokens]

        # Fix the get_token_scores calls - convert to proper batch format
        policy_chosen_scores = []
        policy_rejected_scores = []

        for tokens, mask in zip(chosen_tokens, chosen_masks):
            batch_tokens = tokens.reshape(1, -1)  # Shape: (1, seq_len)
            batch_mask = mask.reshape(1, -1)  # Shape: (1, seq_len)
            score = get_token_scores(model, batch_tokens, batch_mask)
            policy_chosen_scores.append(score)

        for tokens, mask in zip(rejected_tokens, rejected_masks):
            batch_tokens = tokens.reshape(1, -1)
            batch_mask = mask.reshape(1, -1)
            score = get_token_scores(model, batch_tokens, batch_mask)
            policy_rejected_scores.append(score)

        policy_chosen_score = mx.array(
            [
                compute_score(score, mask, loss_type)
                for score, mask in zip(policy_chosen_scores, chosen_masks)
            ]
        )
        policy_rejected_score = mx.array(
            [
                compute_score(score, mask, loss_type)
                for score, mask in zip(policy_rejected_scores, rejected_masks)
            ]
        )

        if ref_model is None:
            reference_chosen_logprobs = mx.zeros_like(policy_chosen_score)
            reference_rejected_logprobs = mx.zeros_like(policy_rejected_score)
        else:
            ref_chosen_scores = []
            ref_rejected_scores = []

            for tokens, mask in zip(chosen_tokens, chosen_masks):
                batch_tokens = tokens.reshape(1, -1)
                batch_mask = mask.reshape(1, -1)
                score = mx.stop_gradient(
                    get_token_scores(ref_model, batch_tokens, batch_mask)
                )
                ref_chosen_scores.append(score)

            for tokens, mask in zip(rejected_tokens, rejected_masks):
                batch_tokens = tokens.reshape(1, -1)
                batch_mask = mask.reshape(1, -1)
                score = mx.stop_gradient(
                    get_token_scores(ref_model, batch_tokens, batch_mask)
                )
                ref_rejected_scores.append(score)

            reference_chosen_logprobs = mx.array(
                [
                    compute_score(score, mask, loss_type)
                    for score, mask in zip(ref_chosen_scores, chosen_masks)
                ]
            )
            reference_rejected_logprobs = mx.array(
                [
                    compute_score(score, mask, loss_type)
                    for score, mask in zip(ref_rejected_scores, rejected_masks)
                ]
            )

        # Convert masks to token counts
        chosen_mask_counts = mx.array([mask.sum() for mask in chosen_masks])
        rejected_mask_counts = mx.array([mask.sum() for mask in rejected_masks])

        # Compute loss
        loss_value, reward, toks, metrics = loss_fn(
            policy_chosen_score=policy_chosen_score,
            policy_rejected_score=policy_rejected_score,
            reference_chosen_score=reference_chosen_logprobs,
            reference_rejected_score=reference_rejected_logprobs,
            chosen_masks=chosen_mask_counts,
            rejected_masks=rejected_mask_counts,
            beta=beta,
            epsilon=epsilon,
        )

        all_losses += loss_value * toks
        all_rewards += reward
        ntokens += toks

        if all_metrics is None:
            all_metrics = {k: v * toks for k, v in metrics.items()}
        else:
            for k, v in metrics.items():
                all_metrics[k] += v * toks

    mx.eval(all_losses, all_rewards, ntokens)

    # Distributed reduction
    all_losses = mx.distributed.all_sum(all_losses)
    all_rewards = mx.distributed.all_sum(all_rewards)
    ntokens = mx.distributed.all_sum(ntokens)
    all_metrics = {k: mx.distributed.all_sum(v) for k, v in all_metrics.items()}

    # Compute averages
    avg_metrics = {k: (v / ntokens).item() for k, v in all_metrics.items()}
    avg_rewards = (all_rewards / ntokens).tolist()
    avg_loss = (all_losses / ntokens).item()

    return avg_loss, avg_rewards, ntokens, avg_metrics


def train_ppo(
    model,
    ref_model,
    tokenizer,
    optimizer,
    train_dataset,
    val_dataset,
    judge_config,
    args: PPOTrainingArgs = PPOTrainingArgs(),
    judge_model: mx.array = None,
    judge_tokenizer: mx.array = None,
    loss_fn: callable = ppo_loss,
    training_callback: TrainingCallback = None,
):
    mx.set_wired_limit(mx.metal.device_info()["max_recommended_working_set_size"])
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
            model,
            tokenizer,
            prompts,
            max_tokens=args.max_completion_length,
            temperature=args.temperature,
        )

        # Judge the completions
        if judge_model == "human":
            judger = HumanPairwiseJudge()
            judged = judger.judge(prompt_texts, completions=completions)
        else:
            judger = LLMPairwiseJudge(
                model=judge_model,
                tokenizer=judge_tokenizer,
                system_prompt=judge_config.get("system_prompt", None),
            )
            judged = judger.judge(prompt_texts, completions=completions)

        # Process judged results to create chosen/rejected pairs
        chosen = []
        rejected = []
        for i, (prompt_text, completion_pair, judgment) in enumerate(
            zip(prompt_texts, completions, judged)
        ):
            if judgment == 0:  # First completion is preferred
                chosen.append(prompt_text + completion_pair[0])
                rejected.append(prompt_text + completion_pair[1])
            else:  # Second completion is preferred
                chosen.append(prompt_text + completion_pair[1])
                rejected.append(prompt_text + completion_pair[0])

        # Tokenize chosen and rejected
        chosen_tokens = [mx.array(tokenizer.encode(text)) for text in chosen]
        rejected_tokens = [mx.array(tokenizer.encode(text)) for text in rejected]

        # Create masks
        chosen_masks = [mx.ones(len(tokens)) for tokens in chosen_tokens]
        rejected_masks = [mx.ones(len(tokens)) for tokens in rejected_tokens]

        # Get policy scores
        policy_chosen_scores = []
        policy_rejected_scores = []

        for tokens, mask in zip(chosen_tokens, chosen_masks):
            batch_tokens = tokens.reshape(1, -1)
            batch_mask = mask.reshape(1, -1)
            score = get_token_scores(model, batch_tokens, batch_mask)
            policy_chosen_scores.append(score)

        for tokens, mask in zip(rejected_tokens, rejected_masks):
            batch_tokens = tokens.reshape(1, -1)
            batch_mask = mask.reshape(1, -1)
            score = get_token_scores(model, batch_tokens, batch_mask)
            policy_rejected_scores.append(score)

        policy_chosen_score = mx.array(
            [
                compute_score(score, mask, args.loss_type)
                for score, mask in zip(policy_chosen_scores, chosen_masks)
            ]
        )
        policy_rejected_score = mx.array(
            [
                compute_score(score, mask, args.loss_type)
                for score, mask in zip(policy_rejected_scores, rejected_masks)
            ]
        )

        # Get reference scores
        ref_chosen_scores = []
        ref_rejected_scores = []

        for tokens, mask in zip(chosen_tokens, chosen_masks):
            batch_tokens = tokens.reshape(1, -1)
            batch_mask = mask.reshape(1, -1)
            score = mx.stop_gradient(
                get_token_scores(ref_model, batch_tokens, batch_mask)
            )
            ref_chosen_scores.append(score)

        for tokens, mask in zip(rejected_tokens, rejected_masks):
            batch_tokens = tokens.reshape(1, -1)
            batch_mask = mask.reshape(1, -1)
            score = mx.stop_gradient(
                get_token_scores(ref_model, batch_tokens, batch_mask)
            )
            ref_rejected_scores.append(score)

        reference_chosen_logprobs = mx.array(
            [
                compute_score(score, mask, args.loss_type)
                for score, mask in zip(ref_chosen_scores, chosen_masks)
            ]
        )
        reference_rejected_logprobs = mx.array(
            [
                compute_score(score, mask, args.loss_type)
                for score, mask in zip(ref_rejected_scores, rejected_masks)
            ]
        )

        # Stack masks into proper 2D tensors
        chosen_mask_array = mx.stack(chosen_masks)
        rejected_mask_array = mx.stack(rejected_masks)

        # Compute loss and gradients
        (lvalue, reward, toks, metrics), grad = loss_value_and_grad(
            policy_chosen_score,
            policy_rejected_score,
            reference_chosen_logprobs,
            reference_rejected_logprobs,
            chosen_mask_array,
            rejected_mask_array,
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

    def loss_wrapper(
        policy_chosen_score,
        policy_rejected_score,
        reference_chosen_score,
        reference_rejected_score,
        chosen_masks,
        rejected_masks,
    ):
        return loss_fn(
            policy_chosen_score=policy_chosen_score,
            policy_rejected_score=policy_rejected_score,
            reference_chosen_score=reference_chosen_score,
            reference_rejected_score=reference_rejected_score,
            chosen_masks=chosen_masks,
            rejected_masks=rejected_masks,
            beta=args.beta,
            epsilon=args.epsilon,
        )

    loss_value_and_grad = nn.value_and_grad(model, loss_wrapper)

    model.train()
    losses = 0
    rewards = mx.zeros((2,))
    n_tokens = 0
    steps = 0
    trained_tokens = 0

    accumulated_metrics = {
        "policy_loss": 0,
        "kl_penalty": 0,
        "advantages_mean": 0,
        "ratios_mean": 0,
        "clip_fraction": 0,
        "policy_chosen_logps": 0,
        "policy_rejected_logps": 0,
        "reference_chosen_logps": 0,
        "reference_rejected_logps": 0,
        "accuracies": 0,
        "margins": 0,
        "chosen_logits_mean": 0,
        "rejected_logits_mean": 0,
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
            val_loss, val_rewards, val_ntokens, val_metrics = evaluate_ppo(
                model=model,
                ref_model=ref_model,
                tokenizer=tokenizer,
                dataset=val_dataset,
                batch_size=args.batch_size,
                num_batches=args.val_batches,
                max_seq_length=args.max_seq_length,
                loss_fn=loss_fn,
                beta=args.beta,
                epsilon=args.epsilon,
                loss_type=args.loss_type,
                judge_config=judge_config,
                judge_model=judge_model,
                judge_tokenizer=judge_tokenizer,
                max_tokens=args.max_completion_length,
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

        lvalue, reward, toks, metrics, grad_accum = step(
            batch,
            grad_accum,
            it % grad_accum_steps == 0,
        )
        losses += lvalue
        rewards += reward
        n_tokens += toks
        steps += 1

        # Safely accumulate metrics - only add if the key exists in accumulated_metrics
        for k, v in metrics.items():
            if k in accumulated_metrics:
                accumulated_metrics[k] += v
            else:
                # Log warning for missing keys
                print(f"Warning: Metric key '{k}' not found in accumulated_metrics")

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
                    f"Accuracy {avg_metrics['accuracies']:.3f}, "
                    f"Margin {avg_metrics['margins']:.3f}, "
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
            # Reset accumulated metrics
            accumulated_metrics = {k: 0 for k in accumulated_metrics.keys()}
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
