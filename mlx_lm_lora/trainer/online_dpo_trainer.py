import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.nn.utils import average_gradients
from mlx.utils import tree_flatten, tree_map
from mlx_lm.generate import generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.tuner.callbacks import TrainingCallback
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from .dpo_trainer import get_token_scores
from .judge import HumanPairwiseJudge, LLMPairwiseJudge
from .sft_trainer import SFTTrainingArgs, grad_checkpoint


@dataclass
class OnlineDPOTrainingArgs(SFTTrainingArgs):
    beta: float = field(
        default=0.1, metadata={"help": "Temperature parameter for DPO training."}
    )
    loss_type: str = field(
        default="sigmoid",
        metadata={"help": "DPO loss type: 'sigmoid', 'hinge', 'ipo', or 'dpop'."},
    )
    delta: float = field(
        default=50.0, metadata={"help": "Delta parameter for DPOP loss type."}
    )
    temperature: float = field(
        default=0.8,
        metadata={
            "help": "Temperature for sampling. The higher the temperature, the more random the completions."
        },
    )
    judge: str = field(
        default="human",
        metadata={
            "help": "What LLM to use as the judge, if 'human' empty, it's going to be you (human)."
        },
    )
    judge_system: str = field(
        default=None, metadata={"help": "How the judge should base its judging."}
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


def generate_for_online_dpo(
    model: nn.Module,
    tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
    prompts,
    max_tokens: int = 512,
    temperature: float = 0.8,
) -> list[list[str]]:
    completions = []

    sampler = make_sampler(
        temperature,
        top_p=1.0,
        min_p=0.0,
        min_tokens_to_keep=1,
        top_k=0,
        xtc_probability=0.0,
        xtc_threshold=0.0,
        xtc_special_tokens=tokenizer.encode("\n") + list(tokenizer.eos_token_ids),
    )

    for prompt in prompts:
        # Convert prompt tokens back to text if needed
        if isinstance(prompt, list):
            prompt_text = tokenizer.decode(prompt)
        else:
            prompt_text = prompt

        generated_1 = generate(
            model, tokenizer, prompt_text, max_tokens=max_tokens, sampler=sampler
        )
        generated_2 = generate(
            model, tokenizer, prompt_text, max_tokens=max_tokens, sampler=sampler
        )

        completions.append([generated_1, generated_2])
    return completions


def compute_score(scores, mask, loss_type):
    if isinstance(mask, list):
        mask = mx.array([m.sum() if hasattr(m, "sum") else m for m in mask])
    token_count = mask.sum(-1) if hasattr(mask, "sum") else mask
    return scores.sum(-1) / token_count if loss_type == "ipo" else scores.sum(-1)


def online_dpo_loss(
    policy_chosen_score: mx.array,
    policy_rejected_score: mx.array,
    reference_chosen_score: mx.array,
    reference_rejected_score: mx.array,
    chosen_masks: mx.array,
    rejected_masks: mx.array,
    beta: float,
    delta: float,
    loss_type: str = "sigmoid",
):
    # Preference logits
    logits = (policy_chosen_score - policy_rejected_score) - (
        reference_chosen_score - reference_rejected_score
    )

    # Loss calculation
    if loss_type == "sigmoid":
        losses = -nn.log_sigmoid(beta * logits)
    elif loss_type == "hinge":
        losses = nn.relu(1 - beta * logits)
    elif loss_type == "ipo":
        losses = (logits - 1 / (2 * beta)) ** 2
    elif loss_type == "dpop":
        penalty = mx.maximum(
            mx.zeros_like(policy_chosen_score),
            reference_chosen_score - policy_chosen_score,
        )
        losses = -(nn.log_sigmoid(beta * logits) - delta * penalty)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    # Token counts
    num_chosen_tokens = chosen_masks.sum(-1)
    num_rejected_tokens = rejected_masks.sum(-1)
    num_tokens = (num_chosen_tokens + num_rejected_tokens).sum()

    # Per-sample rewards
    chosen_reward = beta * (policy_chosen_score - reference_chosen_score)
    rejected_reward = beta * (policy_rejected_score - reference_rejected_score)
    reward = mx.stack([mx.mean(chosen_reward), mx.mean(rejected_reward)])

    # Metrics
    metrics = {
        "accuracies": mx.mean((chosen_reward > rejected_reward).astype(mx.float32)),
        "margins": mx.mean(chosen_reward - rejected_reward),
        "policy_rejected_logps": mx.mean(policy_rejected_score),
        "policy_chosen_logps": mx.mean(policy_chosen_score),
        "rejected_logits_mean": mx.mean(policy_rejected_score),
        "chosen_logits_mean": mx.mean(policy_chosen_score),
    }

    mx.clear_cache()
    return mx.mean(losses), reward, num_tokens, metrics


def iterate_online_dpo_batches(dataset, batch_size, max_seq_length, train=False):
    idx = sorted(range(len(dataset)), key=lambda idx: len(dataset[idx]["prompt"]))

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
            prompts = [x["prompt"] for x in batch]
            prompt_text = [x["prompt_text"] for x in batch]

            yield prompts, prompt_text
        if not train:
            break


def evaluate_online_dpo(
    model,
    ref_model,
    dataset,
    batch_size,
    num_batches,
    beta: float,
    delta: float,
    max_seq_length,
    loss_type,
    judge_config,
    loss_fn: callable = online_dpo_loss,
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
            loss_type=loss_type,
            beta=beta,
            delta=delta,
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


def train_online_dpo(
    model,
    ref_model,
    tokenizer,
    optimizer,
    train_dataset,
    val_dataset,
    judge_config,
    args: OnlineDPOTrainingArgs = OnlineDPOTrainingArgs(),
    judge_model: mx.array = None,
    judge_tokenizer: mx.array = None,
    loss_fn: callable = online_dpo_loss,
    training_callback: TrainingCallback = None,
):
    tqdm.write(f"Starting Online DPO training..., iters: {args.iters}")
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
            else:  #  Second completion is preferred
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
            delta=args.delta,
            loss_type=args.loss_type,
        )

    loss_value_and_grad = nn.value_and_grad(model, loss_wrapper)

    model.train()
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
            iterate_online_dpo_batches(
                dataset=train_dataset,
                batch_size=args.batch_size,
                max_seq_length=args.max_seq_length,
                train=True,
            )
        )
        if it == 1 or it % args.steps_per_eval == 0 or it == args.iters:
            stop = time.perf_counter()
            val_loss, val_rewards, val_ntokens, val_metrics = evaluate_online_dpo(
                model=model,
                ref_model=ref_model,
                tokenizer=tokenizer,
                dataset=val_dataset,
                batch_size=args.batch_size,
                num_batches=args.val_batches,
                max_seq_length=args.max_seq_length,
                loss_fn=loss_fn,
                beta=args.beta,
                delta=args.delta,
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
