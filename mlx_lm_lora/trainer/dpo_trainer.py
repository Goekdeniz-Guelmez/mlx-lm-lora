import time
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.utils import average_gradients
from mlx.utils import tree_map
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.tuner.callbacks import TrainingCallback
from tqdm import tqdm

from ..recurrent_patch import enable_memory_safe_recurrences, model_uses_recurrence
from .preference_utils import (
    accumulate_score_gradients,
    compute_score,
    compute_scores_chunked,
    evaluate_preference_batches,
    get_token_scores,
)
from .preference_utils import iterate_preference_batches as iterate_dpo_batches
from .sft_trainer import (
    SFTTrainingArgs,
    _install_qat_hooks,
    grad_checkpoint,
)
from .training_utils import save_adapters


@dataclass
class DPOTrainingArgs(SFTTrainingArgs):
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
    reference_model_path: str = field(
        default=None,
        metadata={
            "help": "Path to reference model weights. If None, uses the same model."
        },
    )


def dpo_loss(
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

    # Token counts and rewards
    num_chosen_tokens = chosen_masks.sum(-1)
    num_rejected_tokens = rejected_masks.sum(-1)
    num_tokens = (num_chosen_tokens + num_rejected_tokens).sum()

    chosen_reward = beta * mx.mean(policy_chosen_score - reference_chosen_score)
    rejected_reward = beta * mx.mean(policy_rejected_score - reference_rejected_score)
    reward = mx.stack([chosen_reward, rejected_reward])

    # Metrics
    metrics = {
        "accuracies": mx.mean((chosen_reward > rejected_reward).astype(mx.float32)),
        "margins": mx.mean(chosen_reward - rejected_reward),
        "policy_rejected_logps": mx.mean(policy_rejected_score / num_rejected_tokens),
        "policy_chosen_logps": mx.mean(policy_chosen_score / num_chosen_tokens),
        "rejected_logits_mean": mx.mean(policy_rejected_score),
        "chosen_logits_mean": mx.mean(policy_chosen_score),
    }

    mx.clear_cache()
    return mx.mean(losses), reward, num_tokens, metrics


def dpo_loss_from_model(
    model,
    ref_model,
    chosen,
    rejected,
    chosen_masks,
    rejected_masks,
    beta: float,
    delta: float,
    loss=dpo_loss,
    loss_type: str = "sigmoid",
):
    """Compute DPO loss while keeping reference model scores detached."""
    policy_chosen_score = compute_score(
        get_token_scores(model, chosen, chosen_masks), chosen_masks, loss_type
    )
    policy_rejected_score = compute_score(
        get_token_scores(model, rejected, rejected_masks), rejected_masks, loss_type
    )
    if ref_model is None:
        reference_chosen_score = mx.zeros_like(policy_chosen_score)
        reference_rejected_score = mx.zeros_like(policy_rejected_score)
    else:
        reference_chosen_score = compute_score(
            mx.stop_gradient(get_token_scores(ref_model, chosen, chosen_masks)),
            chosen_masks,
            loss_type,
        )
        reference_rejected_score = compute_score(
            mx.stop_gradient(get_token_scores(ref_model, rejected, rejected_masks)),
            rejected_masks,
            loss_type,
        )
    return loss(
        policy_chosen_score=policy_chosen_score,
        policy_rejected_score=policy_rejected_score,
        reference_chosen_score=reference_chosen_score,
        reference_rejected_score=reference_rejected_score,
        chosen_masks=chosen_masks,
        rejected_masks=rejected_masks,
        beta=beta,
        delta=delta,
        loss_type=loss_type,
    )


def evaluate_dpo(
    model,
    ref_model,
    dataset,
    batch_size,
    num_batches,
    beta: float,
    delta: float,
    max_seq_length,
    loss_type,
    loss_fn: callable = dpo_loss,
):
    model.eval()
    return evaluate_preference_batches(
        iterate_dpo_batches(dataset, batch_size, max_seq_length),
        num_batches,
        partial(
            dpo_loss_from_model,
            model,
            ref_model,
            beta=beta,
            delta=delta,
            loss=loss_fn,
            loss_type=loss_type,
        ),
        weight_rewards=False,
    )


def train_dpo(
    model,
    ref_model,
    optimizer,
    train_dataset,
    val_dataset: Optional[Any] = None,
    args: DPOTrainingArgs = DPOTrainingArgs(),
    loss_fn: callable = dpo_loss,
    training_callback: TrainingCallback = None,
    loss_type="sigmoid",
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
    if args.qat_start_step < 1:
        raise ValueError("qat_start_step must be at least 1")

    qat_installed = False
    efficient = True if args.seq_step_size is not None else False
    if efficient:
        cache = make_prompt_cache(model)
        seq_step_size = args.seq_step_size
        ref_cache = make_prompt_cache(ref_model) if ref_model is not None else None

    state = [model.state, optimizer.state, mx.random.state]

    def loss_wrapper(chosen, rejected, chosen_masks, rejected_masks):
        return dpo_loss_from_model(
            model,
            ref_model,
            chosen,
            rejected,
            chosen_masks,
            rejected_masks,
            beta=args.beta,
            delta=args.delta,
            loss=loss_fn,
            loss_type=loss_type,
        )

    loss_value_and_grad = nn.value_and_grad(model, loss_wrapper)

    @partial(mx.compile, inputs=state, outputs=state)
    def step(batch, prev_grad, do_update):
        chosen, rejected, chosen_masks, rejected_masks = batch

        (lvalue, reward, toks, metrics), grad = loss_value_and_grad(
            chosen, rejected, chosen_masks, rejected_masks
        )

        if prev_grad is not None:
            grad = tree_map(lambda x, y: x + y, grad, prev_grad)

        if do_update:
            grad = average_gradients(grad)
            if args.gradient_accumulation_steps > 1:
                grad = tree_map(lambda x: x / args.gradient_accumulation_steps, grad)
            optimizer.update(model, grad)
            grad = None

        return lvalue, reward, toks, metrics, grad

    def seq_split_step(batch, prev_grad, do_update):
        chosen, rejected, chosen_masks, rejected_masks = batch

        # 1. Forward Pass (No Grad) - compute scores
        c_score = compute_scores_chunked(
            model, cache, chosen, chosen_masks, seq_step_size
        )
        r_score = compute_scores_chunked(
            model, cache, rejected, rejected_masks, seq_step_size
        )

        if ref_model is not None:
            c_ref_score = compute_scores_chunked(
                ref_model, ref_cache, chosen, chosen_masks, seq_step_size
            )
            r_ref_score = compute_scores_chunked(
                ref_model, ref_cache, rejected, rejected_masks, seq_step_size
            )
        else:
            c_ref_score = mx.zeros_like(c_score)
            r_ref_score = mx.zeros_like(r_score)

        c_tokens_count = chosen_masks[:, :-1].sum(-1)
        r_tokens_count = rejected_masks[:, :-1].sum(-1)

        if loss_type == "ipo":
            c_score_arg = c_score / c_tokens_count
            r_score_arg = r_score / r_tokens_count
            c_ref_score_arg = c_ref_score / c_tokens_count
            r_ref_score_arg = r_ref_score / r_tokens_count
        else:
            c_score_arg = c_score
            r_score_arg = r_score
            c_ref_score_arg = c_ref_score
            r_ref_score_arg = r_ref_score

        # 2. Compute Gradients Weights
        def internal_loss_fn(c, r):
            l, _, _, _ = loss_fn(
                policy_chosen_score=c,
                policy_rejected_score=r,
                reference_chosen_score=c_ref_score_arg,
                reference_rejected_score=r_ref_score_arg,
                chosen_masks=chosen_masks,
                rejected_masks=rejected_masks,
                beta=args.beta,
                delta=args.delta,
                loss_type=loss_type,
            )
            return l

        lvalue, reward, toks, metrics = loss_fn(
            policy_chosen_score=c_score_arg,
            policy_rejected_score=r_score_arg,
            reference_chosen_score=c_ref_score_arg,
            reference_rejected_score=r_ref_score_arg,
            chosen_masks=chosen_masks,
            rejected_masks=rejected_masks,
            beta=args.beta,
            delta=args.delta,
            loss_type=loss_type,
        )

        (g_c, g_r) = mx.grad(internal_loss_fn, argnums=[0, 1])(c_score_arg, r_score_arg)

        w_c = g_c
        w_r = g_r

        if loss_type == "ipo":
            w_c = w_c / c_tokens_count
            w_r = w_r / r_tokens_count

        # 3. Backward chunks
        seq_grad_accum = accumulate_score_gradients(
            model, cache, chosen, chosen_masks, w_c, seq_step_size
        )
        seq_grad_accum = accumulate_score_gradients(
            model, cache, rejected, rejected_masks, w_r, seq_step_size, seq_grad_accum
        )

        if prev_grad is not None:
            seq_grad_accum = tree_map(lambda x, y: x + y, seq_grad_accum, prev_grad)

        if do_update:
            seq_grad_accum = average_gradients(seq_grad_accum)
            if args.gradient_accumulation_steps > 1:
                seq_grad_accum = tree_map(
                    lambda x: x / args.gradient_accumulation_steps, seq_grad_accum
                )
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
    opt_step = 0

    start = time.perf_counter()
    pbar = tqdm(range(1, args.iters + 1), desc="Training", disable=rank != 0)
    for it in pbar:
        batch = next(
            iterate_dpo_batches(
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
            val_loss, val_rewards, val_ntokens, val_metrics = evaluate_dpo(
                model=model,
                ref_model=ref_model,
                dataset=val_dataset,
                batch_size=args.batch_size,
                num_batches=args.val_batches,
                max_seq_length=args.max_seq_length,
                loss_fn=loss_fn,
                beta=args.beta,
                delta=args.delta,
                loss_type=loss_type,
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

            model.train()
            start = time.perf_counter()

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

        if it % grad_accum_steps == 0:
            opt_step += 1
            if (
                args.qat_enable
                and not qat_installed
                and opt_step >= args.qat_start_step
            ):
                _install_qat_hooks(model, args)
                qat_installed = True

        losses += lvalue
        rewards += reward
        n_tokens += toks
        steps += 1

        for k, v in metrics.items():
            accumulated_metrics[k] += v

        _acc = [v for v in accumulated_metrics.values() if isinstance(v, mx.array)]
        mx.eval(state, losses, rewards, n_tokens, grad_accum, *_acc)

        if it % args.steps_per_report == 0 or it == args.iters:
            stop = time.perf_counter()

            train_loss = mx.distributed.all_sum(losses).item() / (steps * world_size)
            train_rewards = mx.distributed.all_sum(rewards).tolist()
            train_rewards = [r / (steps * world_size) for r in train_rewards]
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
            save_adapters(model, args.adapter_file, it)

    save_adapters(model, args.adapter_file)
