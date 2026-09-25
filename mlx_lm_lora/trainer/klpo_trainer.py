"""KL-Regularized Policy Optimization (KLPO) for MLX-LM-LoRA.

The trainer follows the reference KLPO implementation while reusing the local
GRPO pipeline for datasets, reward callbacks, generation batching, model
scoring, microbatch boundaries, distributed reductions, and checkpointing.
"""

import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx import nn
from mlx.utils import tree_flatten, tree_map
from mlx_lm.generate import BatchGenerator
from mlx_lm.tuner.callbacks import TrainingCallback
from tqdm import tqdm

from ..recurrent_patch import enable_memory_safe_recurrences, model_uses_recurrence
from .grpo_reward_functions import (
    r1_accuracy_reward_func,
    r1_count_xml,
    r1_int_reward_func,
    r1_soft_format_reward_func,
    r1_strict_format_reward_func,
)
from .grpo_trainer import _prepare_grpo_inputs, iterate_grpo_batches
from .sft_trainer import SFTTrainingArgs, average_gradients, grad_checkpoint


@dataclass
class KLPOTrainingArgs(SFTTrainingArgs):
    beta: float = field(default=0.1, metadata={"help": "KLPO regularization strength."})
    route: str = field(
        default="token",
        metadata={"help": "KLPO regression route: 'token' or 'sequence'."},
    )
    kl_estimator: str = field(
        default="mc",
        metadata={"help": "KL estimator: 'mc', 'topk', 'binary', or 'full'."},
    )
    mc_samples: int = field(
        default=128,
        metadata={"help": "Auxiliary MC-KL draws per visited prefix."},
    )
    top_k: int = field(
        default=128,
        metadata={"help": "Stored sampler head size for TopK-KL."},
    )
    tail_floor: float = field(
        default=1e-6,
        metadata={"help": "TopK-KL tail probability floor."},
    )
    max_completion_length: int = field(
        default=512, metadata={"help": "Maximum generated tokens."}
    )
    temperature: float = field(
        default=1.0, metadata={"help": "Collection sampler temperature."}
    )
    reward_weights: list[float] | None = field(
        default=None,
        metadata={"help": "Weights for each reward function."},
    )


def _log1mexp(x):
    """Stable log(1-exp(x)) for negative log-probabilities."""
    # Float32 log-softmax can round a near-certain action to log(p)=0. The
    # binary complement is undefined at that rounded boundary, so keep the
    # numerical calculation strictly inside p < 1.
    x = mx.minimum(x, -float(np.finfo(np.float32).eps))
    cutoff = -math.log(2.0)
    return mx.where(x < cutoff, mx.log1p(-mx.exp(x)), mx.log(-mx.expm1(x)))


def _masked_logps(logps, mask):
    return mx.where(mask, logps.astype(mx.float32), 0)


def _binary_terms(current, behavior, mask):
    # Use finite sentinels in padding before complement arithmetic and clamp
    # rounded-one active probabilities to a strictly negative log-probability.
    log_p = mx.minimum(mx.where(mask, current, -1.0), -float(np.finfo(np.float32).eps))
    log_q = mx.minimum(
        mx.where(mask, mx.stop_gradient(behavior), -1.0),
        -float(np.finfo(np.float32).eps),
    )
    log_pc = _log1mexp(log_p)
    log_qc = _log1mexp(log_q)
    ell = log_p - log_q
    q_complement = -mx.expm1(log_q)
    binary_kl = -mx.exp(log_q) * ell + q_complement * (log_qc - log_pc)
    centered_ratio = q_complement * (ell + log_qc - log_pc)
    correction = mx.exp(log_qc - log_pc)
    return tuple(
        mx.where(mask, value, 0)
        for value in (binary_kl, centered_ratio, correction)
    )


def _prepare_loss_inputs(logps, behavior_logps, rewards, mask, beta):
    if beta <= 0 or not math.isfinite(beta):
        raise ValueError("KLPO beta must be positive and finite.")
    current = _masked_logps(logps, mask)
    behavior = mx.stop_gradient(_masked_logps(behavior_logps, mask))
    returns = mx.stop_gradient(rewards.astype(mx.float32))
    return current, behavior, returns


def _token_loss(current, behavior, rewards, mask, *, estimator, beta, aux=None, head=None, full=None, tail_floor=1e-6):
    """Return the KLPO token-regression backward surrogate and statistics."""
    h = mx.stop_gradient(
        mx.where(mask, rewards[:, None] - beta * (current - behavior), 0)
    )

    if estimator == "mc":
        mc_current, mc_behavior = aux
        local_kl = mx.mean(
            mx.stop_gradient(mc_behavior) - mx.stop_gradient(mc_current), axis=-1
        )
        loss = -(h * (current - mx.mean(mc_current, axis=-1))).sum(axis=-1).mean()
        return loss, local_kl, mx.array(0.0), {}

    if estimator == "binary":
        binary_kl, _, correction = _binary_terms(current, behavior, mask)
        loss = -(h * mx.stop_gradient(correction) * current).sum(axis=-1).mean()
        return loss, binary_kl, mx.array(0.0), {}

    p_log, q_log = head if estimator == "topk" else full
    p_log = p_log.astype(mx.float32)
    q_log = mx.stop_gradient(q_log.astype(mx.float32))
    q = mx.exp(q_log)
    safe_q_log = mx.where(mx.isfinite(q_log), q_log, 0)
    p_detached = mx.stop_gradient(p_log)

    if estimator == "full":
        coefficients = q
        local_kl = (q * (safe_q_log - p_detached)).sum(axis=-1)
    else:
        if not math.isfinite(tail_floor) or not 0 < tail_floor < 1:
            raise ValueError("tail_floor must be finite and in (0, 1)")
        p = mx.exp(p_detached)
        sampler_tail = 1 - q.sum(axis=-1)
        trainer_tail = 1 - p.sum(axis=-1)
        q_tail = mx.maximum(sampler_tail, tail_floor)
        p_tail = mx.maximum(trainer_tail, tail_floor)
        coefficients = q - (q_tail / p_tail)[..., None] * p
        local_kl = (q * (safe_q_log - p_detached)).sum(axis=-1)
        local_kl = local_kl + q_tail * (mx.log(q_tail) - mx.log(p_tail))

    local_kl = mx.where(mask, local_kl, 0)
    correction = (coefficients * p_log).sum(axis=-1)
    correction = mx.where(mask, correction, 0)
    loss = -(h * (current - correction)).sum(axis=-1).mean()
    extra = {}
    if estimator == "topk":
        extra["floored_tokens"] = mx.sum(
            ((sampler_tail < tail_floor) | (trainer_tail < tail_floor)) & mask
        )
    return loss, local_kl, mx.array(0.0), extra


def _sequence_loss(current, behavior, rewards, mask, *, estimator, beta, aux=None, head=None, full=None, tail_floor=1e-6):
    """Return the KLPO sequence-regression backward surrogate and statistics."""
    if estimator == "binary":
        binary_kl, centered_ratio, correction = _binary_terms(current, behavior, mask)
        residual = mx.stop_gradient(rewards - beta * centered_ratio.sum(axis=-1))
        regression = residual * residual / (2 * beta)
        loss = -(residual * (mx.stop_gradient(correction) * current).sum(axis=-1)).mean()
        return loss, binary_kl, regression.mean(), {"residual": residual}

    if estimator == "mc":
        mc_current, mc_behavior = aux
        m = mc_current.shape[-1]
        if m < 2:
            raise ValueError("sequence MC-KL needs at least two auxiliary samples")
        ratio = mx.stop_gradient(mc_behavior) - mx.stop_gradient(mc_current)
        residuals = rewards[:, None] - beta * (
            (mx.stop_gradient(current) - behavior).sum(axis=-1)[:, None]
            + ratio.sum(axis=1)
        )
        residual = residuals.mean(axis=-1)
        centered = residuals - residual[:, None]
        other_residual = residual[:, None] - centered / (m - 1)
        regression = (
            residual * residual - centered * centered.mean(axis=-1)[:, None] / (m - 1)
        ) / (2 * beta)
        corrected = current.sum(axis=-1)[:, None] - mc_current.sum(axis=1)
        loss = -(other_residual * corrected).mean(axis=-1).mean()
        local_kl = ratio.mean(axis=-1)
        return loss, local_kl, regression.mean(), {
            "residual": residual,
            "mc_samples": mx.array(m),
        }

    p_log, q_log = head if estimator == "topk" else full
    p_log = p_log.astype(mx.float32)
    q_log = mx.stop_gradient(q_log.astype(mx.float32))
    q = mx.exp(q_log)
    safe_q_log = mx.where(mx.isfinite(q_log), q_log, 0)
    p_detached = mx.stop_gradient(p_log)

    if estimator == "full":
        local_kl = (q * (safe_q_log - p_log)).sum(axis=-1)
        correction = local_kl
        extra = {}
    else:
        if not math.isfinite(tail_floor) or not 0 < tail_floor < 1:
            raise ValueError("tail_floor must be finite and in (0, 1)")
        p = mx.exp(p_log)
        p_for_compare = mx.exp(p_detached)
        sampler_tail = 1 - q.sum(axis=-1)
        trainer_tail = 1 - p.sum(axis=-1)
        trainer_tail_for_compare = 1 - p_for_compare.sum(axis=-1)
        q_tail = mx.maximum(sampler_tail, tail_floor)
        p_tail = mx.maximum(trainer_tail, tail_floor)
        local_kl = (q * (safe_q_log - p_log)).sum(axis=-1)
        local_kl = local_kl + q_tail * (mx.log(q_tail) - mx.log(p_tail))
        extra = {
            "floored_tokens": mx.sum(
                ((sampler_tail < tail_floor) | (trainer_tail_for_compare < tail_floor))
                & mask
            )
        }
        correction = local_kl

    local_kl = mx.where(mask, local_kl, 0)
    residual = mx.stop_gradient(
        rewards - beta * (current - behavior + local_kl).sum(axis=-1)
    )
    regression = residual * residual / (2 * beta)
    loss = -(residual * (current + correction).sum(axis=-1)).mean()
    return loss, local_kl, regression.mean(), {"residual": residual, **extra}


def _pad_record_rows(rows, prompt_lengths, start, width, tail_shape=()):
    """Align variable-length generation records to the GRPO dense mask."""
    padded = []
    for row, prompt_length in zip(rows, prompt_lengths):
        row = mx.array(row)
        offset = int(prompt_length) - 1 - start
        right = width - offset - row.shape[0]
        if right < 0 or offset < 0:
            raise ValueError("KLPO record does not align with the generated completion")
        pad_spec = ((offset, right),) + tuple((0, 0) for _ in tail_shape)
        padded.append(mx.pad(row, pad_spec))
    if not padded:
        return mx.zeros((0, width) + tuple(tail_shape), dtype=mx.float32)
    return mx.stack(padded)


def _prepare_klpo_records(batch, completions, batch_indices, records, inputs, start):
    prompt_lengths = [len(batch[0][idx]) for idx in batch_indices]
    width = inputs.shape[1] - 1 - start
    action = _pad_record_rows(
        [record["action_logps"] for record in records],
        prompt_lengths,
        start,
        width,
    )
    result = {"action": action}

    if records and records[0].get("mc_ids") is not None:
        m = records[0]["mc_ids"].shape[-1]
        result["mc_ids"] = _pad_record_rows(
            [record["mc_ids"] for record in records],
            prompt_lengths,
            start,
            width,
            (m,),
        ).astype(mx.int32)
        result["mc_behavior"] = _pad_record_rows(
            [record["mc_logps"] for record in records],
            prompt_lengths,
            start,
            width,
            (m,),
        )

    if records and records[0].get("head_ids") is not None:
        k = records[0]["head_ids"].shape[-1]
        result["head_ids"] = _pad_record_rows(
            [record["head_ids"] for record in records],
            prompt_lengths,
            start,
            width,
            (k,),
        ).astype(mx.int32)
        result["head_behavior"] = _pad_record_rows(
            [record["head_logps"] for record in records],
            prompt_lengths,
            start,
            width,
            (k,),
        )

    if records and records[0].get("full_logps") is not None:
        vocab = records[0]["full_logps"].shape[-1]
        result["full_behavior"] = _pad_record_rows(
            [record["full_logps"] for record in records],
            prompt_lengths,
            start,
            width,
            (vocab,),
        )
    return result


def _get_full_logps(model, inputs, mask, start):
    logits = model(inputs[:, :-1])[:, start:].astype(mx.float32)
    logits = mx.where(mask[..., None], logits, 0)
    return logits - mx.logsumexp(logits, axis=-1, keepdims=True)


def _gather_logps(logps, targets, mask):
    gathered = mx.take_along_axis(logps, targets[..., None], axis=-1).squeeze(-1)
    return mx.where(mask, gathered, 0)


def _gather_aux_logps(logps, targets, mask):
    samples = targets.shape[-1]
    expanded = mx.broadcast_to(logps[..., None, :], logps.shape[:-1] + (samples, logps.shape[-1]))
    gathered = mx.take_along_axis(expanded, targets[..., None], axis=-1).squeeze(-1)
    return mx.where(mask[..., None], gathered, 0)


def _validate_klpo_args(route, estimator, mc_samples, top_k, beta, tail_floor):
    if route not in {"token", "sequence"}:
        raise ValueError("KLPO route must be 'token' or 'sequence'.")
    if estimator not in {"mc", "topk", "binary", "full"}:
        raise ValueError("KLPO estimator must be 'mc', 'topk', 'binary', or 'full'.")
    if not isinstance(mc_samples, int) or mc_samples < 1:
        raise ValueError("KLPO mc_samples must be a positive integer.")
    if route == "sequence" and estimator == "mc" and mc_samples < 2:
        raise ValueError("Sequence MC-KL requires mc_samples >= 2.")
    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError("KLPO top_k must be a positive integer.")
    if beta <= 0 or not math.isfinite(beta):
        raise ValueError("KLPO beta must be positive and finite.")
    if not 0 < tail_floor < 1 or not math.isfinite(tail_floor):
        raise ValueError("KLPO tail_floor must be finite and in (0, 1).")


def klpo_loss(
    model,
    batch,
    completions,
    rollout_records,
    batch_indices,
    rewards,
    reward_metrics=None,
    beta=0.1,
    route="token",
    kl_estimator="mc",
    mc_samples=128,
    top_k=128,
    tail_floor=1e-6,
    max_tokens=512,
    **_,
):
    """Compute a KLPO surrogate on complete sampled responses."""
    _validate_klpo_args(route, kl_estimator, mc_samples, top_k, beta, tail_floor)
    if not completions or len(completions) != len(rollout_records):
        raise ValueError("KLPO requires one rollout record per completion.")
    if any(completion.size < 1 for completion in completions):
        raise ValueError("KLPO requires complete responses with at least one token.")

    inputs, mask, lengths = _prepare_grpo_inputs(batch, completions, batch_indices)
    start = min(len(batch[0][idx]) for idx in batch_indices) - 1
    mask = mask[:, start:]
    records = _prepare_klpo_records(
        batch, completions, batch_indices, rollout_records, inputs, start
    )
    behavior = records["action"]

    needs_full = kl_estimator in {"mc", "topk", "full"}
    if needs_full:
        current_full = _get_full_logps(model, inputs, mask, start)
        current = _gather_logps(current_full, inputs[:, start + 1 :], mask)
    else:
        from .grpo_trainer import _get_token_logps

        current = _get_token_logps(model, inputs, mask, start)
        current_full = None

    aux = head = full = None
    if kl_estimator == "mc":
        ids = records["mc_ids"]
        aux = (
            _gather_aux_logps(current_full, ids, mask),
            records["mc_behavior"],
        )
    elif kl_estimator == "topk":
        ids = records["head_ids"]
        head = (
            mx.take_along_axis(current_full, ids, axis=-1),
            records["head_behavior"],
        )
    elif kl_estimator == "full":
        full = (current_full, records["full_behavior"])

    if route == "token":
        loss, local_kl, regression, extra = _token_loss(
            current,
            behavior,
            rewards,
            mask,
            estimator=kl_estimator,
            beta=beta,
            aux=aux,
            head=head,
            full=full,
            tail_floor=tail_floor,
        )
    else:
        loss, local_kl, regression, extra = _sequence_loss(
            current,
            behavior,
            rewards,
            mask,
            estimator=kl_estimator,
            beta=beta,
            aux=aux,
            head=head,
            full=full,
            tail_floor=tail_floor,
        )

    counts = mask.sum(axis=1)
    metrics = {
        "kl": (local_kl.sum(axis=-1) / mx.maximum(counts, 1)).mean(),
        "regression_loss": regression,
        "average_generated_tokens": lengths.astype(mx.float32).mean(),
        "max_generated_tokens": lengths.max(),
        "min_generated_tokens": lengths.min(),
        "hit_max_tokens_ratio": (lengths >= max_tokens).astype(mx.float32).mean(),
        "reward_mean": rewards.astype(mx.float32).mean(),
        "reward_std": rewards.astype(mx.float32).std(),
        "floored_tokens": extra.get("floored_tokens", mx.array(0.0)),
        "mc_samples": extra.get("mc_samples", mx.array(0.0)),
    }
    metrics.update(reward_metrics or {})
    return loss, mask.sum(), metrics


class _KLPORecords:
    def __init__(self, estimator, mc_samples, top_k):
        self.action_logps = []
        self.mc_ids = [] if estimator == "mc" else None
        self.mc_logps = [] if estimator == "mc" else None
        self.head_ids = [] if estimator == "topk" else None
        self.head_logps = [] if estimator == "topk" else None
        self.full_logps = [] if estimator == "full" else None
        self.estimator = estimator
        self.mc_samples = mc_samples
        self.top_k = top_k

    def as_dict(self, length=None):
        def stack(values, tail=()):
            if length is not None:
                values = values[:length]
            if values:
                return mx.stack(values)
            return None

        return {
            "action_logps": stack(self.action_logps),
            "mc_ids": stack(self.mc_ids, (self.mc_samples,)) if self.mc_ids is not None else None,
            "mc_logps": stack(self.mc_logps, (self.mc_samples,)) if self.mc_logps is not None else None,
            "head_ids": stack(self.head_ids, (self.top_k,)) if self.head_ids is not None else None,
            "head_logps": stack(self.head_logps, (self.top_k,)) if self.head_logps is not None else None,
            "full_logps": stack(self.full_logps) if self.full_logps is not None else None,
        }


def _recording_sampler(record, temperature):
    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("KLPO temperature must be positive and finite.")

    def sampler(logprobs):
        logits = logprobs.astype(mx.float32) / temperature
        q_logps = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        sampled = mx.random.categorical(logits)
        action_logp = mx.take_along_axis(
            q_logps, sampled[..., None], axis=-1
        ).squeeze(-1)
        record.action_logps.append(action_logp.squeeze(0))

        if record.estimator == "mc":
            draws = mx.random.categorical(
                mx.broadcast_to(logits, (record.mc_samples, logits.shape[-1]))
            )
            draw_logps = mx.take_along_axis(
                mx.broadcast_to(q_logps, (record.mc_samples, q_logps.shape[-1])),
                draws[..., None],
                axis=-1,
            ).squeeze(-1)
            record.mc_ids.append(draws.astype(mx.int32))
            record.mc_logps.append(draw_logps)
        elif record.estimator == "topk":
            vocab = q_logps.shape[-1]
            k = min(record.top_k, vocab)
            record.top_k = k
            ids = mx.argpartition(-q_logps, kth=k - 1, axis=-1)[..., :k]
            head_logps = mx.take_along_axis(q_logps, ids, axis=-1)
            record.head_ids.append(ids.squeeze(0).astype(mx.int32))
            record.head_logps.append(head_logps.squeeze(0))
        elif record.estimator == "full":
            record.full_logps.append(q_logps.squeeze(0))
        return sampled

    return sampler


def generate_klpo(
    model,
    tokenizer,
    prompt_tokens,
    max_tokens,
    batch_size,
    end_token,
    temperature,
    kl_estimator,
    mc_samples,
    top_k,
):
    """Generate one response per prompt and capture KLPO sampler records."""
    if not prompt_tokens or min(max_tokens, batch_size) < 1:
        raise ValueError("Prompts and positive generation sizes are required.")
    stop_tokens = [[token] for token in tokenizer.eos_token_ids]
    if end_token:
        end_ids = tokenizer.encode(end_token, add_special_tokens=False)
        if end_ids and end_ids not in stop_tokens:
            stop_tokens.append(end_ids)

    was_training = model.training
    model.eval()
    try:
        completions, texts, records = [], [], []
        for start in range(0, len(prompt_tokens), batch_size):
            indices = list(range(start, min(start + batch_size, len(prompt_tokens))))
            prompts = [
                prompt_tokens[idx].tolist()
                if isinstance(prompt_tokens[idx], mx.array)
                else list(prompt_tokens[idx])
                for idx in indices
            ]
            stores = [
                _KLPORecords(kl_estimator, mc_samples, top_k) for _ in prompts
            ]
            generator = BatchGenerator(
                model,
                stop_tokens=stop_tokens,
                # KLPO captures the exact distribution used by each sampler.
                sampler=None,
                completion_batch_size=batch_size,
                prefill_batch_size=batch_size,
            )
            try:
                uids = generator.insert(
                    prompts,
                    [max_tokens] * len(prompts),
                    samplers=[_recording_sampler(store, temperature) for store in stores],
                )
                tokens = {uid: [] for uid in uids}
                while responses := generator.next_generated():
                    for response in responses:
                        tokens[response.uid].append(response.token)
                for uid, store in zip(uids, stores):
                    ids = tokens[uid]
                    completions.append(mx.array(ids, dtype=mx.int32))
                    text_ids = (
                        ids[:-1] if ids and ids[-1] in tokenizer.eos_token_ids else ids
                    )
                    texts.append(tokenizer.decode(text_ids))
                    # BatchGenerator evaluates one look-ahead sampler step before
                    # returning the final response token. Only records aligned
                    # with returned completion tokens belong to the rollout.
                    records.append(store.as_dict(len(ids)))
            finally:
                generator.close()
        return completions, texts, records, list(range(len(prompt_tokens)))
    finally:
        model.train(was_training)


def calculate_rewards(
    reward_funcs,
    expanded_prompts,
    completion_texts,
    expanded_answers,
    expanded_types,
    reward_weights=None,
):
    """Evaluate terminal rewards once without GRPO group normalization."""
    if not reward_funcs or not completion_texts:
        raise ValueError("At least one reward function and completion are required.")
    if reward_weights is not None and len(reward_weights) != len(reward_funcs):
        raise ValueError("Number of reward weights must match number of reward functions")
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
            completions=completion_texts,
            answer=expanded_answers,
            types=expanded_types,
        )
        if raw is None:
            raw = [None] * len(completion_texts)
        if len(raw) != len(completion_texts):
            raise ValueError(f"{reward_func.__name__} must return one reward per completion.")
        values = np.asarray([np.nan if value is None else float(value) for value in raw])
        if np.isinf(values).any():
            raise ValueError(f"{reward_func.__name__} returned an infinite reward.")
        valid = values[~np.isnan(values)]
        metrics[f"{reward_func.__name__}_mean"] = float(valid.mean()) if valid.size else 0.0
        metrics[f"{reward_func.__name__}_std"] = float(valid.std()) if valid.size else 0.0
        metrics[f"{reward_func.__name__}_coverage"] = valid.size / values.size
        columns.append(values)
    rewards = np.stack(columns, axis=1)
    if np.isnan(rewards).all(axis=1).any():
        raise RuntimeError("All reward functions returned None or NaN for a completion.")
    rewards = (np.nan_to_num(rewards, nan=0.0) * weights).sum(axis=1)
    if not np.isfinite(rewards).all():
        raise ValueError("Weighted rewards must be finite.")
    metrics.update(reward_mean=float(rewards.mean()), reward_std=float(rewards.std()))
    return mx.array(rewards, dtype=mx.float32), {
        key: mx.array(value, dtype=mx.float32) for key, value in metrics.items()
    }


def _rollout_rewards(batch, texts, indices, reward_funcs, reward_weights):
    _, _, prompts, answers, types = batch
    return calculate_rewards(
        reward_funcs,
        [prompts[i] for i in indices],
        texts,
        [answers[i] for i in indices],
        [types[i] if types is not None else None for i in indices],
        reward_weights,
    )


def _klpo_microbatches(compute, model, chunk_size, *, with_grad=False, **kwargs):
    completions = kwargs.pop("completions")
    texts = kwargs.pop("completion_texts")
    indices = kwargs.pop("batch_indices")
    records = kwargs.pop("rollout_records")
    rewards = kwargs.pop("rewards")
    sample_count = len(completions)
    total_loss, total_tokens, all_metrics, accumulated = 0, 0, None, None
    for start in range(0, sample_count, chunk_size):
        stop = min(start + chunk_size, sample_count)
        weight = (stop - start) / sample_count
        result = compute(
            model,
            completions=completions[start:stop],
            completion_texts=texts[start:stop],
            batch_indices=indices[start:stop],
            rollout_records=records[start:stop],
            rewards=rewards[start:stop],
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
        else:
            loss, tokens, metrics = result
        total_loss = total_loss + loss * weight
        total_tokens = total_tokens + tokens
        weighted = {
            key: value * weight
            for key, value in metrics.items()
            if key not in {"max_generated_tokens", "min_generated_tokens"}
        }
        if all_metrics is None:
            all_metrics = dict(weighted)
            all_metrics["max_generated_tokens"] = metrics["max_generated_tokens"]
            all_metrics["min_generated_tokens"] = metrics["min_generated_tokens"]
        else:
            for key, value in weighted.items():
                all_metrics[key] = all_metrics[key] + value
            all_metrics["max_generated_tokens"] = mx.maximum(
                all_metrics["max_generated_tokens"], metrics["max_generated_tokens"]
            )
            all_metrics["min_generated_tokens"] = mx.minimum(
                all_metrics["min_generated_tokens"], metrics["min_generated_tokens"]
            )
        mx.eval(total_loss, total_tokens, all_metrics, accumulated)
    result = total_loss, total_tokens, all_metrics
    return (result, accumulated) if with_grad else result


def _klpo_value_and_grad(loss_value_and_grad, model, chunk_size, **kwargs):
    return _klpo_microbatches(
        loss_value_and_grad, model, chunk_size, with_grad=True, **kwargs
    )


def evaluate_klpo(
    model,
    dataset,
    tokenizer,
    batch_size,
    num_batches,
    beta,
    route,
    kl_estimator,
    mc_samples,
    top_k,
    tail_floor,
    max_seq_length,
    max_tokens,
    temperature,
    reward_funcs=None,
    reward_weights=None,
    iterate_batches=iterate_grpo_batches,
    end_answer_token="</answer>",
):
    reward_funcs = reward_funcs or [
        r1_accuracy_reward_func,
        r1_int_reward_func,
        r1_strict_format_reward_func,
        r1_soft_format_reward_func,
        r1_count_xml,
    ]
    was_training = model.training
    try:
        model.eval()
        all_losses, ntokens, responses_seen, all_metrics = 0, 0, mx.array(0), None
        index_iterator = iter(range(num_batches)) if num_batches != -1 else iter(int, 1)
        for _, batch in zip(
            index_iterator,
            iterate_batches(dataset=dataset, batch_size=batch_size, max_seq_length=max_seq_length),
        ):
            prompt_tokens, answer_tokens, prompt_text, answer_text, type_info = batch
            completions, texts, records, indices = generate_klpo(
                model, tokenizer, prompt_tokens, max_tokens, batch_size,
                end_answer_token, temperature, kl_estimator, mc_samples, top_k,
            )
            rewards, reward_metrics = _rollout_rewards(
                batch, texts, indices, reward_funcs, reward_weights
            )
            loss, tokens, metrics = _klpo_microbatches(
                klpo_loss, model, max(len(prompt_tokens), 1),
                batch=(prompt_tokens, answer_tokens, prompt_text, answer_text, type_info),
                completions=completions,
                completion_texts=texts,
                batch_indices=indices,
                rollout_records=records,
                rewards=rewards,
                reward_metrics=reward_metrics,
                beta=beta,
                route=route,
                kl_estimator=kl_estimator,
                mc_samples=mc_samples,
                top_k=top_k,
                tail_floor=tail_floor,
                max_tokens=max_tokens,
            )
            response_count = len(completions)
            all_losses += loss * response_count
            ntokens += tokens
            responses_seen += response_count
            if all_metrics is None:
                all_metrics = {key: value * response_count for key, value in metrics.items()}
            else:
                for key, value in metrics.items():
                    all_metrics[key] += value * response_count
            mx.eval(all_losses, ntokens, responses_seen, *all_metrics.values())
        if all_metrics is None:
            raise ValueError("Evaluation requires at least one batch.")
        all_losses = mx.distributed.all_sum(all_losses, stream=mx.cpu)
        ntokens = mx.distributed.all_sum(ntokens, stream=mx.cpu)
        responses_seen = mx.distributed.all_sum(responses_seen, stream=mx.cpu)
        all_metrics = {key: mx.distributed.all_sum(value) for key, value in all_metrics.items()}
        avg_metrics = {
            key: (value / mx.maximum(responses_seen, 1)).item()
            for key, value in all_metrics.items()
        }
        return (all_losses / mx.maximum(responses_seen, 1)).item(), ntokens, avg_metrics
    finally:
        model.train(was_training)


def train_klpo(
    model,
    tokenizer,
    optimizer,
    train_dataset,
    val_dataset=None,
    reward_funcs=None,
    args=None,
    training_callback: TrainingCallback = None,
    end_answer_token="</answer>",
):
    args = args or KLPOTrainingArgs()
    reward_funcs = reward_funcs or [
        r1_accuracy_reward_func,
        r1_int_reward_func,
        r1_strict_format_reward_func,
        r1_soft_format_reward_func,
        r1_count_xml,
    ]
    _validate_klpo_args(
        args.route, args.kl_estimator, args.mc_samples, args.top_k, args.beta, args.tail_floor
    )
    if model_uses_recurrence(model):
        enable_memory_safe_recurrences(chunk_size=args.recurrence_chunk_size)
    world = mx.distributed.init()
    world_size, rank = world.size(), world.rank()
    if world_size > 1:
        tqdm.write(f"Node {rank} of {world_size}")
    if args.grad_checkpoint:
        grad_checkpoint(model.layers[0])
    grad_accum_steps = args.gradient_accumulation_steps
    if grad_accum_steps < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1")

    state = [model.state, optimizer.state, mx.random.state]
    loss_value_and_grad = nn.value_and_grad(model, klpo_loss)

    def step(batch, completions, texts, indices, records, rewards, reward_metrics, previous_grad, update, accumulation_count):
        prompt_tokens, answer_tokens, prompt_text, answer_text, type_info = batch
        (loss, tokens, metrics), grads = _klpo_value_and_grad(
            loss_value_and_grad,
            model,
            max(len(prompt_tokens), 1),
            batch=(prompt_tokens, answer_tokens, prompt_text, answer_text, type_info),
            completions=completions,
            completion_texts=texts,
            batch_indices=indices,
            rollout_records=records,
            rewards=rewards,
            reward_metrics=reward_metrics,
            beta=args.beta,
            route=args.route,
            kl_estimator=args.kl_estimator,
            mc_samples=args.mc_samples,
            top_k=args.top_k,
            tail_floor=args.tail_floor,
            max_tokens=args.max_completion_length,
        )
        if previous_grad is not None:
            grads = tree_map(lambda left, right: left + right, grads, previous_grad)
        finite = mx.isfinite(loss) & mx.all(
            mx.stack([mx.all(mx.isfinite(value)) for _, value in tree_flatten(grads)])
        )
        if not finite.item():
            raise FloatingPointError("Non-finite KLPO loss or gradients; update aborted.")
        if update:
            grads = average_gradients(grads)
            if grad_accum_steps > 1:
                grads = tree_map(lambda value: value / accumulation_count, grads)
            optimizer.update(model, grads)
            grads = None
        return loss, tokens, metrics, grads

    model.train()
    losses, n_tokens, steps, trained_tokens = 0, 0, 0, 0
    accumulated_metrics = {
        "reward_mean": 0,
        "reward_std": 0,
        "kl": 0,
        "regression_loss": 0,
        "floored_tokens": 0,
        "mc_samples": 0,
        "average_generated_tokens": 0,
        "max_generated_tokens": 0,
        "min_generated_tokens": 0,
        "hit_max_tokens_ratio": 0,
    }
    for reward_func in reward_funcs:
        name = reward_func.__name__
        accumulated_metrics[f"{name}_mean"] = 0
        accumulated_metrics[f"{name}_std"] = 0
        accumulated_metrics[f"{name}_coverage"] = 0

    grad_accum = None
    start_time = time.perf_counter()
    pbar = tqdm(range(1, args.iters + 1), desc="KLPO training", disable=rank != 0)
    batches = iter(
        iterate_grpo_batches(
            dataset=train_dataset,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
            train=True,
        )
    )
    for iteration in pbar:
        batch = next(batches)
        if val_dataset is not None and len(val_dataset) > 0 and (
            iteration == 1 or iteration % args.steps_per_eval == 0 or iteration == args.iters
        ):
            val_loss, val_tokens, _val_metrics = evaluate_klpo(
                model, val_dataset, tokenizer, args.batch_size, args.val_batches,
                args.beta, args.route, args.kl_estimator, args.mc_samples, args.top_k,
                args.tail_floor, args.max_seq_length, args.max_completion_length,
                args.temperature, reward_funcs, args.reward_weights,
                end_answer_token=end_answer_token,
            )
            if rank == 0:
                tqdm.write(f"Iter {iteration}: Val loss {val_loss:.3f}")
            if training_callback is not None:
                training_callback.on_val_loss_report(
                    {"iteration": iteration, "val_loss": val_loss, "val_tokens": val_tokens.item()}
                )
            model.train()
            start_time = time.perf_counter()

        prompt_tokens, _, _, _, _ = batch
        completions, texts, records, indices = generate_klpo(
            model, tokenizer, prompt_tokens, args.max_completion_length,
            args.batch_size, end_answer_token, args.temperature, args.kl_estimator,
            args.mc_samples, args.top_k,
        )
        rewards, reward_metrics = _rollout_rewards(
            batch, texts, indices, reward_funcs, args.reward_weights
        )
        loss, tokens, metrics, grad_accum = step(
            batch, completions, texts, indices, records, rewards, reward_metrics,
            grad_accum,
            iteration % grad_accum_steps == 0 or iteration == args.iters,
            (iteration - 1) % grad_accum_steps + 1,
        )
        losses += loss
        n_tokens += tokens
        steps += 1
        for key, value in metrics.items():
            accumulated_metrics[key] += value
        mx.eval(state, losses, n_tokens, grad_accum, *accumulated_metrics.values())

        if iteration % args.steps_per_report == 0 or iteration == args.iters:
            elapsed = time.perf_counter() - start_time
            train_loss = mx.distributed.all_sum(losses).item() / (steps * world_size)
            reduced = {
                key: mx.distributed.all_sum(mx.array(value)) / (steps * world_size)
                for key, value in accumulated_metrics.items()
            }
            mx.eval(reduced)
            avg = {key: value.item() for key, value in reduced.items()}
            total_tokens = mx.distributed.all_sum(n_tokens).item()
            trained_tokens += total_tokens
            if rank == 0:
                pbar.set_postfix({"loss": f"{train_loss:.3f}", "it/s": f"{steps / elapsed:.3f}"})
                tqdm.write(
                    f"\nIter {iteration}: KLPO loss={train_loss:.4f}, "
                    f"reward={avg['reward_mean']:.4f}, KL={avg['kl']:.6f}, "
                    f"tokens={total_tokens}, route={args.route}/{args.kl_estimator}"
                )
            if training_callback is not None:
                training_callback.on_train_loss_report(
                    {
                        "iteration": iteration,
                        "train_loss": train_loss,
                        **{f"train_{key}": value for key, value in avg.items()},
                        "trained_tokens": trained_tokens,
                    }
                )
            losses, n_tokens, steps = 0, 0, 0
            accumulated_metrics = {key: 0 for key in accumulated_metrics}
            start_time = time.perf_counter()

        if iteration % args.steps_per_save == 0:
            adapter_weights = dict(tree_flatten(model.trainable_parameters()))
            mx.save_safetensors(str(args.adapter_file), adapter_weights)
            checkpoint = Path(args.adapter_file).parent / f"{iteration:07d}_adapters.safetensors"
            mx.save_safetensors(str(checkpoint), adapter_weights)
            tqdm.write(f"Iter {iteration}: Saved adapter weights to {args.adapter_file} and {checkpoint}.")

    adapter_weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(args.adapter_file), adapter_weights)
    tqdm.write(f"Saved final weights to {args.adapter_file}.")
