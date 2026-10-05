"""Directional and Similarity-aware Latent Alignment (DSLA) primitives."""

import math
import time
from dataclasses import dataclass, field
from functools import wraps
from typing import Mapping, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.nn.utils import average_gradients
from mlx.utils import tree_map
from mlx_lm.tuner.callbacks import TrainingCallback
from tqdm import tqdm

from ..recurrent_patch import enable_memory_safe_recurrences, model_uses_recurrence
from .cpo_trainer import cpo_loss
from .dpo_trainer import dpo_loss
from .sft_trainer import (
    SFTTrainingArgs,
    _symmetric_fake_quantize_tensor,
    grad_checkpoint,
)
from .training_utils import save_adapters


@dataclass
class DSLATrainingArgs(SFTTrainingArgs):
    """DSLA configuration, independent of the underlying preference trainer."""

    loss_type: str = field(
        default="dpo",
        metadata={"help": "Preference objective: 'dpo', 'orpo', or 'cpo'."},
    )
    dpo_cpo_loss_type: str = field(
        default="sigmoid", metadata={"help": "DPO/CPO margin loss variant."}
    )
    beta: float = field(default=0.1, metadata={"help": "Preference loss scale."})
    delta: float = field(default=50.0, metadata={"help": "DPOP penalty scale."})

    latent_weight: float = field(
        default=0.1, metadata={"help": "Weight of the DSLA hidden-state regularizer."}
    )
    latent_margin: float = field(
        default=0.05, metadata={"help": "Target prompt-response cosine margin."}
    )
    latent_gamma: float = field(
        default=10.0, metadata={"help": "Sharpness of the DSLA soft-margin losses."}
    )
    latent_variant: str = field(
        default="both",
        metadata={"help": "DSLA component: 'similarity', 'direction', or 'both'."},
    )
    latent_pooling: str = field(
        default="answer_mean",
        metadata={
            "help": (
                "Hidden-state pooling: 'answer_mean', 'last_token', "
                "'last_k_mean', or 'prompt_answer_mean'."
            )
        },
    )
    latent_layer: str = field(
        default="final",
        metadata={
            "help": "Residual-stream anchor: 'final', 'middle', 'late', or an index."
        },
    )

    def __post_init__(self):
        if self.loss_type not in ("dpo", "orpo", "cpo"):
            raise ValueError("DSLA loss_type must be 'dpo', 'orpo', or 'cpo'")
        if self.dpo_cpo_loss_type not in ("sigmoid", "hinge", "ipo", "dpop"):
            raise ValueError("Unknown DSLA DPO/CPO loss variant")
        if not math.isfinite(self.beta) or self.beta <= 0:
            raise ValueError("beta must be finite and positive")
        if not math.isfinite(self.delta) or self.delta < 0:
            raise ValueError("delta must be finite and nonnegative")
        if self.qat_enable and not 2 <= self.qat_bits <= 16:
            raise ValueError("qat_bits must be in [2, 16]")
        if self.qat_enable and self.qat_group_size < 0:
            raise ValueError("qat_group_size must be nonnegative")
        for name in ("latent_weight", "latent_margin", "latent_gamma"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.latent_gamma == 0:
            raise ValueError("latent_gamma must be positive")
        if self.latent_variant not in ("similarity", "direction", "both"):
            raise ValueError(
                "latent_variant must be 'similarity', 'direction', or 'both'"
            )
        if self.latent_pooling not in (
            "answer_mean",
            "last_token",
            "last_k_mean",
            "prompt_answer_mean",
        ):
            raise ValueError(f"Unknown DSLA pooling: {self.latent_pooling}")


_LATENT_REPORT_LABELS = (
    ("latent_loss", "latent_loss"),
    ("latent_sim_loss", "sim_loss"),
    ("latent_sim_margin", "sim_margin"),
    ("latent_dir_loss", "dir_loss"),
    ("latent_dir_margin", "dir_margin"),
)


def format_latent_metrics(metrics: Mapping[str, float]) -> str:
    """Format the active DSLA metrics for terminal training reports."""
    parts = [
        f"{label} {float(metrics[key]):.3f}"
        for key, label in _LATENT_REPORT_LABELS
        if key in metrics
    ]
    return (", " + ", ".join(parts)) if parts else ""


def validate_prompt_length(example, prefix: str, sequence_length: int) -> int:
    """Require a nonempty prompt and response after DSLA truncation."""
    prompt_length = example.get(f"{prefix}_prompt_length")
    if not isinstance(prompt_length, int) or not 0 < prompt_length < sequence_length:
        raise ValueError(
            f"DSLA requires {prefix}_prompt_length with a nonempty prompt and "
            "response after truncation; check the chat template and max_seq_length"
        )
    return prompt_length


def forward_logits_and_hidden(
    model: nn.Module, tokens: mx.array, layer_spec="final", cache=None
) -> Tuple[mx.array, mx.array]:
    """Capture hidden states from the native forward, preserving masks and heads.

    The short-lived hook observes one module instance while MLX traces the
    forward. It preserves checkpoint wrappers, caches, tied embeddings, and
    architecture-specific masks and logit scaling without a second forward.
    """
    language_model = getattr(model, "language_model", model)
    backbone = getattr(language_model, "model", None)
    if backbone is None:
        raise ValueError(
            "DSLA requires a model.model (or language_model.model) backbone; "
            "logits cannot be used as residual-stream hidden states"
        )

    target = backbone
    if layer_spec != "final":
        layers = backbone.layers
        if layer_spec == "middle":
            layer_index = len(layers) // 2
        elif layer_spec == "late":
            layer_index = int(0.8 * len(layers))
        else:
            try:
                layer_index = int(layer_spec)
            except (TypeError, ValueError) as error:
                raise ValueError(f"Unknown DSLA layer: {layer_spec}") from error
        if not 0 <= layer_index < len(layers):
            raise ValueError(f"DSLA layer index must be in [0, {len(layers)})")
        target = layers[layer_index]

    module_type = type(target)
    original_call = module_type.__call__
    selected_hidden = None

    @wraps(original_call)
    def capture(module, *args, **kwargs):
        nonlocal selected_hidden
        output = original_call(module, *args, **kwargs)
        if module is target:
            selected_hidden = output
        return output

    module_type.__call__ = capture
    try:
        logits = model(tokens, cache=cache)
    finally:
        module_type.__call__ = original_call
    if not isinstance(selected_hidden, mx.array) or selected_hidden.ndim != 3:
        raise ValueError(
            "DSLA requires a backbone/layer returning [batch, tokens, hidden]"
        )
    return logits, selected_hidden


def _masked_mean(hidden, mask):
    # Pool and normalize in float32 even when the model uses half precision.
    hidden = hidden.astype(mx.float32)
    weights = mask.astype(mx.float32)[..., None]
    return (hidden * weights).sum(1) / mx.maximum(weights.sum(1), 1.0)


def _last_token(hidden, mask):
    # Reversing the mask makes argmax select the last active position.
    reverse_index = mx.argmax(mask[:, ::-1], axis=1).astype(mx.int32)
    index = mask.shape[1] - 1 - reverse_index
    values = hidden[mx.arange(hidden.shape[0]), index].astype(mx.float32)
    return mx.where(mask.sum(-1, keepdims=True) > 0, values, 0.0)


def _pool(hidden, response_mask, prompt_mask, pooling):
    if pooling == "answer_mean":
        return _masked_mean(hidden, response_mask)
    if pooling == "last_token":
        return _last_token(hidden, response_mask)
    if pooling == "last_k_mean":
        positions = mx.arange(response_mask.shape[1])[None, :]
        last = (
            response_mask.shape[1]
            - 1
            - mx.argmax(response_mask[:, ::-1], axis=1).astype(mx.int32)
        )
        last_k_mask = response_mask * (positions >= (last[:, None] - 7))
        return _masked_mean(hidden, last_k_mask)
    if pooling == "prompt_answer_mean":
        return _masked_mean(hidden, mx.minimum(prompt_mask + response_mask, 1.0))
    raise ValueError(f"Unknown DSLA pooling: {pooling}")


def latent_preference_loss(
    chosen_hidden,
    rejected_hidden,
    chosen_response_mask,
    rejected_response_mask,
    chosen_prompt_mask,
    rejected_prompt_mask,
    args,
):
    """Compute the preprint's similarity and batch-direction objectives.

    Masks index the hidden states of the actual tokens, independently of the
    one-position shift used to score next-token probabilities. The direction
    estimate remains differentiable and is computed over this microbatch.
    """
    chosen = _pool(
        chosen_hidden, chosen_response_mask, chosen_prompt_mask, args.latent_pooling
    )
    rejected = _pool(
        rejected_hidden,
        rejected_response_mask,
        rejected_prompt_mask,
        args.latent_pooling,
    )
    prompt = 0.5 * (
        _masked_mean(chosen_hidden, chosen_prompt_mask)
        + _masked_mean(rejected_hidden, rejected_prompt_mask)
    )

    def normalize(value):
        return value / mx.sqrt((value * value).sum(-1, keepdims=True) + 1e-6)

    losses = []
    metrics = {}
    if args.latent_variant in ("similarity", "both"):
        normalized_prompt = normalize(prompt)
        similarity_margin = (normalized_prompt * normalize(chosen)).sum(-1) - (
            normalized_prompt * normalize(rejected)
        ).sum(-1)
        similarity_loss = mx.logaddexp(
            (args.latent_margin - similarity_margin) * args.latent_gamma,
            mx.zeros_like(similarity_margin),
        ).mean()
        losses.append(similarity_loss)
        metrics.update(
            latent_sim_margin=similarity_margin.mean(),
            latent_sim_loss=similarity_loss,
        )
    if args.latent_variant in ("direction", "both"):
        directions = chosen - rejected
        mean_direction = directions.mean(0)
        # sqrt(max(||mean||^2, 1e-16)) equals max(||mean||, 1e-8),
        # with a finite derivative at zero (unlike sqrt(0) before clamping).
        batch_direction = mean_direction / mx.sqrt(
            mx.maximum((mean_direction * mean_direction).sum(), 1e-16)
        )
        directional_margin = (normalize(directions) * batch_direction).sum(-1)
        direction_loss = -nn.log_sigmoid(args.latent_gamma * directional_margin).mean()
        losses.append(direction_loss)
        metrics.update(
            latent_dir_margin=directional_margin.mean(),
            latent_dir_loss=direction_loss,
        )
    if not losses:
        raise ValueError("latent_variant must be 'similarity', 'direction', or 'both'")
    latent_loss = sum(losses) / len(losses)
    metrics["latent_loss"] = latent_loss
    return latent_loss, metrics


def iterate_dsla_batches(dataset, batch_size: int, max_seq_length: int, train=False):
    """Yield preference pairs with separate token-aligned response/prompt masks."""
    if batch_size < 1 or max_seq_length < 2:
        raise ValueError("DSLA requires batch_size >= 1 and max_seq_length >= 2")
    if len(dataset) < batch_size:
        raise ValueError(f"Dataset must have at least batch_size={batch_size} examples")
    world = mx.distributed.init()
    workers, rank = world.size(), world.rank()
    if batch_size % workers:
        raise ValueError("Batch size must be divisible by the number of workers")
    indices = sorted(
        range(len(dataset)),
        key=lambda i: max(len(dataset[i]["chosen"]), len(dataset[i]["rejected"])),
    )
    batch_indices = [
        indices[start + rank : start + batch_size : workers]
        for start in range(0, len(indices) - batch_size + 1, batch_size)
    ]
    while True:
        order = (
            np.random.permutation(len(batch_indices))
            if train
            else range(len(batch_indices))
        )
        for index in order:
            examples = [dataset[i] for i in batch_indices[index]]
            length = min(
                max(
                    len(example[key])
                    for example in examples
                    for key in ("chosen", "rejected")
                ),
                max_seq_length,
            )
            arrays, response_masks, prompt_masks = [], [], []
            for key in ("chosen", "rejected"):
                tokens = np.zeros((len(examples), length), dtype=np.int32)
                response = np.zeros(tokens.shape, dtype=np.float32)
                prompt = np.zeros(tokens.shape, dtype=np.float32)
                for row, example in enumerate(examples):
                    size = min(len(example[key]), length)
                    prefix = validate_prompt_length(example, key, size)
                    tokens[row, :size] = example[key][:size]
                    prompt[row, :prefix] = 1.0
                    response[row, prefix:size] = 1.0
                arrays.append(mx.array(tokens))
                response_masks.append(mx.array(response))
                prompt_masks.append(mx.array(prompt))
            yield (*arrays, *response_masks, *prompt_masks)
        if not train:
            break


def _sequence_logps(logits: mx.array, tokens: mx.array, response_mask: mx.array):
    """Return response log-probability sums, means, and masked logit means."""
    target_mask = response_mask[:, 1:]
    token_logps = -nn.losses.cross_entropy(
        logits.astype(mx.float32), tokens[:, 1:], reduction="none"
    )
    counts = mx.maximum(target_mask.sum(-1), 1.0)
    sums = (token_logps * target_mask).sum(-1)
    logits_mean = (
        logits.astype(mx.float32).mean(-1) * target_mask
    ).sum() / counts.sum()
    return sums, sums / counts, logits_mean


def _log_odds(logps: mx.array) -> mx.array:
    """Implement the preprint's logit_epsilon(exp(logps)), epsilon=1e-6."""
    logps = mx.minimum(logps, math.log1p(-1e-6))
    log_complement = mx.where(
        logps > -math.log(2.0),
        mx.log(-mx.expm1(logps)),
        mx.log1p(-mx.exp(logps)),
    )
    return logps - log_complement


def dsla_loss(
    model,
    chosen,
    rejected,
    chosen_response_mask,
    rejected_response_mask,
    chosen_prompt_mask,
    rejected_prompt_mask,
    args: DSLATrainingArgs,
    ref_model=None,
):
    """Combine a selectable preference objective with the DSLA latent loss.

    DPO and ORPO implement the objectives in the DSLA preprint. CPO and the
    alternative DPO/CPO margin losses are extensions using the same regularizer.
    """
    if args.loss_type == "dpo" and ref_model is None:
        raise ValueError("DSLA-DPO requires a frozen reference model")
    if args.loss_type == "dpo" and ref_model is model:
        raise ValueError("DSLA-DPO requires a reference model separate from the policy")
    chosen_logits, chosen_hidden = forward_logits_and_hidden(
        model, chosen, args.latent_layer
    )
    rejected_logits, rejected_hidden = forward_logits_and_hidden(
        model, rejected, args.latent_layer
    )
    chosen_sum, chosen_mean, chosen_logits_mean = _sequence_logps(
        chosen_logits[:, :-1], chosen, chosen_response_mask
    )
    rejected_sum, rejected_mean, rejected_logits_mean = _sequence_logps(
        rejected_logits[:, :-1], rejected, rejected_response_mask
    )
    if args.loss_type == "orpo":
        margin = _log_odds(chosen_mean) - _log_odds(rejected_mean)
        preference_loss = (-chosen_mean - args.beta * nn.log_sigmoid(margin)).mean()
        chosen_rewards, rejected_rewards = (
            args.beta * chosen_mean,
            args.beta * rejected_mean,
        )
    else:
        use_mean = args.dpo_cpo_loss_type == "ipo"
        chosen_score = chosen_mean if use_mean else chosen_sum
        rejected_score = rejected_mean if use_mean else rejected_sum
        loss_kwargs = dict(
            policy_chosen_score=chosen_score,
            policy_rejected_score=rejected_score,
            chosen_masks=chosen_response_mask[:, 1:],
            rejected_masks=rejected_response_mask[:, 1:],
            beta=args.beta,
            delta=args.delta,
            loss_type=args.dpo_cpo_loss_type,
        )
        reference_chosen, reference_rejected = mx.zeros_like(
            chosen_score
        ), mx.zeros_like(rejected_score)
        if args.loss_type == "dpo":
            reference_scores = []
            for tokens, mask in (
                (chosen, chosen_response_mask),
                (rejected, rejected_response_mask),
            ):
                logits = mx.stop_gradient(ref_model(tokens[:, :-1]))
                logp_sum, logp_mean, _ = _sequence_logps(logits, tokens, mask)
                reference_scores.append(logp_mean if use_mean else logp_sum)
            reference_chosen, reference_rejected = reference_scores
            loss_kwargs.update(
                reference_chosen_score=reference_chosen,
                reference_rejected_score=reference_rejected,
            )
            preference_loss = dpo_loss(**loss_kwargs)[0]
        else:
            preference_loss = cpo_loss(**loss_kwargs)[0]
        chosen_rewards = args.beta * (chosen_score - reference_chosen)
        rejected_rewards = args.beta * (rejected_score - reference_rejected)

    latent_loss, metrics = latent_preference_loss(
        chosen_hidden,
        rejected_hidden,
        chosen_response_mask,
        rejected_response_mask,
        chosen_prompt_mask,
        rejected_prompt_mask,
        args,
    )
    metrics.update(
        preference_loss=preference_loss,
        accuracies=(chosen_rewards > rejected_rewards).astype(mx.float32).mean(),
        margins=(chosen_rewards - rejected_rewards).mean(),
        policy_chosen_logps=chosen_mean.mean(),
        policy_rejected_logps=rejected_mean.mean(),
        chosen_logits_mean=chosen_logits_mean,
        rejected_logits_mean=rejected_logits_mean,
    )
    reward = mx.stack((chosen_rewards.mean(), rejected_rewards.mean()))
    tokens = chosen_response_mask[:, 1:].sum() + rejected_response_mask[:, 1:].sum()
    return preference_loss + args.latent_weight * latent_loss, reward, tokens, metrics


def evaluate_dsla(
    model, dataset, args: DSLATrainingArgs, ref_model=None, num_batches=-1
):
    """Evaluate the same combined objective and component metrics used in training."""
    if num_batches == 0 or num_batches < -1:
        raise ValueError("num_batches must be -1 or positive")
    model.eval()
    if ref_model is not None:
        ref_model.eval()
    total_loss, total_rewards, total_tokens, samples = 0.0, mx.zeros((2,)), 0.0, 0
    total_metrics = {}
    for index, batch in enumerate(
        iterate_dsla_batches(dataset, args.batch_size, args.max_seq_length)
    ):
        if num_batches != -1 and index >= num_batches:
            break
        loss, rewards, tokens, metrics = dsla_loss(
            model, *batch, args=args, ref_model=ref_model
        )
        batch_samples = batch[0].shape[0]
        total_loss += loss * batch_samples
        total_rewards += rewards * batch_samples
        total_tokens += tokens
        samples += batch_samples
        for key, value in metrics.items():
            total_metrics[key] = total_metrics.get(key, 0.0) + value * batch_samples
        mx.eval(total_loss, total_rewards, total_tokens, total_metrics)
    samples = mx.distributed.all_sum(mx.array(samples))
    total_loss = mx.distributed.all_sum(total_loss)
    total_rewards = mx.distributed.all_sum(total_rewards)
    total_tokens = mx.distributed.all_sum(total_tokens)
    total_metrics = {
        key: mx.distributed.all_sum(value) for key, value in total_metrics.items()
    }
    return (
        (total_loss / samples).item(),
        (total_rewards / samples).tolist(),
        total_tokens.item(),
        {key: (value / samples).item() for key, value in total_metrics.items()},
    )


def train_dsla(
    model,
    optimizer,
    train_dataset,
    val_dataset=None,
    args: DSLATrainingArgs = None,
    ref_model=None,
    training_callback: TrainingCallback = None,
):
    """Train DSLA with compiled updates, checkpointing, accumulation, and QAT."""
    args = DSLATrainingArgs() if args is None else args
    if args.seq_step_size is not None:
        raise ValueError(
            "DSLA does not support efficient_long_context: its latent objective "
            "requires pooled representations across the complete sequence. "
            "Use grad_checkpoint and recurrence_chunk_size to reduce memory."
        )
    if args.loss_type == "dpo" and ref_model is None:
        raise ValueError("DSLA-DPO requires a frozen reference model")
    if args.loss_type == "dpo" and ref_model is model:
        raise ValueError("DSLA-DPO requires a reference model separate from the policy")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1")
    if args.qat_start_step < 1:
        raise ValueError("qat_start_step must be at least 1")
    if args.iters < 1:
        raise ValueError("iters must be at least 1")
    if model_uses_recurrence(model):
        enable_memory_safe_recurrences(chunk_size=args.recurrence_chunk_size)
    world = mx.distributed.init()
    rank, workers = world.rank(), world.size()
    mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    checkpoint_originals = {}
    if args.grad_checkpoint:
        # Cover every block type in hybrid models, preserving native fast VJPs.
        checkpointed_types = set()
        for layer in model.layers:
            if type(layer) not in checkpointed_types:
                checkpoint_originals[type(layer)] = type(layer).__call__
                grad_checkpoint(layer)
                checkpointed_types.add(type(layer))
    if ref_model is not None:
        ref_model.eval()
    model.train()
    state = [model.state, optimizer.state, mx.random.state]
    # Include reference state as an input so compilation cannot bake in weights.
    if ref_model is not None:
        state.append(ref_model.state)

    def loss_wrapper(*batch):
        return dsla_loss(model, *batch, args=args, ref_model=ref_model)

    value_and_grad = nn.value_and_grad(model, loss_wrapper)

    def step_impl(batch, previous_grad, update, accumulation_count):
        result, grad = value_and_grad(*batch)
        if previous_grad is not None:
            grad = tree_map(
                lambda current, previous: current + previous, grad, previous_grad
            )
        if update:
            grad = average_gradients(grad)
            grad = tree_map(lambda value: value / accumulation_count, grad)
            optimizer.update(model, grad)
            grad = None
        return (*result, grad)

    step = mx.compile(step_impl, inputs=state, outputs=state)
    batches = iterate_dsla_batches(
        train_dataset, args.batch_size, args.max_seq_length, train=True
    )
    losses, rewards, tokens, metrics, steps = 0.0, mx.zeros((2,)), 0.0, {}, 0
    grad, accumulated_steps, optimizer_step, trained_tokens = None, 0, 0, 0
    qat_installed = False
    start = time.perf_counter()
    qat_originals = {}
    try:
        for iteration in tqdm(
            range(1, args.iters + 1), desc="DSLA training", disable=rank != 0
        ):
            if (
                val_dataset is not None
                and len(val_dataset) > 0
                and args.steps_per_eval is not None
                and (
                    iteration == 1
                    or iteration % args.steps_per_eval == 0
                    or iteration == args.iters
                )
            ):
                val_start = time.perf_counter()
                val_loss, val_rewards, _, val_metrics = evaluate_dsla(
                    model, val_dataset, args, ref_model, args.val_batches
                )
                val_time = time.perf_counter() - val_start
                if rank == 0:
                    tqdm.write(
                        f"Iter {iteration}: Val loss {val_loss:.3f}{format_latent_metrics(val_metrics)}"
                    )
                if training_callback is not None:
                    training_callback.on_val_loss_report(
                        dict(
                            iteration=iteration,
                            val_loss=val_loss,
                            val_time=val_time,
                            val_chosen_reward=val_rewards[0],
                            val_rejected_reward=val_rewards[1],
                            **{
                                f"val_{key}": value
                                for key, value in val_metrics.items()
                            },
                        )
                    )
                model.train()
                start += val_time
            accumulated_steps += 1
            update = (
                accumulated_steps == args.gradient_accumulation_steps
                or iteration == args.iters
            )
            loss, reward, count, batch_metrics, grad = step(
                next(batches), grad, update, accumulated_steps
            )
            if update:
                optimizer_step += 1
                accumulated_steps = 0
                if (
                    args.qat_enable
                    and not qat_installed
                    and optimizer_step >= args.qat_start_step
                ):
                    qat_originals = _install_policy_qat(model, args)
                    qat_installed = True
                    # QAT changes projection callables; retrace the compiled step.
                    step = mx.compile(
                        lambda *step_args: step_impl(*step_args),
                        inputs=state,
                        outputs=state,
                    )
            losses += loss
            rewards += reward
            tokens += count
            steps += 1
            for key, value in batch_metrics.items():
                metrics[key] = metrics.get(key, 0.0) + value
            mx.eval(state, losses, rewards, tokens, metrics, grad)
            if iteration % args.steps_per_report == 0 or iteration == args.iters:
                duration = time.perf_counter() - start
                train_loss = mx.distributed.all_sum(losses).item() / (steps * workers)
                train_rewards = (
                    mx.distributed.all_sum(rewards) / (steps * workers)
                ).tolist()
                train_metrics = {
                    key: (mx.distributed.all_sum(value) / (steps * workers)).item()
                    for key, value in metrics.items()
                }
                token_count = mx.distributed.all_sum(tokens).item()
                trained_tokens += token_count
                learning_rate = optimizer.learning_rate.item()
                peak_memory = mx.get_peak_memory() / 1e9
                if rank == 0:
                    tqdm.write(
                        f"Iter {iteration}: loss {train_loss:.3f}{format_latent_metrics(train_metrics)}, "
                        f"lr {learning_rate:.3e}, tok/s {token_count / duration:.3f}, "
                        f"peak_mem {peak_memory:.3f}GB"
                    )
                if training_callback is not None:
                    training_callback.on_train_loss_report(
                        dict(
                            iteration=iteration,
                            train_loss=train_loss,
                            train_chosen_reward=train_rewards[0],
                            train_rejected_reward=train_rewards[1],
                            learning_rate=learning_rate,
                            iterations_per_second=steps / duration,
                            tokens_per_second=token_count / duration,
                            trained_tokens=trained_tokens,
                            peak_memory=peak_memory,
                            **{
                                f"train_{key}": value
                                for key, value in train_metrics.items()
                            },
                        )
                    )
                losses, rewards, tokens, metrics, steps = (
                    0.0,
                    mx.zeros((2,)),
                    0.0,
                    {},
                    0,
                )
                start = time.perf_counter()
            if iteration % args.steps_per_save == 0 and rank == 0:
                save_adapters(model, args.adapter_file, iteration, report=False)
        if rank == 0:
            save_adapters(model, args.adapter_file)

    finally:
        for cls, original in qat_originals.items():
            cls.__call__ = original
        for cls, original in checkpoint_originals.items():
            cls.__call__ = original


def _install_policy_qat(model, args):
    """Apply shared STE quantization to policy projections, keeping the reference fixed."""
    projections = [
        module for _, module in model.named_modules() if isinstance(module, nn.Linear)
    ]
    policy_ids = {id(module) for module in projections}
    originals = {}
    for module in projections:
        cls = type(module)
        if cls in originals:
            continue
        original = cls.__call__
        originals[cls] = original

        def qat_forward(layer, inputs, _original=original):
            if id(layer) not in policy_ids:
                return _original(layer, inputs)
            weight = layer.weight
            layer.weight = weight + mx.stop_gradient(
                _symmetric_fake_quantize_tensor(
                    weight, args.qat_bits, args.qat_group_size
                )
                - weight
            )
            try:
                return _original(layer, inputs)
            finally:
                layer.weight = weight

        cls.__call__ = qat_forward
    return originals
