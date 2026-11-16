import time
from dataclasses import dataclass, field
from functools import partial, wraps
from pathlib import Path
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.nn.utils import average_gradients
from mlx.utils import tree_flatten, tree_map
from mlx_lm.models.cache import (
    ArraysCache,
    CacheList,
    KVCache,
    RotatingKVCache,
    make_prompt_cache,
)
from mlx_lm.tuner.callbacks import TrainingCallback
from tqdm import tqdm

from ..recurrent_patch import enable_memory_safe_recurrences, model_uses_recurrence
from .datasets import CacheDataset
from .long_context import iter_cached_sft_chunks

_CHUNKED_NLL_CHUNK_SIZE = 256


def reset_prompt_cache(cache):
    if cache is None:
        return None

    reset_fn = getattr(cache, "reset", None)
    if callable(reset_fn):
        reset_fn()
        return cache

    if isinstance(cache, KVCache):
        return type(cache)()

    if isinstance(cache, RotatingKVCache):
        return type(cache)()

    if isinstance(cache, ArraysCache):
        cache.state = [None] * len(cache.state)
        finalize_fn = getattr(cache, "finalize", None)
        if callable(finalize_fn):
            finalize_fn()
        return cache

    if isinstance(cache, CacheList):
        cache.caches = tuple(reset_prompt_cache(c) for c in cache.caches)
        return cache

    if isinstance(cache, list):
        for e, c in enumerate(cache):
            cache[e] = reset_prompt_cache(c)
        return cache

    raise ValueError(f"Unsupported cache type: {type(cache)!r}")


def _find_cache_offset(cache):
    if cache is None:
        return None

    offset = getattr(cache, "offset", None)
    if offset is not None:
        return offset

    if isinstance(cache, (CacheList, list, tuple)):
        for item in cache:
            nested_offset = _find_cache_offset(item)
            if nested_offset is not None:
                return nested_offset

    return None


def grad_checkpoint(layer):
    """
    Update all instances of type(layer) to use gradient checkpointing.
    """
    fn = type(layer).__call__

    def checkpointed_fn(model, *args, **kwargs):
        def inner_fn(params, *args, **kwargs):
            model.update(params)
            return fn(model, *args, **kwargs)

        return mx.checkpoint(inner_fn)(model.trainable_parameters(), *args, **kwargs)

    type(layer).__call__ = checkpointed_fn


@dataclass
class SFTTrainingArgs:
    loss_type: str = field(
        default="nll",
        metadata={"help": "SFT loss type: 'nll', 'chunked_nll', or 'dft'."},
    )
    batch_size: int = field(default=4, metadata={"help": "Minibatch size."})
    iters: int = field(default=100, metadata={"help": "Iterations to train for."})
    gradient_accumulation_steps: int = field(
        default=1, metadata={"help": "Number of gradient accumulation steps."}
    )
    val_batches: int = field(
        default=25,
        metadata={
            "help": "Number of validation batches, -1 uses the entire validation set."
        },
    )
    steps_per_report: int = field(
        default=10,
        metadata={"help": "Number of training steps between loss reporting."},
    )
    steps_per_eval: int = field(
        default=200, metadata={"help": "Number of training steps between validations."}
    )
    steps_per_save: int = field(
        default=100, metadata={"help": "Save the model every number steps"}
    )
    max_seq_length: int = field(
        default=2048, metadata={"help": "Maximum sequence length."}
    )
    adapter_file: str = field(
        default="adapters.safetensors",
        metadata={"help": "Save/load path for the trained adapter weights."},
    )
    grad_checkpoint: bool = field(
        default=False,
        metadata={"help": "Use gradient checkpointing to reduce memory use."},
    )
    seq_step_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "The examples are processsed sequentially in seq_step_size chunks."
        },
    )
    recurrence_chunk_size: int = field(
        default=64,
        metadata={
            "help": "Chunk size used by memory-safe recurrent training fallbacks."
        },
    )
    qat_enable: bool = field(
        default=False,
        metadata={
            "help": "Enable minimal QAT-style projection on trainable weights after updates."
        },
    )
    qat_bits: int = field(
        default=8,
        metadata={"help": "Bit-width used for QAT projection."},
    )
    qat_group_size: int = field(
        default=64,
        metadata={
            "help": "Group size for QAT projection (0 or less means per-tensor)."
        },
    )
    qat_mode: str = field(
        default="affine",
        metadata={
            "help": "QAT mode. Only 'affine' (the mx.quantize grid used by "
            "mlx_lm convert -q) is supported."
        },
    )
    qat_start_step: int = field(
        default=1,
        metadata={"help": "Apply QAT projection starting from this optimizer step."},
    )
    qat_interval: int = field(
        default=1,
        metadata={"help": "Apply QAT projection every N optimizer steps."},
    )


_MX_QUANTIZE_BITS = {2, 3, 4, 5, 6, 8}
_MX_QUANTIZE_GROUP_SIZES = {32, 64, 128}


def _affine_fake_quantize_groups(x, bits: int):
    """Affine quantize-dequantize along the last axis, mirroring ``mx.quantize``.

    Used only for configurations ``mx.quantize`` does not support (per-tensor
    groups, odd group sizes or bit-widths).
    """
    dtype = x.dtype
    x = x.astype(mx.float32)
    n_bins = (1 << bits) - 1
    w_max = mx.max(x, axis=-1, keepdims=True)
    w_min = mx.min(x, axis=-1, keepdims=True)
    mask = mx.abs(w_min) > mx.abs(w_max)
    scale = mx.maximum((w_max - w_min) / n_bins, 1e-7)
    scale = mx.where(mask, scale, -scale)
    edge = mx.where(mask, w_min, w_max)
    q0 = mx.round(edge / scale)
    scale = mx.where(q0 != 0, edge / mx.where(q0 != 0, q0, 1), scale)
    bias = mx.where(q0 == 0, 0, edge)
    q = mx.clip(mx.round((x - bias) / scale), 0, n_bins)
    return (q * scale + bias).astype(dtype)


def _affine_fake_quantize_tensor(x, bits: int, group_size: int):
    """Round ``x`` onto the grid ``mlx_lm convert -q`` stores (scale + bias per group).

    Uses ``mx.quantize``/``mx.dequantize`` directly whenever the configuration
    is supported, so the simulated weights equal the deployed weights exactly.
    """
    if x.ndim == 0:
        return x
    last_dim = x.shape[-1]
    if (
        x.ndim >= 2
        and bits in _MX_QUANTIZE_BITS
        and group_size in _MX_QUANTIZE_GROUP_SIZES
        and last_dim % group_size == 0
    ):
        wq, scales, biases = mx.quantize(x, group_size=group_size, bits=bits)
        return mx.dequantize(
            wq, scales, biases, group_size=group_size, bits=bits
        ).astype(x.dtype)

    if group_size is None or group_size <= 0:
        return _affine_fake_quantize_groups(x.reshape(1, -1), bits).reshape(x.shape)
    if group_size >= last_dim:
        return _affine_fake_quantize_groups(x, bits)

    num_groups = (last_dim + group_size - 1) // group_size
    pad = num_groups * group_size - last_dim
    if pad > 0:
        # Pad with the row's first value so padding never widens a group's range.
        pad_vals = mx.broadcast_to(x[..., :1], x.shape[:-1] + (pad,))
        x = mx.concatenate([x, pad_vals], axis=-1)
    groups = x.reshape(x.shape[:-1] + (num_groups, group_size))
    x_q = _affine_fake_quantize_groups(groups, bits).reshape(x.shape)
    if pad > 0:
        x_q = x_q[..., :last_dim]
    return x_q


# Backwards-compatible alias; the quantizer is affine, matching ``--qat-mode affine``.
_symmetric_fake_quantize_tensor = _affine_fake_quantize_tensor


def _ste_fake_quantize(w, bits: int, group_size: int):
    """Straight-through estimator: forward sees rounded ``w``, backward is identity."""
    return w + mx.stop_gradient(_affine_fake_quantize_tensor(w, bits, group_size) - w)


def _dequantized(layer):
    weight = layer.weight
    if isinstance(layer, (nn.QuantizedLinear, nn.QuantizedEmbedding)):
        weight = mx.dequantize(
            weight,
            layer.scales,
            layer.biases,
            group_size=layer.group_size,
            bits=layer.bits,
            mode=layer.mode,
        )
    return weight


def _adapter_kind(module):
    """Return ``"linear"``/``"embedding"`` for LoRA/DoRA adapters, else ``None``."""
    if not all(hasattr(module, name) for name in ("lora_a", "lora_b", "scale")):
        return None
    base = getattr(module, "linear", None)
    if isinstance(base, (nn.Linear, nn.QuantizedLinear)):
        return "linear"
    base = getattr(module, "embedding", None)
    if isinstance(base, (nn.Embedding, nn.QuantizedEmbedding)):
        return "embedding"
    return None


def _adapter_merged_weight(module, kind=None):
    """The weight ``mlx_lm fuse`` would store for a LoRA or DoRA adapter."""
    kind = kind or _adapter_kind(module)
    if kind == "linear":
        weight = _dequantized(module.linear)
        delta = (module.scale * module.lora_b.T) @ module.lora_a.T
    else:
        weight = _dequantized(module.embedding)
        delta = (module.scale * module.lora_a) @ module.lora_b
    weight = weight + delta.astype(weight.dtype)
    if hasattr(module, "m"):  # DoRA: rescale rows to the learned magnitude
        norm_scale = module.m / mx.linalg.norm(weight, axis=1)
        weight = norm_scale[:, None].astype(weight.dtype) * weight
    return weight


# Kept for callers that only care about LoRA linear layers.
def _lora_merged_weight(module):
    return _adapter_merged_weight(module, "linear")


_ACTIVE_QAT: list = []


def _install_qat(model, bits: int, group_size: int):
    """Install STE fake-quantization on ``model``'s instances; return class originals.

    Every layer is rounded the way ``mlx_lm fuse`` + ``convert -q`` will store it:

    * ``nn.Linear`` / ``nn.Embedding``: the weight is rounded.
    * LoRA and DoRA adapters (linear and embedding): the *merged* weight
      (``base + scale * delta``, rescaled to ``m`` for DoRA) is rounded.
      Adapter dropout is skipped while QAT is active.

    Only modules of ``model`` are affected, so a reference model sharing the
    same classes keeps its full-precision forward.  Restore with
    ``_restore_qat(originals)``.
    """
    if bits < 2 or bits > 16:
        raise ValueError("qat_bits must be in [2, 16]")

    adapter_kinds, linear_ids, embed_ids = {}, set(), set()
    for _, module in model.named_modules():
        kind = _adapter_kind(module)
        if kind is not None:
            adapter_kinds[id(module)] = kind
        elif isinstance(module, nn.Linear):
            linear_ids.add(id(module))
        elif isinstance(module, nn.Embedding):
            embed_ids.add(id(module))

    originals = {}

    def _patch(cls, attr, make):
        key = (cls, attr)
        if key in originals or not hasattr(cls, attr):
            return
        originals[key] = getattr(cls, attr)
        setattr(cls, attr, make(originals[key]))

    def _rounded_adapter(self, kind):
        weight = _adapter_merged_weight(self, kind)
        return _ste_fake_quantize(weight, bits, group_size)

    def _make_adapter_call(orig):
        def fwd(self, x):
            kind = adapter_kinds.get(id(self))
            if kind is None:
                return orig(self, x)
            w = _rounded_adapter(self, kind)
            if kind == "embedding":
                return w[x]
            y = x @ w.astype(x.dtype).T
            if "bias" in self.linear:
                y = y + self.linear.bias
            return y

        return fwd

    def _make_adapter_as_linear(orig):
        def fwd(self, x):
            if adapter_kinds.get(id(self)) != "embedding":
                return orig(self, x)
            return x @ _rounded_adapter(self, "embedding").astype(x.dtype).T

        return fwd

    def _make_weight_swap(ids):
        def make(orig):
            def fwd(self, *a, **kw):
                if id(self) not in ids:
                    return orig(self, *a, **kw)
                w = self.weight
                self.weight = _ste_fake_quantize(w, bits, group_size)
                try:
                    return orig(self, *a, **kw)
                finally:
                    self.weight = w  # optimizer keeps full-precision weights

            return fwd

        return make

    for _, module in model.named_modules():
        mid, cls = id(module), type(module)
        if mid in adapter_kinds:
            _patch(cls, "__call__", _make_adapter_call)
            if adapter_kinds[mid] == "embedding":
                _patch(cls, "as_linear", _make_adapter_as_linear)
        elif mid in linear_ids:
            _patch(cls, "__call__", _make_weight_swap(linear_ids))
        elif mid in embed_ids:
            _patch(cls, "__call__", _make_weight_swap(embed_ids))
            _patch(cls, "as_linear", _make_weight_swap(embed_ids))
    return originals


def _restore_qat(originals):
    for (cls, attr), original in originals.items():
        setattr(cls, attr, original)


def _install_qat_hooks(model, args: SFTTrainingArgs):
    """Enable QAT on ``model`` (see ``_install_qat``). Must be called once."""
    if not args.qat_enable:
        return {}
    originals = _install_qat(model, args.qat_bits, args.qat_group_size)
    _ACTIVE_QAT.append(originals)
    return originals


def restores_qat_hooks(train_fn):
    """Undo any ``_install_qat_hooks`` done during ``train_fn``, even on error."""

    @wraps(train_fn)
    def wrapper(*args, **kwargs):
        depth = len(_ACTIVE_QAT)
        try:
            return train_fn(*args, **kwargs)
        finally:
            while len(_ACTIVE_QAT) > depth:
                _restore_qat(_ACTIVE_QAT.pop())

    return wrapper


def default_loss(model, batch, lengths, cache=None):
    inputs = batch[:, :-1]
    targets = batch[:, 1:]

    offset = _find_cache_offset(cache)
    offset = 0 if offset is None else offset
    logits = model(inputs, cache=cache)

    steps = mx.arange(1, targets.shape[1] + 1) + offset
    mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])

    loss = nn.losses.cross_entropy(logits, targets) * mask
    ntoks = mask.sum()
    loss = loss.astype(mx.float32).sum() / ntoks
    return loss, ntoks


def chunked_nll_loss(model, batch, lengths, cache=None):
    """Compute masked next-token NLL in bounded sequence chunks."""
    inputs = batch[:, :-1]
    targets = batch[:, 1:]

    offset = _find_cache_offset(cache)
    offset = 0 if offset is None else offset
    logits = model(inputs, cache=cache)

    steps = mx.arange(1, targets.shape[1] + 1) + offset
    mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])
    ntoks = mask.sum()

    loss_sum = mx.array(0.0, dtype=mx.float32)
    for start in range(0, targets.shape[1], _CHUNKED_NLL_CHUNK_SIZE):
        end = min(start + _CHUNKED_NLL_CHUNK_SIZE, targets.shape[1])
        chunk_loss = nn.losses.cross_entropy(
            logits[:, start:end], targets[:, start:end]
        )
        loss_sum += (chunk_loss * mask[:, start:end]).astype(mx.float32).sum()

    return loss_sum / ntoks, ntoks


def dft_loss(model, batch, lengths, cache=None):
    """Dynamic fine-tuning loss with detached probability weighting."""
    inputs = batch[:, :-1]
    targets = batch[:, 1:]

    offset = _find_cache_offset(cache)
    offset = 0 if offset is None else offset
    logits = model(inputs, cache=cache)

    steps = mx.arange(1, targets.shape[1] + 1) + offset
    mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])

    logprobs = mx.take_along_axis(
        nn.log_softmax(logits, axis=-1), targets[..., None], axis=-1
    ).squeeze(-1)
    weights = mx.stop_gradient(mx.exp(logprobs))
    ntoks = mask.sum()
    loss = (-(weights * logprobs) * mask).astype(mx.float32).sum() / ntoks
    return loss, ntoks


def get_sft_loss(loss_type: str):
    if loss_type == "nll":
        return default_loss
    if loss_type == "chunked_nll":
        return chunked_nll_loss
    if loss_type == "dft":
        return dft_loss
    raise ValueError(
        f"Unknown SFT loss type: {loss_type!r}. "
        "Expected 'nll', 'chunked_nll', or 'dft'."
    )


def iterate_batches(
    dataset,
    batch_size,
    max_seq_length,
    train=False,
):
    if isinstance(dataset, CacheDataset):
        len_fn = lambda idx: dataset.itemlen(idx)
    else:
        len_fn = lambda idx: len(dataset[idx])
    idx = sorted(range(len(dataset)), key=len_fn)
    if len(dataset) < batch_size:
        raise ValueError(
            f"Dataset must have at least batch_size={batch_size}"
            f" examples but only has {len(dataset)}."
        )

    step = mx.distributed.init().size()
    if batch_size % step != 0:
        raise ValueError("The batch size must be divisible by the number of workers")

    batch_idx = [
        idx[i : i + batch_size : step]
        for i in range(0, len(idx) - batch_size + 1, batch_size)
    ]

    while True:
        indices = np.random.permutation(len(batch_idx))
        for i in indices:
            batch = [dataset[j] for j in batch_idx[i]]
            if len(batch[0]) == 2:
                batch, offsets = zip(*batch)
            else:
                offsets = [0] * len(batch)
            lengths = [len(x) for x in batch]

            pad_to = 32
            max_length_in_batch = 1 + pad_to * ((max(lengths) + pad_to - 1) // pad_to)
            max_length_in_batch = min(max_length_in_batch, max_seq_length)

            batch_arr = np.zeros((batch_size // step, max_length_in_batch), np.int32)

            for j in range(batch_size // step):
                truncated_length = min(lengths[j], max_seq_length)
                batch_arr[j, :truncated_length] = batch[j][:truncated_length]
                lengths[j] = truncated_length
            batch = mx.array(batch_arr)
            yield batch, mx.array(list(zip(offsets, lengths)))

        if not train:
            break


def evaluate_sft(
    model,
    dataset,
    batch_size,
    num_batches,
    max_seq_length=2048,
    loss: callable = default_loss,
    iterate_batches: callable = iterate_batches,
    efficient: bool = False,
    seq_step_size: int = 512,
    recurrence_chunk_size: int = 64,
):
    if model_uses_recurrence(model):
        enable_memory_safe_recurrences(chunk_size=recurrence_chunk_size)

    model.eval()
    all_losses = mx.array(0.0)
    ntokens = mx.array(0)

    index_iterator = iter(range(num_batches)) if num_batches != -1 else iter(int, 1)
    seq_step_size = seq_step_size if efficient else max_seq_length

    cache = make_prompt_cache(model) if efficient else None
    for _, batch in zip(
        index_iterator,
        iterate_batches(
            dataset=dataset,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
        ),
    ):
        if efficient and cache is not None:
            seq_length = batch[0].shape[1]
            for start, end in iter_cached_sft_chunks(seq_length, seq_step_size):
                local_batch = (batch[0][:, start:end], batch[1])
                losses, toks = loss(model, *local_batch, cache)
                all_losses += losses * toks
                ntokens += toks
                if end >= seq_length:
                    reset_prompt_cache(cache)
                mx.eval(all_losses, ntokens)
                if end >= seq_length:
                    break
        else:
            losses, toks = loss(model, *batch)
            all_losses += losses * toks
            ntokens += toks
            mx.eval(all_losses, ntokens)

    all_losses = mx.distributed.all_sum(all_losses, stream=mx.cpu)
    ntokens = mx.distributed.all_sum(ntokens, stream=mx.cpu)

    # Return the token count alongside the loss: callers such as
    # mlx_lm_lora.train.evaluate_model report the number of evaluated
    # tokens, and previously unpacked (loss, ntok) from this function
    # while it only ever returned a float.
    return (all_losses / ntokens).item(), int(ntokens.item())


@restores_qat_hooks
def train_sft(
    model,
    optimizer,
    train_dataset,
    val_dataset: Optional[Any] = None,
    args: SFTTrainingArgs = SFTTrainingArgs(),
    loss: callable = default_loss,
    iterate_batches: callable = iterate_batches,
    training_callback: TrainingCallback = None,
):
    # Direct API users (including the SFT notebook) bypass train.py, so apply
    # the same automatic protection used by the command-line entry point.
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
    if loss is default_loss:
        loss = get_sft_loss(args.loss_type)

    qat_installed = False  # hooks installed lazily at qat_start_step
    efficient = True if args.seq_step_size is not None else False

    if efficient:
        cache = make_prompt_cache(model)

    state = [model.state, optimizer.state, mx.random.state]

    loss_value_and_grad = nn.value_and_grad(model, loss)

    @partial(mx.compile, inputs=state, outputs=state)
    def step(batch, prev_grad, do_update):
        # Regular training without sequence splitting
        (lvalue, toks), grad = loss_value_and_grad(model, *batch)

        # Handle gradient accumulation across steps
        if prev_grad is not None:
            grad = tree_map(lambda x, y: x + y, grad, prev_grad)

        if do_update:
            grad = average_gradients(grad)
            if grad_accum_steps > 1:
                grad = tree_map(lambda x: x / grad_accum_steps, grad)
            optimizer.update(model, grad)
            grad = None

        return lvalue, toks, grad

    # No compilation for seq_split_step since it uses cache mutation
    def seq_split_step(batch, prev_grad, do_update):
        # Sequence splitting logic for efficient training
        losses = mx.array(0.0)
        n_tokens = mx.array(0.0)
        seq_length = batch[0].shape[1]
        seq_grad_accum = None

        for start, end in iter_cached_sft_chunks(seq_length, seq_step_size):
            local_batch = (batch[0][:, start:end], batch[1])
            (lvalue, toks), grad = loss_value_and_grad(model, *local_batch, cache)
            prev_n_tokens = n_tokens
            losses += toks * lvalue
            n_tokens += toks

            if seq_grad_accum is None:
                seq_grad_accum = grad
            else:
                scale_grad = toks / n_tokens
                scale_accum = prev_n_tokens / n_tokens
                seq_grad_accum = tree_map(
                    lambda g, acc: scale_grad * g + scale_accum * acc,
                    grad,
                    seq_grad_accum,
                )

            # Reset prompt cache before the last eval
            if end >= seq_length:
                reset_prompt_cache(cache)

            # Evaluate intermediate results to ensure proper execution
            mx.eval(state, seq_grad_accum, losses, n_tokens)
            if end >= seq_length:
                break

        lvalue = losses / n_tokens
        toks = n_tokens
        grad = seq_grad_accum

        # Handle gradient accumulation across steps
        if prev_grad is not None:
            grad = tree_map(lambda x, y: x + y, grad, prev_grad)

        if do_update:
            grad = average_gradients(grad)
            if grad_accum_steps > 1:
                grad = tree_map(lambda x: x / grad_accum_steps, grad)
            optimizer.update(model, grad)
            grad = None

        return lvalue, toks, grad

    model.train()
    seq_step_size = args.seq_step_size or args.max_seq_length
    losses = 0
    n_tokens = 0
    steps = 0
    trained_tokens = 0
    train_time = 0
    grad_accum = None
    opt_step = 0

    # Main training loop
    pbar = tqdm(range(1, args.iters + 1), desc="Training", disable=rank != 0)
    for it in pbar:
        batch = next(
            iterate_batches(
                dataset=train_dataset,
                batch_size=args.batch_size,
                max_seq_length=args.max_seq_length,
                train=True,
            )
        )
        tic = time.perf_counter()
        if (
            val_dataset is not None
            and len(val_dataset) > 0
            and args.steps_per_eval is not None
            and (it == 1 or it % args.steps_per_eval == 0 or it == args.iters)
        ):
            tic = time.perf_counter()
            val_loss, _ = evaluate_sft(
                model=model,
                dataset=val_dataset,
                loss=loss,
                batch_size=args.batch_size,
                num_batches=args.val_batches,
                max_seq_length=args.max_seq_length,
                iterate_batches=iterate_batches,
                recurrence_chunk_size=args.recurrence_chunk_size,
                efficient=efficient,
                seq_step_size=seq_step_size,
            )
            model.train()
            val_time = time.perf_counter() - tic
            if rank == 0:
                tqdm.write(
                    f"Iter {it}: "
                    f"Val loss {val_loss:.3f}, "
                    f"Val took {val_time:.3f}s",
                )

            if training_callback is not None:
                val_info = {
                    "iteration": it,
                    "val_loss": val_loss,
                    "val_time": val_time,
                }
                training_callback.on_val_loss_report(val_info)

            tic = time.perf_counter()

        if efficient and batch[0].shape[1] > seq_step_size:
            lvalue, toks, grad_accum = seq_split_step(
                batch,
                grad_accum,
                it % grad_accum_steps == 0,
            )
        else:
            lvalue, toks, grad_accum = step(
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
        n_tokens += toks
        steps += 1
        mx.eval(state, losses, n_tokens, grad_accum)
        train_time += time.perf_counter() - tic

        if it % args.steps_per_report == 0 or it == args.iters:
            train_loss = mx.distributed.all_sum(losses, stream=mx.cpu).item()
            train_loss /= steps * world_size
            n_tokens = mx.distributed.all_sum(n_tokens, stream=mx.cpu).item()
            learning_rate = optimizer.learning_rate.item()
            it_sec = args.steps_per_report / train_time
            tokens_sec = float(n_tokens) / train_time
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
                    f"lr {learning_rate:.3e}, "
                    f"it/s {it_sec:.3f}, "
                    f"tok/s {tokens_sec:.3f}, "
                    f"trained_tok {trained_tokens}, "
                    f"peak_mem {peak_mem:.3f}GB"
                )

            if training_callback is not None:
                train_info = {
                    "iteration": it,
                    "train_loss": train_loss,
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
            train_time = 0

        if it % args.steps_per_save == 0 and rank == 0:
            adapter_weights = dict(tree_flatten(model.trainable_parameters()))
            mx.save_safetensors(str(args.adapter_file), adapter_weights)
            checkpoint = (
                Path(args.adapter_file).parent / f"{it:07d}_adapters.safetensors"
            )
            mx.save_safetensors(str(checkpoint), adapter_weights)
            tqdm.write(
                f"\n"
                f"Iter {it}: Saved adapter weights to "
                f"{args.adapter_file} and {checkpoint}."
            )

    if rank == 0:
        adapter_weights = dict(tree_flatten(model.trainable_parameters()))
        mx.save_safetensors(str(args.adapter_file), adapter_weights)
        tqdm.write(f"Saved final weights to {args.adapter_file}.")
