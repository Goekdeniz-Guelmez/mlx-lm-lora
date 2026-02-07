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
from mlx_lm.models.cache import KVCache, make_prompt_cache

from .datasets import CacheDataset


def reset_prompt_cache(cache):
    for e, c in enumerate(cache):
        if isinstance(c, KVCache):
            cache[e] = KVCache()
        else:
            raise ValueError("Unsupported cache")


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
        metadata={"help": "The examples are processsed sequentially in seq_step_size chunks."},
    )


def default_loss(model, batch, lengths, cache=None):
    inputs = batch[:, :-1]
    targets = batch[:, 1:]

    offset = cache[0].offset if cache is not None else 0
    logits = model(inputs, cache=cache)

    steps = mx.arange(1, targets.shape[1] + 1) + offset
    mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])

    loss = nn.losses.cross_entropy(logits, targets) * mask
    ntoks = mask.sum()
    loss = loss.astype(mx.float32).sum() / ntoks
    return loss, ntoks


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
):
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
            for s in range(0, seq_length, seq_step_size):
                end = min(s + seq_step_size, seq_length)
                # If next chunk would have only 1 token, absorb it into this chunk
                if 0 < (seq_length - end) < 2:
                    end = seq_length
                local_batch = (batch[0][:, s : end], batch[1])
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

    return (all_losses / ntokens).item()


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
        
        for s in range(0, seq_length, seq_step_size):
            end = min(s + seq_step_size, seq_length)
            # If next chunk would have only 1 token, absorb it into this chunk
            if 0 < (seq_length - end) < 2:
                end = seq_length
            local_batch = (batch[0][:, s : end], batch[1])
            (lvalue, toks), grad = loss_value_and_grad(model, *local_batch, cache)
            prev_n_tokens = n_tokens
            losses += toks * lvalue
            n_tokens += toks

            if seq_grad_accum is None:
                seq_grad_accum = grad
            else:
                scale_g = toks / n_tokens
                scale_acc = prev_n_tokens / n_tokens
                seq_grad_accum = tree_map(
                    lambda g, acc: scale_g * g + scale_acc * acc, grad, seq_grad_accum
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
            val_loss = evaluate_sft(
                model=model,
                dataset=val_dataset,
                loss=loss,
                batch_size=args.batch_size,
                num_batches=args.val_batches,
                max_seq_length=args.max_seq_length,
                iterate_batches=iterate_batches,
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
