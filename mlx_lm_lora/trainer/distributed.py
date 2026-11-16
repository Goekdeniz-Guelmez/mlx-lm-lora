"""Small helpers for data-parallel training with ``mlx.launch``.

Every trainer follows the same recipe:

* every rank seeds NumPy and MLX with the same seed, so model init and the
  batch order are identical everywhere;
* :func:`shard` picks this rank's rows from a global batch, so a global batch of
  ``batch_size`` rows is split into ``batch_size // world_size`` rows per rank;
* gradients go through ``mlx.nn.utils.average_gradients``;
* :func:`reduce_mean` / :func:`reduce_metrics` turn per-rank running sums into
  global averages for logging, and only rank 0 prints.

With a single process every helper is a no-op, so the trainers need no
special cases.
"""

import os
import sys
from typing import Dict, Sequence

import mlx.core as mx

BACKENDS = ("any", "ring", "jaccl", "mpi")


def world() -> mx.distributed.Group:
    return mx.distributed.init()


def init_distributed(backend: str = "any") -> mx.distributed.Group:
    """Create the global group once, before anything else touches it.

    Outside ``mlx.launch`` this returns a size-1 group. Later calls to
    ``mx.distributed.init()`` return the same group.
    """
    if backend not in BACKENDS:
        raise ValueError(f"Unknown distributed backend {backend!r}; use {BACKENDS}")
    launched = any(
        key in os.environ
        for key in ("MLX_RANK", "OMPI_COMM_WORLD_RANK", "PMI_RANK", "MLX_HOSTFILE")
    )
    return mx.distributed.init(strict=launched, backend=backend)


def is_main_process() -> bool:
    return world().rank() == 0


def silence_worker_stdout() -> None:
    """Send stdout of ranks other than 0 to /dev/null; stderr stays visible."""
    if not is_main_process():
        sys.stdout = open(os.devnull, "w")


def local_batch_size(batch_size: int, size: int = None) -> int:
    size = world().size() if size is None else size
    if batch_size % size != 0:
        raise ValueError(
            f"The batch size ({batch_size}) must be divisible by the number of "
            f"workers ({size})"
        )
    return batch_size // size


def shard(batch: Sequence, rank: int = None, size: int = None) -> list:
    """Return this rank's rows of a global batch (strided, like mlx-lm)."""
    group = world()
    rank = group.rank() if rank is None else rank
    size = group.size() if size is None else size
    local_batch_size(len(batch), size)
    return list(batch[rank::size])


def all_sum(x):
    if world().size() == 1:
        return x if isinstance(x, mx.array) else mx.array(x)
    return mx.distributed.all_sum(x if isinstance(x, mx.array) else mx.array(x))


def reduce_mean(local_sum, steps: int) -> mx.array:
    """Global mean of per-step values that each rank summed over ``steps``."""
    return all_sum(local_sum) / (steps * world().size())


def reduce_metrics(metric_sums: Dict, steps: int) -> Dict[str, float]:
    """Apply :func:`reduce_mean` to a dict of running metric sums."""
    return {k: reduce_mean(v, steps).item() for k, v in metric_sums.items()}


@mx.custom_function
def differentiable_all_sum(x: mx.array) -> mx.array:
    """``all_sum`` whose gradient is also summed over ranks.

    MLX passes the cotangent of ``mx.distributed.all_sum`` through unchanged,
    which drops the other ranks' contributions when a loss uses statistics of
    the global batch.
    """
    return mx.distributed.all_sum(x)


@differentiable_all_sum.vjp
def _differentiable_all_sum_vjp(primals, cotangent, output):
    return mx.distributed.all_sum(cotangent)


def global_batch_mean(x: mx.array) -> mx.array:
    """Mean over axis 0 across the rows of every rank, with exact gradients."""
    if world().size() == 1:
        return x.mean(0)
    count = mx.distributed.all_sum(mx.array(x.shape[0], dtype=mx.float32))
    return differentiable_all_sum(x.sum(0)) / count


def global_mean_std(x: mx.array, eps: float = 1e-8):
    """Mean and std of ``x`` over the rows of every rank.

    Used where a loss normalises with batch statistics, so a 2-process run uses
    the same statistics as one process with the whole batch. Gradients flow
    to every rank's rows, like they do through ``mx.mean`` / ``mx.var``.
    """
    if world().size() == 1:
        return mx.mean(x), mx.sqrt(mx.var(x) + eps)
    mean = global_batch_mean(x.reshape(-1))
    count = mx.distributed.all_sum(mx.array(x.size, dtype=mx.float32))
    var = differentiable_all_sum(mx.square(x - mean).sum()) / count
    return mean, mx.sqrt(var + eps)
