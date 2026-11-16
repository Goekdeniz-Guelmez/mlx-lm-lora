"""Resumable training checkpoints shared by every trainer.

Each trainer only needs three calls:

* ``progress = resume_training(args, model, optimizer)`` before the training
  step is compiled; start the loop at ``progress.start_iteration`` and seed the
  loop counters from ``progress``.
* ``save_checkpoint(args, it, model=..., optimizer=..., ...)`` whenever
  ``it % args.steps_per_save == 0``.
* ``save_checkpoint(args, last_it, ..., final=True)`` after the loop.

A checkpoint directory holds the trainable parameters, the optimizer state
(including the learning-rate schedule step), the MLX and NumPy RNG states,
partially accumulated gradients and the loop counters.  ``state.json`` is
written last, so a directory without it is incomplete and is never loaded.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_map_with_path, tree_unflatten
from tqdm import tqdm

CHECKPOINT_VERSION = 1
STATE_FILE = "state.json"
FINAL_CHECKPOINT_NAME = "final_training_checkpoint"


@dataclass
class TrainingProgress:
    """Loop counters restored from a checkpoint (defaults for a fresh run)."""

    iteration: int = 0
    optimizer_step: int = 0
    trained_tokens: int = 0
    grad_accum: Any = None

    @property
    def start_iteration(self) -> int:
        return self.iteration + 1


def checkpoint_dir(adapter_file, iteration: int) -> Path:
    return Path(adapter_file).parent / f"{iteration:07d}_training_checkpoint"


def _write_safetensors_atomic(path: Path, arrays: dict) -> None:
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    mx.save_safetensors(str(temporary), arrays)
    os.replace(temporary, path)


def _write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _mlx_rng_key() -> mx.array:
    return mx.random.state[0]


def _set_mlx_rng_key(key: mx.array) -> None:
    # The global MLX key is two uint32 words; seeding with their 64-bit value
    # reproduces it exactly and works whether or not the state is assignable.
    high, low = (int(word) for word in key.tolist())
    mx.random.seed((high << 32) | low)


def _numpy_state_to_json() -> dict:
    algorithm, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    return {
        "algorithm": algorithm,
        "keys": keys.tolist(),
        "position": int(position),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached_gaussian),
    }


def _numpy_state_from_json(state: dict) -> None:
    np.random.set_state(
        (
            state["algorithm"],
            np.asarray(state["keys"], dtype=np.uint32),
            state["position"],
            state["has_gauss"],
            state["cached_gaussian"],
        )
    )


def save_training_state(
    path,
    *,
    model,
    optimizer,
    iteration: int,
    optimizer_step: int = 0,
    grad_accum=None,
    trained_tokens: int = 0,
) -> Path:
    """Write the full training state to ``path`` (a directory)."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    # Invalidate an older checkpoint before replacing any of its tensor files.
    (path / STATE_FILE).unlink(missing_ok=True)
    mx.eval(model.trainable_parameters(), optimizer.state, mx.random.state, grad_accum)

    _write_safetensors_atomic(
        path / "model.safetensors", dict(tree_flatten(model.trainable_parameters()))
    )
    _write_safetensors_atomic(
        path / "optimizer.safetensors", dict(tree_flatten(optimizer.state))
    )
    _write_safetensors_atomic(path / "random.safetensors", {"key": _mlx_rng_key()})
    if grad_accum is not None:
        _write_safetensors_atomic(
            path / "grad_accum.safetensors", dict(tree_flatten(grad_accum))
        )
    else:
        (path / "grad_accum.safetensors").unlink(missing_ok=True)

    _write_json_atomic(
        path / STATE_FILE,
        {
            "version": CHECKPOINT_VERSION,
            "iteration": int(iteration),
            "optimizer_step": int(optimizer_step),
            "trained_tokens": int(trained_tokens),
            "has_grad_accum": grad_accum is not None,
            "numpy_random_state": _numpy_state_to_json(),
        },
    )
    return path


def load_training_state(path, *, model, optimizer) -> TrainingProgress:
    """Restore a directory written by :func:`save_training_state`."""
    path = Path(path)
    state_file = path / STATE_FILE
    if not state_file.exists():
        raise FileNotFoundError(
            f"{path} is not a complete training checkpoint (missing {STATE_FILE})."
        )
    metadata = json.loads(state_file.read_text(encoding="utf-8"))
    if metadata.get("version") != CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint version {metadata.get('version')!r}; "
            f"expected {CHECKPOINT_VERSION}."
        )

    model.load_weights(str(path / "model.safetensors"), strict=False)
    # The setter marks the optimizer uninitialised; ``init`` later only fills
    # in entries that are missing, so the restored moments are kept.
    optimizer.state = tree_unflatten(
        list(mx.load(str(path / "optimizer.safetensors")).items())
    )

    _set_mlx_rng_key(mx.load(str(path / "random.safetensors"))["key"])
    _numpy_state_from_json(metadata["numpy_random_state"])

    grad_accum = None
    if metadata["has_grad_accum"]:
        # Rebuild on the parameter tree so empty sub-trees (dropped when
        # flattening) line up with freshly computed gradients.
        flat = mx.load(str(path / "grad_accum.safetensors"))
        grad_accum = tree_map_with_path(
            lambda name, _: flat[name], model.trainable_parameters()
        )
    mx.eval(model.trainable_parameters(), optimizer.state, mx.random.state, grad_accum)
    return TrainingProgress(
        iteration=metadata["iteration"],
        optimizer_step=metadata["optimizer_step"],
        trained_tokens=metadata["trained_tokens"],
        grad_accum=grad_accum,
    )


def resume_training(args, model, optimizer) -> TrainingProgress:
    """Restore ``args.resume_checkpoint`` if set; call before compiling steps."""
    path = getattr(args, "resume_checkpoint", None)
    if path is None:
        return TrainingProgress()
    progress = load_training_state(path, model=model, optimizer=optimizer)
    tqdm.write(f"Resumed training state from {path} at iteration {progress.iteration}.")
    if progress.start_iteration > args.iters:
        tqdm.write(
            f"Checkpoint already reached iteration {progress.iteration}; "
            f"increase --iters (currently {args.iters}) to keep training."
        )
    return progress


def save_checkpoint(
    args,
    iteration: int,
    *,
    model,
    optimizer,
    optimizer_step: int = 0,
    grad_accum=None,
    trained_tokens: int = 0,
    final: bool = False,
) -> None:
    """Save trained weights and the resumable state (rank 0 only).

    Writes ``args.adapter_file`` plus, for periodic saves, a numbered weight
    snapshot and ``NNNNNNN_training_checkpoint/``; for the final save,
    ``final_training_checkpoint/``.
    """
    if mx.distributed.init().rank() != 0:
        return
    adapter_file = Path(args.adapter_file)
    weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(adapter_file), weights)

    if final:
        state_dir = adapter_file.parent / FINAL_CHECKPOINT_NAME
        written = [str(adapter_file)]
    else:
        snapshot = adapter_file.parent / f"{iteration:07d}_adapters.safetensors"
        mx.save_safetensors(str(snapshot), weights)
        state_dir = checkpoint_dir(adapter_file, iteration)
        written = [str(adapter_file), str(snapshot)]

    if getattr(args, "save_training_state", True):
        save_training_state(
            state_dir,
            model=model,
            optimizer=optimizer,
            iteration=iteration,
            optimizer_step=optimizer_step,
            grad_accum=grad_accum,
            trained_tokens=trained_tokens,
        )
        written.append(str(state_dir))

    label = "final weights" if final else f"Iter {iteration}: weights"
    tqdm.write(f"\nSaved {label} to {', '.join(written)}.")
