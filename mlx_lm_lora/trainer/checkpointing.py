"""Full, resumable MLX training checkpoints."""

import json
import os
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten


CHECKPOINT_VERSION = 1


def _write_safetensors_atomic(path: Path, arrays: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    mx.save_safetensors(str(temporary), arrays)
    os.replace(temporary, path)


def _write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def save_training_checkpoint(
    path,
    *,
    model,
    optimizer,
    iteration: int,
    optimizer_step: int,
    grad_accum=None,
    trained_tokens: int = 0,
) -> Path:
    """Atomically save all state needed to continue a training loop."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    # Invalidate an older checkpoint before replacing any of its tensor files.
    (path / "state.json").unlink(missing_ok=True)
    mx.eval(model.trainable_parameters(), optimizer.state, mx.random.state, grad_accum)

    _write_safetensors_atomic(
        path / "model.safetensors",
        dict(tree_flatten(model.trainable_parameters())),
    )
    _write_safetensors_atomic(
        path / "optimizer.safetensors", dict(tree_flatten(optimizer.state))
    )
    _write_safetensors_atomic(
        path / "random.safetensors",
        {f"mlx.{key}": value for key, value in tree_flatten(mx.random.state)},
    )
    if grad_accum is not None:
        _write_safetensors_atomic(
            path / "grad_accum.safetensors", dict(tree_flatten(grad_accum))
        )
    else:
        (path / "grad_accum.safetensors").unlink(missing_ok=True)

    np_state = np.random.get_state()
    metadata = {
        "version": CHECKPOINT_VERSION,
        "iteration": iteration,
        "optimizer_step": optimizer_step,
        "trained_tokens": int(trained_tokens),
        "has_grad_accum": grad_accum is not None,
        "numpy_random_state": {
            "algorithm": np_state[0],
            "keys": np_state[1].tolist(),
            "position": np_state[2],
            "has_gauss": np_state[3],
            "cached_gaussian": np_state[4],
        },
    }
    # Written last: its presence means every tensor file is complete.
    _write_json_atomic(path / "state.json", metadata)
    return path


def load_training_checkpoint(path, *, model, optimizer) -> dict:
    """Restore a checkpoint and return its loop counters and accumulated grads."""
    path = Path(path)
    metadata = json.loads((path / "state.json").read_text(encoding="utf-8"))
    if metadata.get("version") != CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint version {metadata.get('version')!r}; "
            f"expected {CHECKPOINT_VERSION}."
        )

    model.load_weights(str(path / "model.safetensors"), strict=False)
    optimizer.state = tree_unflatten(dict(mx.load(str(path / "optimizer.safetensors"))))
    optimizer._initialized = True

    random_values = {
        key[len("mlx.") :] if key.startswith("mlx.") else key: value
        for key, value in mx.load(str(path / "random.safetensors")).items()
    }
    restored_random_state = tree_unflatten(random_values)
    if isinstance(mx.random.state, list):
        mx.random.state[:] = restored_random_state
    else:
        mx.random.state = restored_random_state

    np_state = metadata["numpy_random_state"]
    np.random.set_state(
        (
            np_state["algorithm"],
            np.asarray(np_state["keys"], dtype=np.uint32),
            np_state["position"],
            np_state["has_gauss"],
            np_state["cached_gaussian"],
        )
    )
    grad_accum = None
    if metadata["has_grad_accum"]:
        grad_accum = tree_unflatten(
            dict(mx.load(str(path / "grad_accum.safetensors")))
        )
    mx.eval(model.trainable_parameters(), optimizer.state, mx.random.state, grad_accum)
    return {**metadata, "grad_accum": grad_accum}
