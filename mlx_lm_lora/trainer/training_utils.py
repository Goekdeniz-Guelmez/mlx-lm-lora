"""Small shared operations used by the algorithm-specific training loops."""

from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten
from tqdm import tqdm


def save_adapters(model, adapter_file, iteration=None, *, report=True):
    """Save trainable weights and, for periodic saves, a numbered checkpoint.

    Callers retain control over save frequency and distributed rank. Flatten
    the weights once so the current and numbered files contain the same state.
    """
    weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(adapter_file), weights)
    if iteration is not None:
        checkpoint = Path(adapter_file).parent / f"{iteration:07d}_adapters.safetensors"
        mx.save_safetensors(str(checkpoint), weights)
        if report:
            tqdm.write(
                f"Iter {iteration}: Saved adapter weights to "
                f"{adapter_file} and {checkpoint}."
            )
    elif report:
        tqdm.write(f"Saved final weights to {adapter_file}.")
