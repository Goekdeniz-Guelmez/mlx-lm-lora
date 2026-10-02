"""Local memory workaround for the MLX-LM SSM training fallback.

``mlx_lm.models.ssm.ssm_update`` uses ``ssm_attn`` for multi-token inputs.
That implementation forms an O(block_size**2) per-head temporary.  Reducing
the block size trades some throughput for a substantially lower peak during
the forward and backward passes, without modifying the installed ``mlx_lm``.
"""

from __future__ import annotations

from typing import Callable


def set_ssm_attention_chunk_size(chunk_size: int) -> None:
    """Override the SSM fallback block size in the current Python process.

    This must run before the first model forward pass.  It patches the
    ``ssm_attn`` global used by the already-imported ``ssm_update`` function,
    so it also covers model modules that imported ``ssm_update`` directly.
    """
    if chunk_size < 1:
        raise ValueError("ssm_attention_chunk_size must be a positive integer")

    from mlx_lm.models import ssm

    original: Callable = getattr(ssm, "_mlx_lm_lora_original_ssm_attn", ssm.ssm_attn)
    ssm._mlx_lm_lora_original_ssm_attn = original

    def chunked_ssm_attn(*args, **kwargs):
        # Deliberately override a caller-supplied upstream default as well:
        # this is an explicit memory-safety option.
        kwargs["step"] = chunk_size
        return original(*args, **kwargs)

    ssm.ssm_attn = chunked_ssm_attn
