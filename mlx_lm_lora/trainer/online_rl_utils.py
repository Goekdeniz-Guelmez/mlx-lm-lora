"""Shared helpers for memory-efficient online RL training."""

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache


def bucket_sequence_length(length: int, alignment: int = 64) -> int:
    """Round long sequence widths to reusable shapes without excess padding."""
    if length < 128:
        return length
    rounded = ((length + alignment - 1) // alignment) * alignment
    padding = rounded - length
    return rounded if padding <= max(16, length // 8) else length


def build_shared_prompt_cache(model, prompt_tokens):
    """Prefill all but the final prompt token for reuse by sibling rollouts."""
    if len(prompt_tokens) < 8:
        return None
    cache = make_prompt_cache(model)
    prefix = mx.array([prompt_tokens[:-1]], dtype=mx.uint32)
    model(prefix, cache=cache)
    mx.eval([entry.state for entry in cache])
    return cache


def fork_prompt_cache(model, template):
    """Create a new cache object whose immutable prefix arrays are shared."""
    if template is None:
        return None
    cache = make_prompt_cache(model)
    if len(cache) != len(template):
        return None
    try:
        for target, source in zip(cache, template):
            target.state = source.state
            meta_state = getattr(source, "meta_state", None)
            if meta_state is not None:
                target.meta_state = meta_state
    except (AttributeError, TypeError, ValueError):
        return None
    return cache
