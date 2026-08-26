"""Shared helpers for cached long-context training."""


def iter_cached_sft_chunks(seq_length: int, seq_step_size: int):
    """Yield cached next-token-loss bounds with one token of left context.

    A loss over ``batch[:, start:end]`` predicts targets in
    ``[start + 1, end)``.  Continuation chunks therefore include the previous
    target token as context, making every next-token target appear exactly
    once while preserving the cached autoregressive state.
    """
    for chunk_start in range(0, seq_length, seq_step_size):
        end = min(chunk_start + seq_step_size, seq_length)
        if 0 < (seq_length - end) < 2:
            end = seq_length
        yield max(0, chunk_start - 1), end
        if end >= seq_length:
            break
