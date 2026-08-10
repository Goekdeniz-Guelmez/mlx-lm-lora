"""Opt-in memory-safe training fallbacks for recurrent MLX-LM layers.

The installed MLX-LM package is never changed.  The patch replaces the
training fallbacks of its shared recurrence primitives before model loading.
Each long recurrence is split into checkpointed blocks, so MLX recomputes
within a block during backward rather than retaining the whole sequence.
"""

from __future__ import annotations


_RECURRENT_MODEL_MODULES = frozenset(
    {
        "bailing_moe_linear",
        "falcon_h1",
        "granitemoehybrid",
        "jamba",
        "kimi_linear",
        "mamba",
        "mamba2",
        "nemotron_h",
        "plamo2",
        "qwen3_5",
        "qwen3_next",
    }
)


def model_uses_recurrence(model) -> bool:
    """Return whether a loaded model includes a supported recurrent layer."""
    modules = getattr(model, "named_modules", None)
    if not callable(modules):
        return False
    for _, module in modules():
        module_name = type(module).__module__.rsplit(".", maxsplit=1)[-1]
        if module_name in _RECURRENT_MODEL_MODULES:
            return True
    return False


def enable_memory_safe_recurrences(chunk_size: int = 64) -> None:
    """Install checkpointed training fallbacks for supported recurrent layers.

    Covers scalar and vector-gated delta (Qwen3.5/Next and Kimi Linear), and
    recurrent GLA (Bailing MoE Linear).  Shared SSM models already use a
    chunked fallback; their temporary block is reduced to the same size.
    """
    if chunk_size < 1:
        raise ValueError("recurrent chunk size must be a positive integer")

    import mlx.core as mx
    from mlx_lm.models import gated_delta

    if not getattr(gated_delta, "_mlx_lm_lora_memory_safe", False):
        step = gated_delta._gated_delta_step_ops

        def recurrence_chunk(q, k, v, g, beta, state, mask):
            outputs = []
            for t in range(q.shape[1]):
                output, state = step(
                    q[:, t],
                    k[:, t],
                    v[:, t],
                    g[:, t],
                    beta[:, t],
                    state,
                    None if mask is None else mask[:, t],
                )
                outputs.append(output)
            return mx.stack(outputs, axis=1), state

        checkpointed_recurrence_chunk = mx.checkpoint(recurrence_chunk)

        def checkpointed_gated_delta(q, k, v, g, beta, state=None, mask=None):
            """Exact recurrence with checkpointed blocks for scalar or vector g."""
            batch, length, key_heads, key_dim = q.shape
            value_heads, value_dim = v.shape[-2:]
            if state is None:
                state = mx.zeros(
                    (batch, value_heads, value_dim, key_dim), dtype=mx.float32
                )
            if value_heads % key_heads:
                raise ValueError("gated-delta value heads must be divisible by key heads")
            repeat_factor = value_heads // key_heads
            if repeat_factor > 1:
                q = mx.repeat(q, repeat_factor, axis=-2)
                k = mx.repeat(k, repeat_factor, axis=-2)

            outputs = []
            for start in range(0, length, chunk_size):
                end = min(start + chunk_size, length)
                output, state = checkpointed_recurrence_chunk(
                    q[:, start:end],
                    k[:, start:end],
                    v[:, start:end],
                    g[:, start:end],
                    beta[:, start:end],
                    state,
                    None if mask is None else mask[:, start:end],
                )
                outputs.append(output)
            return mx.concatenate(outputs, axis=1), state

        gated_delta.gated_delta_ops = checkpointed_gated_delta
        gated_delta._mlx_lm_lora_memory_safe = True

    try:
        from mlx_lm.models import bailing_moe_linear
    except ImportError:
        bailing_moe_linear = None

    if bailing_moe_linear is not None and not getattr(
        bailing_moe_linear, "_mlx_lm_lora_memory_safe", False
    ):
        def gla_chunk(q, k, v, state, decay):
            outputs = []
            for t in range(q.shape[2]):
                key = k[:, :, t : t + 1]
                value = v[:, :, t : t + 1]
                state = state * decay + key.transpose(0, 1, 3, 2) @ value
                outputs.append(q[:, :, t : t + 1] @ state)
            return mx.concatenate(outputs, axis=2), state

        checkpointed_gla_chunk = mx.checkpoint(gla_chunk)

        def checkpointed_recurrent_gla(q, k, v, g, scale, h=None):
            """Exact GLA recurrence with checkpointed training blocks."""
            batch, _, length, key_dim = q.shape
            value_heads, value_dim = v.shape[1], v.shape[-1]
            if h is None:
                h = mx.zeros((batch, value_heads, key_dim, value_dim), dtype=q.dtype)
            decay = mx.exp(g)[:, None, None].astype(q.dtype)
            q = q * scale
            outputs = []
            for start in range(0, length, chunk_size):
                end = min(start + chunk_size, length)
                output, h = checkpointed_gla_chunk(
                    q[:, :, start:end], k[:, :, start:end], v[:, :, start:end], h, decay
                )
                outputs.append(output)
            return mx.concatenate(outputs, axis=2), h

        bailing_moe_linear.recurrent_gla = checkpointed_recurrent_gla
        bailing_moe_linear._mlx_lm_lora_memory_safe = True

    # Older Mamba and Jamba implementations own their recurrent Python loops
    # instead of using mlx_lm.models.ssm.  Keep their math unchanged and put
    # checkpoint boundaries around short contiguous time spans.
    try:
        from mlx_lm.models import mamba
    except ImportError:
        mamba = None

    if mamba is not None and not getattr(mamba, "_mlx_lm_lora_memory_safe", False):
        def checkpointed_mamba_process(self, x, conv_cache, state_cache):
            _, length, _ = x.shape
            xz = self.in_proj(x)
            x, z = xz.split(indices_or_sections=2, axis=-1)
            kernel_size = self.conv_kernel_size
            if conv_cache is not None:
                x_full = mx.concatenate([conv_cache, x], axis=1)
            else:
                x_full = mx.pad(x, [(0, 0), (kernel_size - 1, 0), (0, 0)])
            conv_out = self.conv1d(x_full)
            new_conv_cache = x_full[:, -(kernel_size - 1) :, :]
            x = mamba.nn.silu(conv_out)
            A = -mx.exp(self.A_log)
            if state_cache is None:
                state_cache = mx.zeros(
                    (x.shape[0], self.intermediate_size, self.ssm_state_size),
                    dtype=x.dtype,
                )

            def mamba_chunk(x_chunk, state, A):
                outputs = []
                for t in range(x_chunk.shape[1]):
                    output, state = self.ssm_step(x_chunk[:, t], A, state)
                    outputs.append(output)
                return mx.stack(outputs, axis=1), state

            checkpointed_mamba_chunk = mx.checkpoint(mamba_chunk)
            outputs = []
            for start in range(0, length, chunk_size):
                end = min(start + chunk_size, length)
                output, state_cache = checkpointed_mamba_chunk(
                    x[:, start:end], state_cache, A
                )
                outputs.append(output)
            y = mx.concatenate(outputs, axis=1)
            return self.out_proj(mamba.swiglu(z, y)), (new_conv_cache, state_cache)

        mamba.MambaBlock._process_sequence = checkpointed_mamba_process
        mamba._mlx_lm_lora_memory_safe = True

    try:
        from mlx_lm.models import jamba
    except ImportError:
        jamba = None

    jamba_mixer = getattr(jamba, "JambaMambaMixer", None)
    if jamba_mixer is not None and not getattr(
        jamba, "_mlx_lm_lora_memory_safe", False
    ):
        original_ssm_step = jamba_mixer.ssm_step

        def checkpointed_jamba_ssm_step(self, x, A, state=None):
            if state is None:
                state = mx.zeros(
                    (x.shape[0], self.intermediate_size, self.ssm_state_size),
                    dtype=x.dtype,
                )

            def jamba_chunk(x_chunk, state, A):
                return original_ssm_step(self, x_chunk, A, state)

            checkpointed_jamba_chunk = mx.checkpoint(jamba_chunk)
            outputs = []
            for start in range(0, x.shape[1], chunk_size):
                end = min(start + chunk_size, x.shape[1])
                output, state = checkpointed_jamba_chunk(x[:, start:end], state, A)
                outputs.append(output)
            return mx.concatenate(outputs, axis=1), state

        jamba_mixer.ssm_step = checkpointed_jamba_ssm_step
        jamba._mlx_lm_lora_memory_safe = True

    # ssm_update resolves ssm_attn in its defining module, so this also covers
    # model files that imported ssm_update directly.
    from .ssm_patch import set_ssm_attention_chunk_size

    set_ssm_attention_chunk_size(chunk_size)
