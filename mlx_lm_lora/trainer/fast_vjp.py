"""Opt into MLX's accelerated attention VJPs when the installed build has them."""

from __future__ import annotations

import sys
import warnings
from functools import wraps

import mlx.core as mx

_WARNED_MISSING_GATED_DELTA_VJP = False


def enable_fast_vjps() -> bool:
    """Route supported gated-delta training calls through MLX's fast primitive.

    Standard attention already goes through ``mx.fast.scaled_dot_product_attention``
    in mlx-lm, so MLX selects its attention VJP without trainer-side changes.
    Newer MLX builds also expose ``mx.fast.gated_delta_update``. This function
    connects mlx-lm's Qwen3.5/Next and other gated-delta layers to that op for
    unmasked training; MLX chooses the Metal or Neural Accelerator kernel when
    the device and shape support it. Masked and otherwise unsupported calls
    continue through mlx-lm's existing implementation.

    Returns whether the installed MLX build provides the fast gated-delta op.
    """
    global _WARNED_MISSING_GATED_DELTA_VJP

    from mlx_lm.models import gated_delta

    original = getattr(
        gated_delta,
        "_mlx_lm_lora_fast_vjp_original",
        gated_delta.gated_delta_update,
    )
    patched = getattr(gated_delta, "_mlx_lm_lora_fast_vjp_wrapper", None)
    model_modules = tuple(
        module
        for module in sys.modules.values()
        if module is not None
        and getattr(module, "__name__", "").startswith("mlx_lm.models.")
        and getattr(module, "gated_delta_update", None) in (original, patched)
    )

    fast_update = getattr(getattr(mx, "fast", None), "gated_delta_update", None)
    if not callable(fast_update):
        if model_modules and not _WARNED_MISSING_GATED_DELTA_VJP:
            warnings.warn(
                "This MLX build does not expose mx.fast.gated_delta_update; "
                "gated-delta training will use the existing checkpointed fallback. "
                "Upgrade to an MLX build containing the gated-delta VJP to enable "
                "the accelerated path.",
                RuntimeWarning,
                stacklevel=2,
            )
            _WARNED_MISSING_GATED_DELTA_VJP = True
        return False

    if patched is None:

        @wraps(original)
        def patched(
            q,
            k,
            v,
            a,
            b,
            A_log,
            dt_bias,
            state=None,
            mask=None,
            use_kernel=True,
            **kwargs,
        ):
            lower_bound = kwargs.get("lower_bound")
            allow_neg_eigval = kwargs.get("allow_neg_eigval", False)
            unsupported_options = kwargs.keys() - {
                "lower_bound",
                "allow_neg_eigval",
            }
            if (
                not use_kernel
                and mask is None
                and lower_bound is None
                and not allow_neg_eigval
                and not unsupported_options
                # MLX's fused GDN kernels assume scalar gates. Vector-gated
                # models (for example Kimi Linear) must use the ops fallback.
                and a.ndim == 3
                and b.ndim == 3
            ):
                beta = mx.sigmoid(b)
                g = gated_delta.compute_g(A_log, a, dt_bias)
                if state is None:
                    batch, _, _, key_dim = q.shape
                    value_heads, value_dim = v.shape[-2:]
                    state = mx.zeros(
                        (batch, value_heads, value_dim, key_dim), dtype=mx.float32
                    )
                return fast_update(q, k, v, g, beta, initial_state=state)

            return original(
                q,
                k,
                v,
                a,
                b,
                A_log,
                dt_bias,
                state,
                mask,
                use_kernel=use_kernel,
                **kwargs,
            )

        gated_delta._mlx_lm_lora_fast_vjp_original = original
        gated_delta._mlx_lm_lora_fast_vjp_wrapper = patched
        gated_delta.gated_delta_update = patched

    # Model files import this function directly, so update their bound module
    # globals as well as the defining gated_delta module. Models loaded later
    # receive the wrapper through the normal import.
    for module in model_modules:
        if (
            getattr(module, "gated_delta_update", None) is original
        ):
            module.gated_delta_update = patched

    return True
