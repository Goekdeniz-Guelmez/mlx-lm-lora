"""Memory-bounded selected-token scoring for MLX language models."""

import mlx.core as mx


DEFAULT_LOGIT_CHUNK_SIZE = 128


@mx.compile
def select_token_logps(logits, targets, mask):
    """Compute exact selected-token log probabilities from dense logits."""
    if logits.shape[1] == 0:
        return mx.zeros(mask.shape, dtype=mx.float32)
    logits = mx.where(mask[..., None], logits, 0).astype(mx.float32)
    logits = logits - mx.stop_gradient(logits.max(axis=-1, keepdims=True))
    selected = mx.take_along_axis(logits, targets[..., None], axis=-1).squeeze(-1)
    normalizer = mx.log(mx.exp(logits).sum(axis=-1))
    return mx.where(mask, selected - normalizer, 0)


@mx.compile
def _select_flat_token_logps(logits, targets, mask):
    """Score one flat chunk; targets may include a trailing sample axis."""
    logits = mx.where(mask[:, None], logits, 0).astype(mx.float32)
    logits = logits - mx.stop_gradient(logits.max(axis=-1, keepdims=True))
    normalizer = mx.log(mx.exp(logits).sum(axis=-1))
    if targets.ndim == 1:
        selected = mx.take_along_axis(logits, targets[:, None], axis=-1).squeeze(-1)
        return mx.where(mask, selected - normalizer, 0)
    selected = mx.take_along_axis(logits, targets, axis=-1)
    return mx.where(mask[:, None], selected - normalizer[:, None], 0)


def _output_projection(model):
    """Return the conventional MLX-LM transformer body and output head.

    A few architectures apply extra transforms after their output projection.
    Keep those on the ordinary model-call fallback instead of silently changing
    their scoring semantics.
    """
    owners = [model]
    language_model = getattr(model, "language_model", None)
    if language_model is not None:
        owners.append(language_model)

    for owner in owners:
        body = getattr(owner, "model", None)
        if body is None:
            continue

        # Some architectures apply a scalar after the output projection.
        custom_scale = False
        model_args = (
            getattr(model, "args", None),
            getattr(owner, "args", None),
            getattr(body, "args", None),
        )
        for args in model_args:
            for name in ("lm_head_multiplier", "embedding_multiplier", "logit_scale"):
                value = getattr(args, name, None)
                if value is not None and value != 1:
                    custom_scale = True
                    break
            if custom_scale:
                break
        if custom_scale:
            continue

        head = getattr(owner, "lm_head", None)
        if head is not None:
            return body, head

        embedding = getattr(body, "embed_tokens", None)
        as_linear = getattr(embedding, "as_linear", None)
        if callable(as_linear):
            return body, as_linear
    return None


def get_selected_token_logps(
    model,
    inputs,
    targets,
    mask,
    *,
    cache=None,
    chunk_size=DEFAULT_LOGIT_CHUNK_SIZE,
    return_logit_sum=False,
    position_start=0,
):
    """Return exact log probabilities for selected next-token targets.

    ``inputs`` is the full causal context. ``targets`` and ``mask`` select the
    scored sequence range; ``position_start`` gives its offset in ``inputs``.
    ``targets`` may add a final sample dimension, as used by MC/top-k KLPO.
    The transformer still processes the full context, while its vocabulary
    head runs in bounded row chunks and only selected-token scores are returned.
    Models with nonstandard output projections fall back to their public call
    API.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if (
        inputs.ndim != mask.ndim
        or inputs.shape[0] != mask.shape[0]
        or position_start < 0
        or inputs.shape[1] < position_start + mask.shape[1]
    ):
        raise ValueError("mask must select a valid sequence range in inputs")
    if targets.shape[: mask.ndim] != mask.shape:
        raise ValueError("targets must align with the input batch/sequence axes")
    mask = mask.astype(mx.bool_)

    projection = _output_projection(model)
    if projection is None:
        logits = model(inputs, cache=cache) if cache is not None else model(inputs)
        logits = logits[:, position_start : position_start + mask.shape[1]]
        logit_sum = logits.astype(mx.float32).sum() if return_logit_sum else None
        if targets.ndim == mask.ndim:
            scores = select_token_logps(logits, targets, mask)
            return (scores, logit_sum) if return_logit_sum else scores

        logits = mx.where(mask[..., None], logits, 0).astype(mx.float32)
        logits = logits - mx.stop_gradient(logits.max(axis=-1, keepdims=True))
        expanded_logits = mx.broadcast_to(
            logits[..., None, :], targets.shape + (logits.shape[-1],)
        )
        selected = mx.take_along_axis(expanded_logits, targets[..., None], axis=-1)
        selected = selected.squeeze(-1)
        normalizer = mx.log(mx.exp(logits).sum(axis=-1))
        scores = mx.where(mask[..., None], selected - normalizer[..., None], 0)
        return (scores, logit_sum) if return_logit_sum else scores

    body, head = projection
    if cache is None:
        hidden = body(inputs)
    else:
        hidden = body(inputs, cache=cache)
    if isinstance(hidden, mx.array) and hidden.ndim == inputs.ndim + 1:
        hidden = hidden[:, position_start : position_start + mask.shape[1], :]
    if not isinstance(hidden, mx.array) or hidden.ndim != inputs.ndim + 1:
        logits = model(inputs, cache=cache) if cache is not None else model(inputs)
        logits = logits[:, position_start : position_start + mask.shape[1]]
        logit_sum = logits.astype(mx.float32).sum() if return_logit_sum else None
        if targets.ndim == mask.ndim:
            scores = select_token_logps(logits, targets, mask)
            return (scores, logit_sum) if return_logit_sum else scores
        logits = mx.where(mask[..., None], logits, 0).astype(mx.float32)
        logits = logits - mx.stop_gradient(logits.max(axis=-1, keepdims=True))
        expanded_logits = mx.broadcast_to(
            logits[..., None, :], targets.shape + (logits.shape[-1],)
        )
        selected = mx.take_along_axis(expanded_logits, targets[..., None], axis=-1)
        normalizer = mx.log(mx.exp(logits).sum(axis=-1))
        scores = mx.where(
            mask[..., None], selected.squeeze(-1) - normalizer[..., None], 0
        )
        return (scores, logit_sum) if return_logit_sum else scores

    flat_hidden = hidden.reshape((-1, hidden.shape[-1]))
    flat_targets = targets.reshape(
        (mask.size,) if targets.ndim == mask.ndim else (mask.size, targets.shape[-1])
    )
    flat_mask = mask.reshape((-1,))
    scores = []
    logit_sum = mx.array(0.0, dtype=mx.float32) if return_logit_sum else None
    for start in range(0, flat_hidden.shape[0], chunk_size):
        stop = min(start + chunk_size, flat_hidden.shape[0])
        logits = head(flat_hidden[start:stop])
        if return_logit_sum:
            logit_sum = logit_sum + logits.astype(mx.float32).sum()
        scores.append(
            _select_flat_token_logps(
                logits,
                flat_targets[start:stop],
                flat_mask[start:stop],
            )
        )
    if not scores:
        output_shape = targets.shape
        result = mx.zeros(output_shape, dtype=mx.float32)
    else:
        result = mx.concatenate(scores, axis=0).reshape(targets.shape)
    return (result, logit_sum) if return_logit_sum else result


def get_last_token_logits(model, inputs, lengths):
    """Project only each row's final non-padding hidden state to vocabulary."""
    projection = _output_projection(model)
    if projection is None:
        logits = model(inputs).astype(mx.float32)
        indices = (lengths - 1).astype(mx.int32)[:, None, None]
        indices = mx.broadcast_to(indices, (logits.shape[0], 1, logits.shape[-1]))
        return mx.take_along_axis(logits, indices, axis=1).squeeze(1)

    body, head = projection
    hidden = body(inputs)
    row_indices = mx.arange(inputs.shape[0])
    final_hidden = hidden[row_indices, (lengths - 1).astype(mx.int32)]
    return head(final_hidden)
