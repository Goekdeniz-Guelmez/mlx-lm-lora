"""Shared Directional Latent Preference Optimization primitives."""

import mlx.core as mx
import mlx.nn as nn


def forward_logits_and_hidden(model, tokens, layer_spec="final", cache=None):
    """Run a causal LM and retain the requested residual-stream representation."""
    if cache is not None or not hasattr(model, "model"):
        logits = model(tokens, cache=cache)
        return logits, logits

    backbone = model.model
    if layer_spec == "final":
        final_hidden = backbone(tokens)
        selected_hidden = final_hidden
    else:
        layers = backbone.layers
        if layer_spec == "middle":
            layer_index = len(layers) // 2
        elif layer_spec == "late":
            layer_index = int(0.8 * len(layers))
        else:
            layer_index = int(layer_spec)
        hidden = backbone.embed_tokens(tokens)
        selected = None
        for index, layer in enumerate(layers):
            hidden = layer(hidden, mask=None)
            if index == layer_index:
                selected = hidden
        final_hidden = hidden
        selected_hidden = final_hidden if selected is None else selected

    if getattr(model.args, "tie_word_embeddings", False):
        logits = backbone.embed_tokens.as_linear(final_hidden)
    else:
        logits = model.lm_head(final_hidden)
    return logits, selected_hidden


def _masked_mean(hidden, mask):
    weights = mask.astype(hidden.dtype)[..., None]
    return (hidden * weights).sum(1) / mx.maximum(weights.sum(1), 1.0)


def _last_token(hidden, mask):
    # Reversing the mask makes argmax select the last active position.
    reverse_index = mx.argmax(mask[:, ::-1], axis=1)
    index = mask.shape[1] - 1 - reverse_index
    return hidden[mx.arange(hidden.shape[0]), index]


def _pool(hidden, response_mask, prompt_mask, pooling):
    if pooling == "answer_mean":
        return _masked_mean(hidden, response_mask)
    if pooling == "last_token":
        return _last_token(hidden, response_mask)
    if pooling == "last_k_mean":
        positions = mx.arange(response_mask.shape[1])[None, :]
        last = response_mask.shape[1] - 1 - mx.argmax(response_mask[:, ::-1], axis=1)
        last_k_mask = response_mask * (positions >= (last[:, None] - 7))
        return _masked_mean(hidden, last_k_mask)
    if pooling == "prompt_answer_mean":
        return _masked_mean(hidden, mx.minimum(prompt_mask + response_mask, 1.0))
    raise ValueError(f"Unknown DLPO pooling: {pooling}")


def latent_preference_loss(
    chosen_hidden,
    rejected_hidden,
    chosen_response_mask,
    rejected_response_mask,
    chosen_prompt_mask,
    rejected_prompt_mask,
    args,
):
    """Compute equations 17--23 of the DLPO paper."""
    chosen = _pool(
        chosen_hidden, chosen_response_mask, chosen_prompt_mask, args.latent_pooling
    )
    rejected = _pool(
        rejected_hidden,
        rejected_response_mask,
        rejected_prompt_mask,
        args.latent_pooling,
    )
    prompt = 0.5 * (
        _masked_mean(chosen_hidden, chosen_prompt_mask)
        + _masked_mean(rejected_hidden, rejected_prompt_mask)
    )

    def normalize(value):
        return value / mx.sqrt((value * value).sum(-1, keepdims=True) + 1e-6)

    losses = []
    metrics = {}
    if args.latent_variant in ("similarity", "both"):
        similarity_margin = (normalize(prompt) * normalize(chosen)).sum(-1) - (
            normalize(prompt) * normalize(rejected)
        ).sum(-1)
        similarity_loss = mx.logaddexp(
            (args.latent_margin - similarity_margin) * args.latent_gamma,
            mx.zeros_like(similarity_margin),
        ).mean()
        losses.append(similarity_loss)
        metrics.update(
            latent_sim_margin=similarity_margin.mean(),
            latent_sim_loss=similarity_loss,
        )
    if args.latent_variant in ("direction", "both"):
        directions = chosen - rejected
        batch_direction = normalize(directions.mean(0, keepdims=True))[0]
        directional_margin = (normalize(directions) * batch_direction).sum(-1)
        direction_loss = -nn.log_sigmoid(
            args.latent_gamma * directional_margin
        ).mean()
        losses.append(direction_loss)
        metrics.update(
            latent_dir_margin=directional_margin.mean(),
            latent_dir_loss=direction_loss,
        )
    if not losses:
        raise ValueError("latent_variant must be 'similarity', 'direction', or 'both'")
    latent_loss = sum(losses) / len(losses)
    metrics["latent_loss"] = latent_loss
    return latent_loss, metrics
