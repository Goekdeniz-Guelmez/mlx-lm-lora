# Final Token Preference Optimization / Antidoom

Use `train_mode: "ftpo"` for targeted final-token preference repair. FTPO
scores the next-token distribution after a templated context. Shared controls
are in [config.md](config.md); generic chosen/rejected sequence records are
not the FTPO input schema.

## Antidoom records

```json
{
  "context_with_chat_template": "Conversation context rendered for this tokenizer",
  "rejected_decoded": "token text",
  "multi_chosen_decoded": ["token text", "another token text"]
}
```

The loader tokenizes the context and response surfaces with the policy
tokenizer. It requires one rejected token, removes duplicate chosen tokens,
keeps only one-token chosen surfaces, and requires at least one usable chosen
token. Empty contexts and contexts exceeding `max_seq_length` are skipped;
the limit is a filter rather than ordinary truncation.

## Loss settings

| Field | Default | Meaning |
| --- | --- | --- |
| `lambda_mse_target` | `0.05` | Target-token MSE weight |
| `tau_mse_target` | `1.0` | Target-logit deviation threshold |
| `lambda_mse` | `0.4` | Non-target-logit MSE weight |
| `clip_epsilon_logits` | `2.0` | Positive preference-margin clipping scale |
| `reference_model_path` | `null` | Frozen reference; null uses the starting model |

## Example

```json
{
  "model": "org/model",
  "data": "org/antidoom-dataset",
  "train": true,
  "train_type": "lora",
  "train_mode": "ftpo",
  "lambda_mse_target": 0.05,
  "tau_mse_target": 1.0,
  "lambda_mse": 0.4,
  "clip_epsilon_logits": 2.0,
  "iters": 100
}
```

FTPO supports quantized loading, checkpointing, accumulation, and recurrent
fallbacks, but does not enable QAT or `efficient_long_context` through the
entrypoint. Read [memory.md](memory.md) and [quantization.md](quantization.md).

Check usable row counts after token filtering. Evaluate repetition/loop
repair on held-out contexts alongside general model behavior; FTPO loss alone
does not measure whether the targeted generation problem improved.
