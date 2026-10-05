# Online Direct Preference Optimization

Use `train_mode: "online_dpo"` to generate two responses per prompt, rank them
with a model judge, and optimize DPO against a frozen reference. Read
[datasets.md](datasets.md) for prompt rows, [judges.md](judges.md) for required
judge configuration, and [config.md](config.md) for common settings.

## Online DPO settings

| Field | MCP default | Meaning |
| --- | --- | --- |
| `judge` | Required | Hub model or approved local judge model |
| `judge_config` | `{}` | Pairwise judge template settings |
| `reference_model_path` | `null` | Frozen reference; null uses the starting model |
| `beta` | `0.1` | Preference-logit scale |
| `dpo_cpo_loss_type` | `"sigmoid"` | `"sigmoid"`, `"hinge"`, `"ipo"`, or `"dpop"` |
| `delta` | `50.0` | DPOP penalty coefficient |
| `temperature` | `1.0` | Candidate sampling temperature |
| `max_completion_length` | `512` | Maximum generated tokens per candidate |

Loss variants are described in [dpo.md](dpo.md). Online DPO obtains pairs by
generation and judging rather than reading an offline preference dataset.

## Example

```json
{
  "model": "org/model",
  "data": "org/prompt-dataset",
  "train": true,
  "train_type": "lora",
  "train_mode": "online_dpo",
  "judge": "org/judge-model",
  "dpo_cpo_loss_type": "sigmoid",
  "beta": 0.1,
  "temperature": 0.8,
  "max_completion_length": 256,
  "micro_batch_size": 1,
  "iters": 100
}
```

Scoring microbatches, checkpointing, accumulation, and recurrent fallbacks are
covered in [memory.md](memory.md). QAT, `efficient_long_context`, GRPO group
settings, and reward callbacks are not used by this mode.

Validate the judge on good, bad, and ambiguous responses. Compare held-out
outputs and KL drift as well as preference metrics; a noisy judge can produce
misleading training targets even when the optimizer behaves correctly.
