# ORPO: Odds Ratio Preference Optimization

## Data and settings

Rows require `prompt`, `chosen`, `rejected`; optional numeric `preference_score` defaults to 1. No frozen reference is loaded. Set `beta` in `extra_config`. `reward_scaling` is passed but does not currently change the active ORPO objective.

## MCP example

Pass these flat arguments to `start_training`; use the same arguments with
`check_training_config` first. Hub IDs below are illustrative.

```json
{
  "model": "org/policy-model",
  "data": "org/training-dataset",
  "train_mode": "orpo",
  "train_type": "lora",
  "iters": 100,
  "batch_size": 1,
  "num_layers": 1,
  "max_seq_length": 512,
  "fuse": false,
  "extra_config": "{\"beta\": 0.1}"
}
```

## Best practices

Compare against SFT because ORPO includes chosen-response likelihood and an odds-ratio preference term. Clean inverted labels and inspect preference_score scale and outliers. Start with beta 0.1 as a baseline rather than treating it as universally optimal. Evaluate both preference quality and ordinary instruction-following before fusion.

## Memory efficiency

Reference-free residency helps, but pair activations and logits remain. Use checkpointing, lower physical batch and accumulation. Sequential long-context chunks and QAT are wired; compare quality before relying on either.

See the [shared memory guide](memory-efficiency.md) for tuning order and the
[MCP contract](config.md) for defaults, monitoring and iteration-budget limits.
