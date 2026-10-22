# CPO: Contrastive Preference Optimization

## Data and settings

Rows require `prompt`, `chosen`, `rejected`. No frozen reference is loaded. Use `beta`, `dpo_cpo_loss_type` and `delta` in `extra_config` for the preference objective.

## MCP example

Pass these flat arguments to `start_training`; use the same arguments with
`check_training_config` first. Hub IDs below are illustrative.

```json
{
  "model": "org/policy-model",
  "data": "org/training-dataset",
  "train_mode": "cpo",
  "train_type": "lora",
  "iters": 100,
  "batch_size": 1,
  "num_layers": 1,
  "max_seq_length": 512,
  "fuse": false,
  "extra_config": "{\"beta\": 0.1, \"dpo_cpo_loss_type\": \"sigmoid\"}"
}
```

## Best practices

Use clean same-prompt pairs and a conservative learning rate. Compare held-out preference quality and general capability because there is no frozen-reference KL anchor. Start with sigmoid; use IPO or DPOP only with an explicit comparison and appropriate beta/delta. Do not assume that an inherited dataclass field is active in this dispatcher.

## Memory efficiency

Avoids reference-model residency but still scores pairs. Use smaller minibatches, accumulation and checkpointing. Sequential long-context chunks are wired; QAT is not forwarded by the current CPO dispatch.

See the [shared memory guide](memory-efficiency.md) for tuning order and the
[MCP contract](config.md) for defaults, monitoring and iteration-budget limits.
