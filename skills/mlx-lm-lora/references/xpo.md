# XPO: Exploratory Preference Optimization

## Data and settings

Use prompt-only rows and a model judge. XPO augments online preference learning with exploration. Extra keys include judge, judge_config, beta, alpha, dpo_cpo_loss_type, delta, max_completion_length and optional reference_model_path. alpha is a list of exploration weights, not a scalar.

## MCP example

Pass these flat arguments to `start_training`; use the same arguments with
`check_training_config` first. Hub IDs below are illustrative.

```json
{
  "model": "org/policy-model",
  "data": "org/training-dataset",
  "train_mode": "xpo",
  "train_type": "lora",
  "iters": 100,
  "batch_size": 1,
  "num_layers": 1,
  "max_seq_length": 512,
  "fuse": false,
  "extra_config": "{\"judge\": \"org/judge-model\", \"beta\": 0.1, \"alpha\": [1e-05, 5e-06], \"max_completion_length\": 128, \"micro_batch_size\": 1}"
}
```

## Best practices

Compare against Online DPO on the same prompts and judge. Begin with small exploration weights and inspect KL, preference gains and generations for capability drift. Always supply an explicit nonempty alpha list: the shared CLI default is a scalar, while XPO's scheduler expects a list. The active scheduler selects entries over segments of total training iterations (despite its dataclass help mentioning epochs). Keep the schedule length no greater than iters to avoid a zero segment size; a one-entry list gives a constant exploration weight. These are not learning rates. The CLI dispatch does not forward a separate temperature to XPO, so an accepted config key is not evidence it changes sampling.

## Memory efficiency

Budget policy/reference/judge and paired rollout storage. Use short rollouts, small batches, checkpointing, accumulation and micro_batch_size. Larger exploration can alter completion lengths and therefore observed memory. QAT and efficient_long_context are not wired.

See the [shared memory guide](memory-efficiency.md) for tuning order and the
[MCP contract](config.md) for defaults, monitoring and iteration-budget limits.
