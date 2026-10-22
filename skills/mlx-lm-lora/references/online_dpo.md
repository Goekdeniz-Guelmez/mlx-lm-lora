# Online DPO

## Data and settings

Use prompt-only rows and supply a model `judge`. The trainer generates two responses and learns from judge preferences. Extra keys include judge_config, beta, dpo_cpo_loss_type, delta, temperature, max_completion_length and optional reference_model_path.

## MCP example

Pass these flat arguments to `start_training`; use the same arguments with
`check_training_config` first. Hub IDs below are illustrative.

```json
{
  "model": "org/policy-model",
  "data": "org/training-dataset",
  "train_mode": "online_dpo",
  "train_type": "lora",
  "iters": 100,
  "batch_size": 1,
  "num_layers": 1,
  "max_seq_length": 512,
  "fuse": false,
  "extra_config": "{\"judge\": \"org/judge-model\", \"beta\": 0.1, \"dpo_cpo_loss_type\": \"sigmoid\", \"temperature\": 0.8, \"max_completion_length\": 128, \"micro_batch_size\": 1}"
}
```

## Best practices

Validate the pairwise judge with clear, ambiguous and reversed response pairs. Use a fixed prompt pool, reference and judge for comparisons. Start with sigmoid and conservative learning rate; inspect generated pair diversity and preference consistency. Monitor held-out capability and KL rather than trusting judge win rate alone.

## Memory efficiency

Budget policy, reference and judge plus two rollouts per prompt. Matching an explicit judge ID to an explicit reference path can reuse that reference; implicit defaults do not guarantee reuse. Reduce completion cap/batch, enable checkpointing and set micro_batch_size for scoring/backprop. This does not cap generation memory. QAT and efficient_long_context are not wired.

See the [shared memory guide](memory-efficiency.md) for tuning order and the
[MCP contract](config.md) for defaults, monitoring and iteration-budget limits.
