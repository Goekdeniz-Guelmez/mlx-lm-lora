# PPO: repository pairwise PPO-style trainer

## Data and settings

Use prompt-only rows and a model judge. This implementation generates pairs, uses judge feedback and applies a clipped preference objective with a KL term; do not assume a conventional separate value-network PPO pipeline. Extra keys include judge, judge_config, reference_model_path, beta, epsilon, temperature, dpo_cpo_loss_type, delta and max_completion_length.

## MCP example

Pass these flat arguments to `start_training`; use the same arguments with
`check_training_config` first. Hub IDs below are illustrative.

```json
{
  "model": "org/policy-model",
  "data": "org/training-dataset",
  "train_mode": "ppo",
  "train_type": "lora",
  "iters": 100,
  "batch_size": 1,
  "num_layers": 1,
  "max_seq_length": 512,
  "fuse": false,
  "extra_config": "{\"judge\": \"org/judge-model\", \"beta\": 0.1, \"epsilon\": 0.2, \"temperature\": 0.8, \"max_completion_length\": 128, \"micro_batch_size\": 1}"
}
```

## Best practices

Test judge consistency and freeze the reference for comparisons. Explicitly choose epsilon for PPO; the shared CLI default is 0.0001, which comes from GRPO and is different from the commonly used 0.2 pilot value. Treat 0.2 as a starting experiment, not a guarantee. Monitor KL, clipping, judge agreement and held-out generations. epsilon_high is a GRPO setting and is not consumed by PPO. Human-judge stdin interaction is not supported by detached MCP execution.

## Memory efficiency

Budget paired generation plus reference/judge residency. Reduce rollout lengths/batch and use scoring micro_batch_size, checkpointing and accumulation. Do not expect microbatching to unload extra models or cap all rollout caches. QAT and efficient_long_context are not wired.

See the [shared memory guide](memory-efficiency.md) for tuning order and the
[MCP contract](config.md) for defaults, monitoring and iteration-budget limits.
