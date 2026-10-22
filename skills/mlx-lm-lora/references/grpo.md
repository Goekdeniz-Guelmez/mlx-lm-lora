# GRPO: Group Relative Policy Optimization

## Data and settings

Use `prompt`, `answer`, optionally `system` and `type`. A frozen reference is loaded. Extra keys include group_size (4), max_completion_length (512), temperature (0.8 through CLI), beta (0.1), epsilon (0.0001), epsilon_high, registered reward_functions and string-encoded reward_weights.

## MCP example

Pass these flat arguments to `start_training`; use the same arguments with
`check_training_config` first. Hub IDs below are illustrative.

```json
{
  "model": "org/policy-model",
  "data": "org/training-dataset",
  "train_mode": "grpo",
  "train_type": "lora",
  "iters": 100,
  "batch_size": 1,
  "num_layers": 1,
  "max_seq_length": 512,
  "fuse": false,
  "extra_config": "{\"group_size\": 4, \"max_completion_length\": 128, \"beta\": 0.1, \"reward_functions\": \"r1_accuracy_reward_func,r1_strict_format_reward_func\", \"reward_weights\": \"[0.8, 0.2]\"}"
}
```

## Best practices

Inspect raw completions and verify reward functions produce meaningful variation within a prompt group. Use at least two completions for informative group-relative advantages. Defaults reward think/answer formatting; match task output and answer references. Monitor rewards, KL, clip behavior and hit_max_tokens_ratio. A zero/constant reward or identical completions gives little learning signal. See the variants guide for sequence ratios and alternative normalization.

## Memory efficiency

Generated work grows with prompts times group_size times completion length. Internal scoring microbatches help backward peaks but do not remove rollout/KV storage. Reduce physical batch and completion caps before enlarging groups; do not use group_size=1 as a meaningful GRPO shortcut. Checkpointing/accumulation are wired; configurable micro_batch_size, QAT and efficient_long_context are not. beta=0 still leaves the dispatcher loading a reference.

See the [shared memory guide](memory-efficiency.md) for tuning order and the
[MCP contract](config.md) for defaults, monitoring and iteration-budget limits.
