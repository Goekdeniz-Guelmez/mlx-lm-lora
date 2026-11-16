# KLPO: KL-Regularized Policy Optimization

## Data and settings

Use GRPO-style prompt/answer reward data and registered reward functions. The current KLPO dispatcher does not load a separate reference: collection stores behavior-policy records. Extra keys use the klpo_ prefix shown in the example, not the trainer-internal route/kl_estimator names.

## MCP example

Pass these flat arguments to `start_training`; use the same arguments with
`check_training_config` first. Hub IDs below are illustrative.

```json
{
  "model": "org/policy-model",
  "data": "org/training-dataset",
  "train_mode": "klpo",
  "train_type": "lora",
  "iters": 100,
  "batch_size": 1,
  "num_layers": 1,
  "max_seq_length": 512,
  "fuse": false,
  "extra_config": "{\"klpo_route\": \"token\", \"klpo_kl_estimator\": \"mc\", \"klpo_mc_samples\": 128, \"klpo_top_k\": 128, \"klpo_tail_floor\": 1e-06, \"beta\": 0.1, \"max_completion_length\": 128, \"reward_functions\": \"r1_accuracy_reward_func\"}"
}
```

## Best practices

Use a positive finite beta and nonconstant task rewards. Routes are token or sequence; estimators are mc, topk, binary or full. For sequence+mc use at least 2 auxiliary samples. top_k and mc_samples must be positive; tail_floor must lie strictly between 0 and 1. Start with the token/mc baseline and compare estimator variance, reward and KL on fixed prompts. Changing route or estimator changes the objective/estimator, not only memory.

## Memory efficiency

Full KL stores full-vocabulary behavior probabilities. MC and top-k reduce stored auxiliary size, but the current implementation still constructs full current-policy log-probabilities; do not promise vocabulary-independent peak memory. Binary avoids those auxiliary distributions. Lower completion length/batch first, then tune sample/head counts with a quality check. Internal scoring microbatches and checkpointing/accumulation are wired; configurable micro_batch_size, QAT and efficient_long_context are not.

See the [shared memory guide](memory-efficiency.md) for tuning order and the
[MCP contract](config.md) for defaults, monitoring and iteration-budget limits.
