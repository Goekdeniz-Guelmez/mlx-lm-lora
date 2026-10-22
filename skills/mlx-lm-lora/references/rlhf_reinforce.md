# RLHF REINFORCE with KL regularization

## Data and settings

Use exact train_mode `rlhf_reinforce`, prompt-only data and a scalar-reward model judge. The current trainer samples two responses per prompt. Extra keys include judge, judge_config, beta, max_completion_length and optional reference_model_path.

## MCP example

Pass these flat arguments to `start_training`; use the same arguments with
`check_training_config` first. Hub IDs below are illustrative.

```json
{
  "model": "org/policy-model",
  "data": "org/training-dataset",
  "train_mode": "rlhf_reinforce",
  "train_type": "lora",
  "iters": 100,
  "batch_size": 1,
  "num_layers": 1,
  "max_seq_length": 512,
  "fuse": false,
  "extra_config": "{\"judge\": \"org/reward-model\", \"judge_config\": {\"system_prompt\": \"Score helpfulness from 0 to 1.\"}, \"beta\": 0.1, \"max_completion_length\": 128, \"micro_batch_size\": 1}"
}
```

## Best practices

Calibrate scalar rewards on obviously good/bad/ambiguous completions before training. Use a conservative learning rate and inspect reward versus KL and general capability to detect reward hacking. Keep prompt data diverse and judge/reference fixed. alpha, epsilon, dpo_cpo_loss_type, registered reward functions and a separate temperature are not consumed by the current REINFORCE dispatch.

## Memory efficiency

Account for policy/reference/judge plus two generated completions. Shorten rollout caps and minibatches, then use checkpointing, accumulation and micro_batch_size for scoring. These controls do not remove extra model residency. QAT and efficient_long_context are not wired.

See the [shared memory guide](memory-efficiency.md) for tuning order and the
[MCP contract](config.md) for defaults, monitoring and iteration-budget limits.
