# FTPO: Final Token Preference Optimization

## Data and settings

Use Antidoom rows with `context_with_chat_template`, `rejected_decoded`, `multi_chosen_decoded`. This is next-token loop repair, not general sequence DPO. Extra keys: `lambda_mse_target` (0.05), `tau_mse_target` (1), `lambda_mse` (0.4), `clip_epsilon_logits` (2), optional `reference_model_path`.

## MCP example

Pass these flat arguments to `start_training`; use the same arguments with
`check_training_config` first. Hub IDs below are illustrative.

```json
{
  "model": "org/policy-model",
  "data": "org/training-dataset",
  "train_mode": "ftpo",
  "train_type": "lora",
  "iters": 100,
  "batch_size": 1,
  "num_layers": 1,
  "max_seq_length": 512,
  "fuse": false,
  "extra_config": "{\"lambda_mse_target\": 0.05, \"tau_mse_target\": 1.0, \"lambda_mse\": 0.4, \"clip_epsilon_logits\": 2.0}"
}
```

## Best practices

Prepare data with the same tokenizer and chat template as the base model. Count rows that survive single-token chosen/rejected filtering and context-length filtering. Keep the reference fixed and clip_epsilon_logits positive. Evaluate repetition-loop rate and general capability; excessive MSE regularization can suppress useful movement.

## Memory efficiency

Budget the frozen reference and vocabulary comparisons even though the objective targets one next token. Lower context and batch carefully: overlong contexts are skipped rather than simply truncated. Checkpointing and accumulation are wired; efficient_long_context and QAT are not.

See the [shared memory guide](memory-efficiency.md) for tuning order and the
[MCP contract](config.md) for defaults, monitoring and iteration-budget limits.
