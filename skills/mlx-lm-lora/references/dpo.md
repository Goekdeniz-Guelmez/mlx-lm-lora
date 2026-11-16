# DPO: Direct Preference Optimization

## Data and settings

Rows require `prompt`, `chosen`, `rejected` (optional `system`). A frozen reference defaults to the base model. Extra keys include `beta` (0.1), `dpo_cpo_loss_type` (sigmoid/hinge/ipo/dpop), `delta` (50) and optional `reference_model_path`.

## MCP example

Pass these flat arguments to `start_training`; use the same arguments with
`check_training_config` first. Hub IDs below are illustrative.

```json
{
  "model": "org/policy-model",
  "data": "org/training-dataset",
  "train_mode": "dpo",
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

Start with sigmoid and trustworthy same-prompt pairs. Preserve the exact starting reference checkpoint. Track preference accuracy, chosen/rejected reward margins, validation and general capabilities. IPO uses sequence-mean scoring and needs a matched beta; DPOP adds a penalty controlled by delta. Compare variants on the same data and do not tune beta and learning rate simultaneously without a baseline.

## Memory efficiency

Budget a second resident model and paired scoring. Policy quantization does not quantize the reference. Reduce physical batch/context, use accumulation/checkpointing, and evaluate fixed sequential long-context chunks before deploying them. QAT is wired.

See the [shared memory guide](memory-efficiency.md) for tuning order and the
[MCP contract](config.md) for defaults, monitoring and iteration-budget limits.
