# Shared MCP configuration

Use this reference for field mapping and options shared across algorithms.
The connected server's `mlx_lm_lora_get_capabilities` response takes precedence
when versions differ. Algorithm-specific options belong in each mode guide.

## Request shape

Training tools receive `config` and an optional `tenant_id` as separate tool
arguments. `model` and `data` are required. MCP forces `train: true` and chooses
an isolated artifact directory when `adapter_path` is omitted.

```json
{
  "tenant_id": "alice",
  "config": {
    "model": "org/model",
    "data": "org/dataset",
    "train": true,
    "train_mode": "sft",
    "train_type": "lora",
    "iters": 100
  }
}
```

Preserve Hub identifiers exactly. Dataset sources and schemas are in
[datasets.md](datasets.md); approved local paths are in
[multi-tenant.md](multi-tenant.md).

## Map the user's words

| User wording | Field | Value |
| --- | --- | --- |
| LoRA / DoRA / full fine-tuning | `train_type` | `"lora"` / `"dora"` / `"full"` |
| Training algorithm | `train_mode` | Exact mode name from capabilities |
| Steps / iterations | `iters` | Positive integer |
| Epochs | `epochs` | Positive integer |
| Maximum context length | `max_seq_length` | Positive integer |
| Batch size | `batch_size` | Positive integer |
| Gradient accumulation | `gradient_accumulation_steps` | Positive integer |
| Scoring microbatch | `micro_batch_size` | See [memory.md](memory.md) |
| Recurrent chunk size | `recurrence_chunk_size` | See [memory.md](memory.md) |
| Quantization / QAT | Loading flags / `qat_*` | See [quantization.md](quantization.md) |
| Save every N steps | `save_every` | Positive integer |
| Resume adapter | `resume_adapter_file` | Approved local weights file |

`iters` takes precedence over `epochs`. Without either, the backend runs 100
iterations. `iters` counts trainer iterations; accumulation can yield fewer
optimizer updates. Preserve an explicit budget; do not add both fields
unnecessarily.

## Shared defaults

These are effective defaults for the MCP dictionary entrypoint.

| Field | Default | Meaning |
| --- | --- | --- |
| `train_mode` | `"sft"` | Algorithm |
| `train_type` | `"lora"` | LoRA, DoRA, or full fine-tuning |
| `optimizer` | `"adam"` | `"adam"`, `"adamw"`, or `"muon"` |
| `optimizer_config` | `{"adam": {}, "adamw": {}, "muon": {}}` | Keyword arguments keyed by optimizer name |
| `learning_rate` | `0.00001` | Finite positive learning rate |
| `lr_schedule` | `null` | MLX-LM schedule mapping; overrides the constant learning rate |
| `batch_size` | `1` | Minibatch or prompt batch size |
| `gradient_accumulation_steps` | `1` | Minibatches accumulated before an update |
| `num_layers` | `-1` | All adapter layers; positive values select fewer layers |
| `lora_parameters` | `{"rank": 8, "dropout": 0.0, "scale": 10.0}` | LoRA/DoRA settings; not used for full fine-tuning |
| `max_seq_length` | `2048` | Token limit, with mode-specific truncation/filtering |
| `val_batches` | `25` | Validation batches; `-1` uses all |
| `steps_per_report` | `10` | Training-log interval |
| `steps_per_eval` | `200` | Validation interval |
| `save_every` | `100` | Checkpoint interval |
| `seed` | `0` | Random seed |
| `wandb` | `null` | Optional Weights & Biases project |
| `test` | `false` | Evaluate the test split after training |
| `test_batches` | `500` | Test batches; `-1` uses all |
| `fuse` | `true` | Merge and save the trained model after training/testing |
| `resume_adapter_file` | `null` | Resume weights; does not restore optimizer/job state |
| `adapter_path` | Per-job tenant artifact directory | Adapter and fused-model output |

For an adapter-only run, explicitly set `fuse: false`. For distributed trainer
use, the batch must be divisible by the worker count; MCP queues one training
job at a time on the host.

Omit unused optional fields. Use JSON booleans and numbers rather than string
spellings, except fields with a specific string format such as reward names
and weights. Memory defaults are in [memory.md](memory.md), and QAT defaults
in [quantization.md](quantization.md).

## Validation boundaries

MCP accepts explicit option values, not a YAML `config` file. `lm_studio_name`
is unsupported because it writes outside the tenant workspace. Use
`mlx_lm_lora_list_reward_functions` instead of putting
`list_reward_functions: true` in a training config.

Validation checks keys, choices, numeric bounds, compatible features, and
local path boundaries. It does not load models or datasets; Hub availability,
split contents, tokenizer compatibility, model-specific layer indices, and
memory capacity are checked when training runs.
