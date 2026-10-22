# MCP contract and configuration

The current server exposes these tools. Client frameworks can add a namespace
prefix; match discovered tool names instead of inventing historical names.

| Tool | Arguments | Result |
| --- | --- | --- |
| `list_training_modes` | none | Modes, train types, optimizers and available rewards |
| `check_training_config` | Flat training settings below | Text verdict and generated YAML |
| `start_training` | Same settings plus optional `name` | Text plus `structured_content.id`, status and paths |
| `training_status` | `run_id` (optional, latest by default) | Text; structured status when a run exists |
| `training_log` | `run_id`, `lines` (default 30, capped at 200) | Log tail text |
| `list_training_runs` | `limit` (default 10, capped at 50) | Text plus `structured_content.items` |
| `stop_training` | `run_id` | Stop result text; saved checkpoints remain |

Use explicit run IDs for follow-ups. Empty/latest selects a run implicitly.
Status values are `starting`, `running`, `completed`, `failed`, `stopped`.
The structured result contains `id`, not `job_id`. No capabilities, upload,
tenant-selection or tenant-list tools exist in this API.

| Flat argument | Server default | Meaning |
| --- | --- | --- |
| `model`, `data` | Required | Hub IDs or local sources |
| `train_mode` | `sft` | One of 12 modes in `list_training_modes` |
| `train_type` | `lora` | `lora`, `dora`, `full` |
| `iters` | 100 | Positive integer iterations |
| `batch_size` | 1 | Per-process minibatch size; generation work is mode-dependent |
| `learning_rate` | 0.00001 | Positive finite learning rate |
| `num_layers` | 16 | Adapter layers; -1 selects all; small models need a suitable count |
| `max_seq_length` | 2048 | Positive context/sequence limit |
| `quantize_bits` | 0 | 0, 4, 6, 8; applies to policy loading |
| `fuse` | false | Merge/save model after training |
| `extra_config` | Empty string | JSON/YAML object encoded as a string |

Backend defaults differ from the flat server defaults. Common extra keys:
`optimizer` (adam; also adamw/muon), `optimizer_config`, `seed` (0),
`gradient_accumulation_steps` (1), `micro_batch_size` (null),
`grad_checkpoint` (false), `steps_per_report` (10), `steps_per_eval` (200),
`save_every` (100), `val_batches` (25), `test` (false), `test_batches` (500),
`lora_parameters` (rank 8, scale 10, dropout 0), `mask_prompt` (false).
Consult each mode for which keys its dispatch actually forwards.

`extra_config` merges before validation and can override flat settings, except
the server strips `adapter_path` and `config`. Unknown keys are only warned
about; accepted YAML does not prove the trainer consumes a key. Reject malformed
values before launch and inspect final generated YAML for conflicting choices.
Do not send optimizer objects, Python callbacks or CLI flags through MCP.

## Example: real SFT smoke-test arguments

These arguments completed three steps on the Mac through MCP. For validation,
send the same object without `name`. No raw training CLI is needed.

```json
{
  "model": "tiny-random/qwen3",
  "data": "mlx-community/wikisql",
  "train_mode": "sft",
  "train_type": "lora",
  "iters": 3,
  "batch_size": 1,
  "num_layers": 1,
  "max_seq_length": 128,
  "fuse": false,
  "name": "tiny-mcp-smoke",
  "extra_config": "{\"steps_per_report\": 1, \"steps_per_eval\": 3, \"val_batches\": 1, \"save_every\": 3, \"lora_parameters\": {\"rank\": 2, \"scale\": 4.0, \"dropout\": 0.0}}"
}
```

## Monitoring limits

`check_training_config` returns text even when validation fails. Inspect its
verdict, not just protocol success. Loss parsing is generic and does not
expose every trainer metric. Use logs for rewards, KL, clipping, latent metrics
and peak memory. Download/dataset progress bars can briefly be mistaken for
training progress before the first `Iter` line; confirm iteration in the log.
Stopping a run sends TERM to its process group, then KILL after the wait;
an unsaved in-flight update can be lost. Resume loads weights, not a promise
of restoring optimizer state, RNG and the exact data cursor.

## Epoch budgets

The backend can derive iterations from epochs only when `iters` is absent/null.
This server always supplies a positive `iters`, so `extra_config` containing
only `epochs` does not implement epoch training. Use a verified step budget.
The backend formula is epochs × ceil(training rows / batch size), before
trainer-specific distributed/accumulation semantics. Do not call every iteration
an optimizer update: with accumulation, these counts can differ by trainer.
