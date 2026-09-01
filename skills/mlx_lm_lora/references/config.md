# MCP training configuration

The MCP tools accept a `config` object matching the training CLI's options.
The server requires `model` and `data`, forces `train: true` when a job starts,
and chooses a tenant-scoped `adapter_path` when one is not supplied.

`data` must be a Hugging Face dataset repository ID, for example
`mlx-community/wikisql` or `org/dataset`. Do not send a local JSONL/CSV path,
`tenant://` dataset path, or HTTP URL as `data`. Tenant-local paths are only
for supported auxiliary inputs such as reward functions and resume adapters.

Common natural-language mappings:

| User wording | MCP field | Example |
| --- | --- | --- |
| LoRA | `train_type` | `"lora"` |
| SFT | `train_mode` | `"sft"` |
| 10 steps/iterations | `iters` | `10` |
| 3 epochs | `epochs` | `3` |
| max context 512 | `max_seq_length` | `512` |
| batch size 2 | `batch_size` | `2` |
| learning rate 1e-5 | `learning_rate` | `0.00001` |
| save every 100 steps | `save_every` | `100` |

Do not send CLI spellings such as `--train-mode`; use JSON field names such as
`train_mode`. Do not send a YAML `config` path through MCP. Put all requested
options directly in the object.

Example:

```json
{
  "model": "Qwen/Qwen3.5-0.8B",
  "data": "mlx-community/wikisql",
  "train": true,
  "train_type": "lora",
  "train_mode": "sft",
  "load_in_4bits": true,
  "iters": 1,
  "max_seq_length": 512
}
```
