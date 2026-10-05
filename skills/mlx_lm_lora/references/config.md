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
| scoring microbatch of 1 | `micro_batch_size` | `1` |
| recurrent chunks of 32 tokens | `recurrence_chunk_size` | `32` |
| mixed FP4 / MXFP4 | `load_in_mxfp4` | `true` |
| DSLA with ORPO objective | `train_mode`, `dsla_loss` | `"dsla"`, `"orpo"` |
| sequence KLPO with TopK-KL | `train_mode`, `klpo_route`, `klpo_kl_estimator` | `"klpo"`, `"sequence"`, `"topk"` |
| learning rate 1e-5 | `learning_rate` | `0.00001` |
| save every 100 steps | `save_every` | `100` |

Do not send CLI spellings such as `--train-mode`; use JSON field names such as
`train_mode`. Do not send a YAML `config` path through MCP. Put all requested
options directly in the object.

Discover accepted fields and enum values from `mlx_lm_lora_get_capabilities`.
Its `features` describes mode support for QAT, cached long-context processing,
microbatches, DSLA, and KLPO. The MCP default prompt batch is 1 and recurrent
chunk size is 64. Optional fields may be omitted to use backend defaults.

Mode-specific details live in [dsla.md](dsla.md), [klpo.md](klpo.md), and
[memory.md](memory.md). A config can pass structural validation while failing
later on an unavailable Hub repository, incompatible dataset schema, model
architecture, or insufficient memory; inspect the job status and log.

`qat_group_size: 0` means per-tensor quantization. Set only one quantized-loading
flag. Online DPO, XPO, RLHF REINFORCE, and PPO require `judge` to name a model.
Use the reward-listing MCP tool for discovery rather than
`list_reward_functions: true`, which exits the backend without training.

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
