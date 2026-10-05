---
name: mlx-lm-lora
description: Translate MLX-LM-LoRA fine-tuning requests into validated tenant-scoped MCP jobs, including SFT, DSLA, offline preferences, GRPO, KLPO, online judge training, quantized loading, and QAT.
---

# MLX-LM-LoRA training

Use the `mlx-lm-lora` MCP server for training operations. The MCP server owns
tenant isolation, job state, logs, and artifact paths. Do not run the direct
training CLI when the MCP server is available.

## Translate the request

Extract the user's values and send them as the MCP tool's `config` object.
Preserve model and dataset identifiers exactly as written.

- `lora`, `dora`, or `full` maps to `train_type`.
- `sft`, `dpo`, `dsla`, `ftpo`, `cpo`, `orpo`, `grpo`, `klpo`,
  `online_dpo`, `xpo`, `rlhf_reinforce`, and `ppo` map to `train_mode`.
  GSPO, Dr. GRPO, BNPO, and DAPO use GRPO settings, not separate mode names.
- A number of steps maps to `iters`; a number of epochs maps to `epochs`.
- A maximum context length maps to `max_seq_length`.
- `4bit`, `6bit`, `8bit`, or `MXFP4` maps to `load_in_4bits`, `load_in_6bits`,
  `load_in_8bits`, or `load_in_mxfp4`. Set at most one. If the model ID already
  clearly names a pre-quantized model, do not add a second quantization flag.
- QAT maps to `qat_enable` and the requested `qat_*` fields; supported modes
  are SFT, DPO, ORPO, and DSLA. It is separate from quantized model loading.
- Scoring microbatches map to `micro_batch_size` for online DPO, XPO,
  RLHF REINFORCE, and PPO. Recurrent chunk size maps to `recurrence_chunk_size`.
- `data` must be a Hugging Face dataset repository ID such as
  `mlx-community/wikisql`; never turn it into a local JSONL path or URL.
  `tenant://...` is reserved for supported auxiliary files such as reward
  functions or a resume adapter.
- Include `model`, `data`, `train: true`, and the fields requested by the user.
  Do not invent unsupported config keys.

Read the mode-specific reference when needed:

- SFT: [references/sft.md](references/sft.md)
- Offline preference optimization: [references/preference_optimization.md](references/preference_optimization.md)
- Reinforcement learning and reward functions: [references/reinforcement_learning.md](references/reinforcement_learning.md)
- DSLA latent alignment: [references/dsla.md](references/dsla.md)
- KLPO routes and KL estimators: [references/klpo.md](references/klpo.md)
- Microbatch and recurrent memory controls: [references/memory.md](references/memory.md)
- Quantization: [references/quantization.md](references/quantization.md)
- Tenant selection and local paths: [references/multi-tenant.md](references/multi-tenant.md)
- Full field mapping: [references/config.md](references/config.md)

For example:

> Train Qwen/Qwen3.5-0.8B on LoRA and 4bit using SFT with
> mlx-community/wikisql. Use 1 step and max context length 512.

becomes:

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

## MCP workflow

1. Call `mlx_lm_lora_get_capabilities` when the server is first used. Use its
   `training_config_keys`, `config_choices`, and `features` as the authority
   for this server version. If a requested feature is absent, report the
   version mismatch instead of substituting a different algorithm.
2. Resolve the tenant from the pinned server tenant or the user's explicit
   tenant. Never silently switch tenants.
3. For GRPO/KLPO reward selection, use `mlx_lm_lora_list_reward_functions`;
   do not submit `list_reward_functions: true` as a training job.
   Call `mlx_lm_lora_validate_training_config` with the extracted config.
4. If validation succeeds, call `mlx_lm_lora_start_training`, unless the user
   asked for a dry run.
5. Poll `mlx_lm_lora_get_training_status`; use
   `mlx_lm_lora_get_training_log` when progress or an error needs explaining.
6. Report the job ID, tenant, status, artifact path, and any failure details.

Use `mlx_lm_lora_list_training_runs` to recover a job when its ID is unknown.
`mlx_lm_lora_cancel_training` only cancels queued jobs; it cannot stop training
once the worker has started. When a user asks only to validate or inspect a
config, return validation without enqueuing a job.

If a required value is missing, ask only for that value. A request that names
the model, dataset, method, quantization, and step/epoch budget is complete and
should be started after validation without an extra confirmation.
