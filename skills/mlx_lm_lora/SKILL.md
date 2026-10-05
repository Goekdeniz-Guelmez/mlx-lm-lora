---
name: mlx-lm-lora
description: Translate MLX-LM-LoRA fine-tuning requests into validated tenant-scoped MCP jobs. Use for supervised, preference, or reward training with LoRA, DoRA, full fine-tuning, quantized loading, or QAT, and for inspecting those training jobs.
---

# MLX-LM-LoRA training

Use the `mlx-lm-lora` MCP server when available. It validates configurations,
queues training, and owns tenant-scoped logs and artifacts. Preserve the user's
model, dataset, algorithm, and training budget.

## Load the relevant guidance

Read only the guide for the requested mode. If the user has not chosen a mode,
use the appropriate overview to distinguish the data and feedback requirements.

| Request | Reference |
| --- | --- |
| Supervised fine-tuning; NLL, chunked NLL, or DFT | [sft.md](references/sft.md) |
| Choose an offline preference method | [preference_optimization.md](references/preference_optimization.md) |
| DPO | [dpo.md](references/dpo.md) |
| CPO | [cpo.md](references/cpo.md) |
| ORPO | [orpo.md](references/orpo.md) |
| DSLA | [dsla.md](references/dsla.md) |
| FTPO / Antidoom | [ftpo.md](references/ftpo.md) |
| Choose a generated-response training method | [reinforcement_learning.md](references/reinforcement_learning.md) |
| GRPO, GSPO, BNPO, Dr. GRPO, or DAPO-style settings | [grpo.md](references/grpo.md) |
| KLPO | [klpo.md](references/klpo.md) |
| Online DPO | [online_dpo.md](references/online_dpo.md) |
| XPO | [xpo.md](references/xpo.md) |
| PPO | [ppo.md](references/ppo.md) |
| RLHF REINFORCE | [rlhf_reinforce.md](references/rlhf_reinforce.md) |

For shared options, use [config.md](references/config.md). Read
[datasets.md](references/datasets.md) for input schemas,
[quantization.md](references/quantization.md) for quantized loading or QAT,
[memory.md](references/memory.md) for memory controls, and
[multi-tenant.md](references/multi-tenant.md) for tenant or local-path handling.
Algorithm guides link to shared reward and judge details where needed.

## Translate and run the request

1. Call `mlx_lm_lora_get_capabilities` on first use. Its config keys, enum
   choices, and feature support describe the connected server version. Report
   a missing requested feature instead of substituting another algorithm.
2. Resolve the pinned or authenticated tenant, or use the user's explicit
   tenant. Never silently change tenants.
3. Build a JSON `config` with `model`, `data`, `train: true`, and the requested
   options. Use underscore field names, not CLI flags. `data` must be a Hugging
   Face dataset repository ID; do not convert it to a URL or local path.
4. Call `mlx_lm_lora_validate_training_config`. Correct mapping errors; ask
   only for missing information that changes the intended run. For a dry run
   or validation-only request, return the validated config without starting.
5. For an authorized training request, call `mlx_lm_lora_start_training` after
   validation. A complete training request needs no extra confirmation.
6. Track the returned `job_id` with `mlx_lm_lora_get_training_status`. Read
   `mlx_lm_lora_get_training_log` for progress or failure details. Report the
   tenant, job ID, status, and artifact path; distinguish queued/running jobs
   from completed training.

Use `mlx_lm_lora_list_training_runs` to recover a missing job ID. Use
`mlx_lm_lora_cancel_training` when asked to cancel a queued job; running jobs
cannot be cancelled through this tool. Failed-job status and logs guide a
revised config; do not silently submit duplicate jobs or change the budget.
