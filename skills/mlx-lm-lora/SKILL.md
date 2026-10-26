---
name: mlx-lm-lora
description: Translate MLX-LM-LoRA fine-tuning requests into validated MCP runs, including SFT, offline preference training, DSLA, GRPO, KLPO, and online judge modes. Use for starting, monitoring, stopping, or configuring training through this server.
---

# MLX-LM-LoRA training

Use the connected MLX-LM-LoRA MCP server for training. Discover its live tool
schemas first; the server must expose `inspect_dataset` and `validate_model_support`.
This server manages local run state, logs, configs and adapters. It has no
tenant system. Do not assume another MCP server's contract matches this one.

## Translate and execute

1. Discover tools and call `list_training_modes`. Preserve the user's model,
   dataset, method and budget. Read the relevant algorithm reference below.
2. Build **flat tool arguments**, not a nested `config`. The flat keys are
   `model`, `data`, `train_mode`, `train_type`, `iters`, `batch_size`,
   `learning_rate`, `num_layers`, `max_seq_length`, `quantize_bits`, `fuse`,
   `extra_config`; `start_training` additionally accepts `name`.
3. Put backend-only settings such as `beta`, `judge`, `grad_checkpoint`,
   `gradient_accumulation_steps`, `lora_parameters` and reward settings in
   `extra_config`, encoded as a JSON **string**. Do not invent backend keys.
4. Call `validate_model_support` for the effective model, then call
   `inspect_dataset` with the effective data, model, train_mode and extra_config
   before training. Run these calls sequentially, never concurrently: both
   initialize Transformers' lazy imports on a cold MCP server.
   Treat either tool error or missing tool as blocking: report it and stop.
   Do not bypass these checks with ad hoc scripts, source code scraping, or
   git clones. Repeat the checks whenever their inputs change.
   Call `check_training_config` with those arguments, excluding `name`.
   Its text must say `Config looks good.`; `is_error=false` alone is not
   validation success. Explain warnings. Validation checks local structure,
   not Hub access, dataset schemas, every algorithm constraint or GPU fit.
5. For an authorized training request, call `start_training`. On success,
   record `structured_content.id`. Poll `training_status` with that exact
   `run_id`; use `training_log` for details and `list_training_runs` to reconnect.
6. Report run ID, terminal status, exit code, relevant metrics, adapter and
   log paths. Check saved artifacts and logs before calling a run successful;
   exit code 0 alone can conceal a backend early return. Use `stop_training`
   when the user asks to stop a run; stopping the server is separate.

Use supplied settings. If the user asks only for a tiny smoke test, choose
and state a small step budget, batch/context and trainable-layer count.
Do not start training for a configuration-only or dry-run request. Do not
restart a stopped server or launch extra experiments merely to inspect docs.

## Important mappings and limits

- `train_mode: "sft"` supports both causal language-model pretraining on text
  (including continued/domain pretraining) and supervised fine-tuning. The
  SFT name does not restrict this mode to instruction tuning. Map pretraining
  requests to `sft` with text rows and `sft_loss_type: "nll"`; do not reject
  pretraining because the mode or tool description says fine-tuning. Preserve
  the requested `train_type`: use `full` when all weights should be trained,
  and adapters when requested. This training API loads an existing model;
  training from randomly initialized weights is a separate capability and
  must not be implied by support for continued pretraining.

- `train_type`: `lora`, `dora`, `full`. LoRA is the usual memory-conscious
  starting point; full training has much larger gradient/optimizer storage.
- `data`: preserve a Hub dataset ID, or accept a server-local folder with
  `train.jsonl` and optional `valid.jsonl`, `test.jsonl`. Match the mode's row schema.
- Map explicit 4/6/8-bit loading requests to flat `quantize_bits: 4/6/8`.
  Use `0` for a model already quantized; avoid quantizing it twice.
- `train: true` and `adapter_path` are server-managed. Do not set `config`
  or `adapter_path` in `extra_config`; the server ignores them.
- Backend-only options can override flat options through `extra_config`;
  keep each setting in one place to avoid contradictory configs.
- For an epoch budget, call `calculate_iters` after `inspect_dataset`, passing
  `num_samples` from `rows.train`, the effective `batch_size` (including any
  extra_config override), and the requested positive integer `epochs`.
  Pass its returned `iters` to both `check_training_config` and `start_training`.
  The formula is `epochs * ceil(num_samples / batch_size)`, matching the backend.
  Recalculate if dataset settings, batch size or epochs change. These are
  training iterations, not necessarily optimizer updates with accumulation.
  Do not put only `epochs` in extra_config: the default `iters: 100` overrides it.
  If the calculation tool is missing, report that limitation; never silently
  substitute 100 steps for requested epochs.
- One active run is permitted per shared runs directory. HTTP authentication
  is optional bearer authentication; it does not provide tenant isolation.
- Dataset and model examples in references use illustrative Hub IDs. Inspect
  real access/schema and model compatibility before starting them.

## References

- [MCP contract and defaults](references/config.md)
- [Local paths and server lifecycle](references/multi-tenant.md)
- [Memory efficiency and support matrix](references/memory-efficiency.md)
- [Quantized loading and QAT](references/quantization.md)
- [SFT, NLL, chunked NLL and DFT](references/sft.md)
- [Offline preference overview](references/preference_optimization.md):
  [DPO](references/dpo.md), [CPO](references/cpo.md),
  [ORPO](references/orpo.md), [FTPO](references/ftpo.md), [DSLA](references/dsla.md)
- [Reward/online overview](references/reinforcement_learning.md):
  [GRPO](references/grpo.md), [GRPO variants](references/grpo_variants.md),
  [KLPO](references/klpo.md), [Online DPO](references/online_dpo.md),
  [XPO](references/xpo.md), [PPO](references/ppo.md),
  [RLHF REINFORCE](references/rlhf_reinforce.md)

Algorithm settings are checked against `train.py`, its dataset dispatch and
trainer implementations at PR #104 commit `a16c417`. Discover live schemas when versions differ; do not scrape source or clone repositories. All modes are documented; only SFT has been
verified with a real GPU run in this session. Schema checks are not training tests.
