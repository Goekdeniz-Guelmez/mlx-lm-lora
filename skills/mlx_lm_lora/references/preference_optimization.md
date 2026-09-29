# Offline preference optimization

Use this reference for preference datasets whose chosen and rejected responses
are already present in the dataset. The supported offline modes are:

| Method | `train_mode` | Reference model | Dataset shape |
| --- | --- | --- | --- |
| Direct Preference Optimization | `"dpo"` | Required; defaults to the base model frozen before training | `prompt`, `chosen`, `rejected` |
| Contrastive Preference Optimization | `"cpo"` | Not used | `prompt`, `chosen`, `rejected` |
| Odds Ratio Preference Optimization | `"orpo"` | Not used | `prompt`, `chosen`, `rejected`; optional `preference_score` |
| Final Token Preference Optimization | `"ftpo"` | Required; defaults to the base model frozen before training | Antidoom fields: `context_with_chat_template`, `rejected_decoded`, `multi_chosen_decoded` |

`online_dpo`, `xpo`, `rlhf_reinforce`, and `ppo` generate responses during
training and are therefore not offline trainers. They are documented in
[reinforcement_learning.md](reinforcement_learning.md).

## Common configuration

The examples below use MCP field names. Use `train: true`, preserve the model
and dataset identifiers exactly, and do not send CLI spellings such as
`--train-mode` in an MCP config.

All four modes inherit these general training settings:

| Field | Default | Use |
| --- | ---: | --- |
| `train_type` | `"lora"` | `"lora"`, `"dora"`, or `"full"` |
| `optimizer` | `"adam"` | `"adam"`, `"adamw"`, or `"muon"` |
| `optimizer_config` | optimizer-specific empty mapping | Extra optimizer keyword arguments |
| `learning_rate` | `1e-5` | Positive optimizer learning rate |
| `lr_schedule` | `null` | MLX-LM schedule expression |
| `batch_size` | `4` | Per-step batch size; keep it divisible by worker count |
| `iters` | `null` | Number of optimizer iterations |
| `epochs` | `null` | Converted to iterations when `iters` is omitted |
| `gradient_accumulation_steps` | `1` | Accumulate this many minibatches before updating |
| `max_seq_length` | `2048` | Truncation/padding limit for tokenized preference rows |
| `num_layers` | `-1` | Number of LoRA/DoRA layers; `-1` means all |
| `lora_parameters` | `rank: 8`, `dropout: 0.0`, `scale: 10.0` | LoRA/DoRA adapter settings |
| `grad_checkpoint` | `false` | Trade compute for lower activation memory |
| `efficient_long_context` | `false` | Use 512-token sequential cached chunks; supported by DPO, CPO, and ORPO, not FTPO |
| `val_batches` | `25` | Validation batches; `-1` means all |
| `steps_per_report` | `10` | Training-log interval |
| `steps_per_eval` | `200` | Validation interval |
| `save_every` | `100` | Adapter checkpoint interval |
| `resume_adapter_file` | `null` | Tenant-local adapter to resume from |
| `adapter_path` | server-selected | Tenant-local output directory |
| `seed` | `0` | Random seed |
| `wandb` | `null` | Optional Weights & Biases project name |
| `test` | `false` | Evaluate the test split after training |
| `test_batches` | `500` | Test batches; `-1` means all |
| `fuse` | `true` | Merge and save the trained adapter with the base model |

For quantized loading, set exactly one of `load_in_4bits`, `load_in_6bits`, or
`load_in_8bits` to `true`. QAT is available for SFT, DPO, and ORPO; its fields
are `qat_enable`, `qat_bits` (default `8`), `qat_group_size` (default `64`),
`qat_mode: "affine"`, `qat_start_step` (default `1`), and `qat_interval`
(default `1`). Do not add QAT fields to FTPO or CPO requests.

## Shared DPO/CPO loss settings

DPO and CPO use the same loss selector:

| Field | Default | Values/effect |
| --- | ---: | --- |
| `beta` | `0.1` | Preference-logit temperature/scale |
| `dpo_cpo_loss_type` | `"sigmoid"` | `"sigmoid"`, `"hinge"`, `"ipo"`, or `"dpop"` |
| `delta` | `50.0` | DPOP penalty/margin; relevant when the loss type is `"dpop"` |

The loss variants are:

- `sigmoid`: smooth logistic preference loss; the safest starting point.
- `hinge`: margin loss; `delta` is not used.
- `ipo`: squared target-gap loss and per-sequence mean log-probabilities; it is
  sensitive to `beta` and sequence-length normalization.
- `dpop`: sigmoid DPO with a penalty when the policy lowers the chosen
  response relative to the reference; uses `delta`.

### DPO

DPO compares policy and frozen-reference log-probability differences. A
reference model is loaded automatically from `model` when
`reference_model_path` is omitted.

```json
{
  "model": "Qwen/Qwen3.5-0.8B",
  "data": "org/preference-dataset",
  "train": true,
  "train_type": "lora",
  "train_mode": "dpo",
  "iters": 100,
  "batch_size": 2,
  "learning_rate": 1e-5,
  "beta": 0.1,
  "dpo_cpo_loss_type": "sigmoid",
  "reference_model_path": "Qwen/Qwen3.5-0.8B",
  "max_seq_length": 2048
}
```

`reference_model_path` may be a Hub model ID or a tenant-approved local
reference path. The reference is frozen and is not updated by the optimizer.

### CPO

CPO optimizes the chosen-versus-rejected policy scores directly and does not
use a reference model. It accepts the same `beta`, `dpo_cpo_loss_type`, and
`delta` fields as DPO.

```json
{
  "model": "Qwen/Qwen3.5-0.8B",
  "data": "org/preference-dataset",
  "train": true,
  "train_type": "lora",
  "train_mode": "cpo",
  "iters": 100,
  "beta": 0.1,
  "dpo_cpo_loss_type": "sigmoid",
  "gradient_accumulation_steps": 4
}
```

Leave `reference_model_path` unset for CPO. The current CPO implementation
does not apply QAT, although its inherited argument structure includes the
shared DPO fields.

### ORPO

ORPO combines chosen-response NLL with an odds-ratio preference term and does
not need a reference model. ORPO-specific fields are:

| Field | Default | Use |
| --- | ---: | --- |
| `beta` | `0.1` | Weight/temperature for the odds-ratio term |
| `reward_scaling` | `1.0` | Accepted for compatibility but currently unused by the trainer |

`preference_score` is an input data field, not a training config field. It is
optional and defaults to `1.0`; when present, the current ORPO dataset loader
converts it to a float and uses it to scale the chosen-response score.

```json
{
  "model": "Qwen/Qwen3.5-0.8B",
  "data": "org/preference-dataset",
  "train": true,
  "train_type": "lora",
  "train_mode": "orpo",
  "iters": 100,
  "beta": 0.1,
  "max_seq_length": 2048,
  "qat_enable": true,
  "qat_bits": 8,
  "qat_group_size": 64
}
```

ORPO accepts simple string responses, structured response objects, and
response message lists. The prompt is formatted with the model tokenizer's
chat template before scoring.

### FTPO / Antidoom

FTPO is a final-token preference objective for Antidoom-style data. It scores
the next-token distribution after `context_with_chat_template`; it is not a
general chosen/rejected sequence trainer.

| Field | Default | Use |
| --- | ---: | --- |
| `lambda_mse_target` | `0.05` | Weight for the target-token MSE term |
| `tau_mse_target` | `1.0` | Threshold for excess target-logit deviation |
| `lambda_mse` | `0.4` | Weight for non-target-logit MSE |
| `clip_epsilon_logits` | `2.0` | Positive preference-margin clipping scale |
| `reference_model_path` | `null` | Frozen reference; defaults to the base model |

Each row must contain:

```json
{
  "context_with_chat_template": "<tokenized conversation context>",
  "rejected_decoded": "token text",
  "multi_chosen_decoded": ["token text", "another token text"]
}
```

The loader keeps only one-token chosen and rejected surfaces, removes duplicate
chosen tokens, and skips rows that do not fit `max_seq_length`. At least one
usable chosen token must remain per row. The dataset should be prepared with
the same tokenizer/chat template as the model.

```json
{
  "model": "org/base-model",
  "data": "org/antidoom-dataset",
  "train": true,
  "train_type": "lora",
  "train_mode": "ftpo",
  "iters": 100,
  "lambda_mse_target": 0.05,
  "tau_mse_target": 1.0,
  "lambda_mse": 0.4,
  "clip_epsilon_logits": 2.0,
  "reference_model_path": "org/base-model"
}
```

## Dataset requirements

For DPO and CPO, use rows with `prompt`, `chosen`, and `rejected` fields. A
`system` field is optional. The chosen and rejected responses should differ in
quality, share the same prompt, and be formatted with the same tokenizer chat
template.

```json
{
  "prompt": "Explain why the sky is blue.",
  "chosen": "A concise, scientifically correct explanation.",
  "rejected": "Because the sky is painted blue."
}
```

An SFT-only `text`, `messages`, or `prompt`/`completion` dataset is not enough
for DPO or CPO. For ORPO, use the same pair fields and optionally add a numeric
`preference_score`. For FTPO, use only the Antidoom schema described above.

## Best practices by algorithm

### DPO

- Start with `dpo_cpo_loss_type: "sigmoid"`, `beta: 0.1`, and a small learning
  rate. Tune `beta` only after checking preference accuracy and reward margin.
- Keep the reference model equal to the exact pre-training checkpoint unless
  a deliberate reference policy is part of the experiment.
- Apply the same chat template and prompt boundary to both responses. Template
  mismatches create artificial preference margins.
- Filter ties, malformed responses, empty completions, and pairs where the
  rejected answer is actually better. Monitor `accuracies`, `margins`, and
  validation loss together.
- Use `grad_checkpoint`, lower `batch_size`, or gradient accumulation for
  long pairs. Use `efficient_long_context` for supported long-context runs.

### CPO

- Prefer CPO when a separate reference model is undesirable or when direct
  contrastive policy scoring is the intended objective.
- Because there is no reference KL anchor, use conservative learning rates and
  watch for capability drift on a held-out general-instruction set.
- Begin with the sigmoid loss. Use IPO only when sequence-length normalization
  and the corresponding `beta` have been deliberately tuned.
- Keep batches large enough to contain varied pair difficulty; accumulation is
  often safer than increasing the per-device batch size.

### ORPO

- Use clean, same-prompt pairs and inspect the optional `preference_score`
  distribution before training. Scores with inconsistent scales can dominate
  the chosen-response term.
- Start with `beta: 0.1` and compare against an SFT baseline because ORPO also
  retains a chosen-response likelihood objective.
- Do not assume `reward_scaling` changes training in the current backend; it is
  accepted but not implemented in the active ORPO loss path.
- Use QAT only when the deployment quantization target is known. First establish
  a non-QAT baseline, then compare validation preference metrics and generation
  quality after quantization.

### FTPO

- Validate tokenizer alignment and count usable rows after Antidoom filtering;
  a large raw dataset can become small if chosen surfaces are not single
  tokens.
- Keep `clip_epsilon_logits` positive. Tune the MSE weights only with a held-out
  loop-repair set; excessive MSE weighting can suppress useful policy movement.
- Use a frozen reference from the same base checkpoint when the goal is a
  targeted repair. Evaluate repetition-loop rate and general capability, not
  just FTPO loss.
- Do not enable QAT or treat `max_seq_length` as a way to truncate FTPO context:
  rows whose context exceeds the limit are skipped by the loader.

## Operational checklist

Before starting an offline preference job:

1. Confirm `data` is a Hugging Face dataset ID for MCP, not a local JSONL path.
2. Confirm the first dataset row has the fields required by the selected mode.
3. Choose `iters` or `epochs`; if both are supplied, `iters` wins.
4. Keep `batch_size` at least as large as the distributed worker count and
   divisible by it.
5. Set `reference_model_path` only for DPO or FTPO unless the request
   explicitly needs a compatible shared config.
6. Validate on held-out preference data and on a general capability set before
   fusing the adapter.
