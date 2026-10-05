# Group Relative Policy Optimization and variants

Use `train_mode: "grpo"` to sample completion groups, score registered rewards,
and optimize within-group advantages with clipped updates and a frozen
reference. Read [datasets.md](datasets.md) for prompt/answer rows,
[reward_functions.md](reward_functions.md) for callbacks, and
[config.md](config.md) for shared training settings.

## GRPO controls

| Field | MCP default | Meaning |
| --- | --- | --- |
| `group_size` | `4` | Completions per prompt |
| `max_completion_length` | `512` | Maximum generated tokens per completion |
| `temperature` | `1.0` | Sampling temperature |
| `beta` | `0.1` | Reference KL coefficient |
| `epsilon` | `0.0001` | Lower clipping bound and default upper bound |
| `epsilon_high` | `null` | Separate upper bound; null uses `epsilon` |
| `importance_sampling_level` | `"token"` | `"token"` or `"sequence"` |
| `grpo_loss_type` | `"grpo"` | `"grpo"`, `"bnpo"`, or `"dr_grpo"` |
| `reference_model_path` | `null` | Frozen reference; null uses the starting model |

The MCP defaults come from the training entrypoint, not necessarily the
standalone trainer dataclass. The current generation path uses `</answer>` as
the end-answer marker; no configurable marker is exposed by MCP.

## Variant mapping

| Requested variant | Config | Loss behavior |
| --- | --- | --- |
| Base GRPO | `grpo_loss_type: "grpo"` | Mean of per-completion mean token losses |
| BNPO | `grpo_loss_type: "bnpo"` | Sum token losses divided by all valid generated tokens |
| Dr. GRPO | `grpo_loss_type: "dr_grpo"` | Divide by completion count times `max_completion_length` |
| GSPO-style sampling | `importance_sampling_level: "sequence"` | Sequence-level importance ratios |
| DAPO-style dual clipping | Explicit `epsilon` and `epsilon_high` | Distinct lower/upper ratio bounds |

These controls can be combined when explicitly requested. Keep the mode name
`grpo` and preserve the requested variant rather than inventing `gspo`,
`dapo`, or `dr_grpo` mode names.

## Example

```json
{
  "model": "org/model",
  "data": "org/prompt-answer-dataset",
  "train": true,
  "train_type": "lora",
  "train_mode": "grpo",
  "group_size": 4,
  "grpo_loss_type": "grpo",
  "importance_sampling_level": "token",
  "reward_functions": "r1_accuracy_reward_func,r1_strict_format_reward_func",
  "reward_weights": "[0.8, 0.2]",
  "max_completion_length": 256,
  "temperature": 0.8,
  "iters": 100
}
```

Generation work scales with prompt batch, group size, and completion length.
Use [memory.md](memory.md) for available controls. `micro_batch_size` is not
exposed to GRPO through the entrypoint; QAT and `efficient_long_context` are
unsupported.

Inspect per-reward means/variance/coverage, KL, clipping, completion length,
and held-out task accuracy. Constant or identical group rewards supply little
preference signal. A high completion-cap hit rate may indicate truncated
answers or a mismatch with the expected stopping format.
