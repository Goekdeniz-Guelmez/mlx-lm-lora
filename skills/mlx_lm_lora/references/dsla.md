# DSLA latent alignment

Use `train_mode: "dsla"` for Directional and Similarity-aware Latent Alignment.
It combines an offline preference objective with residual-stream alignment.
Use a Hugging Face dataset with `prompt`, `chosen`, and `rejected` fields.

| Field | Default | Values |
| --- | --- | --- |
| `dsla_loss` | `"dpo"` | `"dpo"`, `"orpo"`, `"cpo"` |
| `latent_weight` | `0.1` | Finite nonnegative regularizer weight |
| `latent_margin` | `0.05` | Finite nonnegative cosine margin |
| `latent_gamma` | `10.0` | Finite positive soft-margin sharpness |
| `latent_variant` | `"both"` | `"similarity"`, `"direction"`, `"both"` |
| `latent_pooling` | `"answer_mean"` | `"answer_mean"`, `"last_token"`, `"last_k_mean"`, `"prompt_answer_mean"` |
| `latent_layer` | `"final"` | `"final"`, `"middle"`, `"late"`, or a zero-based layer index |

DSLA-DPO loads a frozen reference from `reference_model_path` or the original
`model`. DSLA-ORPO and DSLA-CPO do not use a reference. The shared `beta`,
`dpo_cpo_loss_type`, and `delta` fields select the preference behavior; see
[preference_optimization.md](preference_optimization.md).

```json
{
  "model": "org/model",
  "data": "org/preference-dataset",
  "train": true,
  "train_mode": "dsla",
  "train_type": "lora",
  "dsla_loss": "dpo",
  "latent_weight": 0.1,
  "latent_variant": "both",
  "latent_pooling": "answer_mean",
  "latent_layer": "final",
  "grad_checkpoint": true,
  "recurrence_chunk_size": 64,
  "iters": 100
}
```

DSLA requires `max_seq_length >= 2` and a model backbone whose residual hidden
states can be captured. A numeric layer index is checked against model depth
when the model loads; config validation alone cannot verify architecture.
`efficient_long_context` is unsupported because alignment needs pooled
representations from the complete sequence. Use checkpointing, accumulation,
and recurrent chunk controls instead.

QAT is supported. DSLA uses `qat_bits`, `qat_group_size`, and `qat_start_step`
for policy projections; `qat_interval` and `qat_mode` are accepted shared
options but do not alter the DSLA projection schedule.
