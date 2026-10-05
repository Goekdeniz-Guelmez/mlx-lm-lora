# Directional and Similarity-aware Latent Alignment

Use `train_mode: "dsla"` to add hidden-state alignment to a DPO, ORPO, or CPO
preference objective. Shared training settings are in [config.md](config.md),
and pair records in [datasets.md](datasets.md).

## Objective and latent controls

| Field | Default | Meaning / values |
| --- | --- | --- |
| `dsla_loss` | `"dpo"` | `"dpo"`, `"orpo"`, or `"cpo"` |
| `latent_weight` | `0.1` | Finite nonnegative latent regularizer weight |
| `latent_margin` | `0.05` | Finite nonnegative cosine margin |
| `latent_gamma` | `10.0` | Finite positive soft-margin sharpness |
| `latent_variant` | `"both"` | `"similarity"`, `"direction"`, or `"both"` |
| `latent_pooling` | `"answer_mean"` | `"answer_mean"`, `"last_token"`, `"last_k_mean"`, or `"prompt_answer_mean"` |
| `latent_layer` | `"final"` | `"final"`, `"middle"`, `"late"`, or a zero-based layer index |

DSLA-DPO loads a frozen reference from `reference_model_path` or the starting
`model`. DSLA-ORPO and DSLA-CPO are reference-free. `beta` defaults to `0.1`;
DPO/CPO variants also use `dpo_cpo_loss_type` and `delta` as described in
[dpo.md](dpo.md) and [cpo.md](cpo.md). The DSLA-ORPO path uses the DSLA pair
loader; standalone ORPO's `preference_score` does not apply.

## Example

```json
{
  "model": "org/model",
  "data": "org/preference-dataset",
  "train": true,
  "train_type": "lora",
  "train_mode": "dsla",
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

## Compatibility and results

DSLA requires `max_seq_length >= 2`, a supported residual-state backbone, and
an exact generation-prompt token prefix for both responses. Model depth and
hidden-state capture are verified when the model loads, not by MCP validation.
A layer index must lie within the loaded model's depth.

`efficient_long_context` is rejected because latent pooling needs the complete
sequence. Use checkpointing, accumulation, and recurrent chunks from
[memory.md](memory.md). QAT is supported; DSLA's projection schedule differs
from the other QAT paths as documented in [quantization.md](quantization.md).

Track preference loss/margins and latent similarity/direction metrics
separately. Compare with the same preference objective without latent
regularization before attributing a quality change to alignment.
