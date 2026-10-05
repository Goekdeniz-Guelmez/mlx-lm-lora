# Direct Preference Optimization

Use `train_mode: "dpo"` for offline chosen/rejected preferences relative to a
frozen reference. Read [datasets.md](datasets.md) for pair records and
[config.md](config.md) for shared options.

The backend freezes a copy of `model` as the reference unless
`reference_model_path` specifies another Hub model or approved local model.
Keep the reference at the intended starting checkpoint when comparing runs.

## Preference loss controls

| Field | Default | Meaning |
| --- | --- | --- |
| `beta` | `0.1` | Preference-logit scale |
| `dpo_cpo_loss_type` | `"sigmoid"` | `"sigmoid"`, `"hinge"`, `"ipo"`, or `"dpop"` |
| `delta` | `50.0` | DPOP penalty coefficient |
| `reference_model_path` | `null` | Frozen reference; null uses the starting model |

Sigmoid uses a logistic preference loss; hinge uses a margin objective. IPO
uses a squared target gap and per-sequence mean token scores. DPOP penalizes
lower chosen-response scores relative to the reference. `delta` affects DPOP,
not sigmoid, hinge, or IPO.

These selector names also appear in CPO, DSLA-DPO/CPO, and online preference
trainers. Their reference and scoring behavior follows the respective guide;
matching selector names do not make the methods interchangeable.

## Example

```json
{
  "model": "org/model",
  "data": "org/preference-dataset",
  "train": true,
  "train_type": "lora",
  "train_mode": "dpo",
  "dpo_cpo_loss_type": "sigmoid",
  "beta": 0.1,
  "iters": 100
}
```

DPO supports QAT and cached long-context processing. See
[quantization.md](quantization.md) and [memory.md](memory.md).

Inspect validation preference accuracy, chosen/rejected reward margins, and
held-out general capabilities together. Ties, inverted labels, empty answers,
and inconsistent chat templates can create misleading preference scores.
