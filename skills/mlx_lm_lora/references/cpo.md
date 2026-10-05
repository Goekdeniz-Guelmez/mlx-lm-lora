# Contrastive Preference Optimization

Use `train_mode: "cpo"` to contrast chosen and rejected policy scores without
loading a reference model. Read [datasets.md](datasets.md) for offline pairs
and [config.md](config.md) for shared training options.

CPO accepts `beta` (default `0.1`), `dpo_cpo_loss_type` (default `"sigmoid"`),
and `delta` (default `50.0`). Loss choices are `"sigmoid"`, `"hinge"`, `"ipo"`,
and `"dpop"`; the shared selector descriptions are in [dpo.md](dpo.md).

CPO's DPOP variant uses a policy-only penalty when the rejected score exceeds
the chosen score. It cannot apply DPO's reference-relative penalty because
there is no reference. IPO uses mean token scores; other variants use summed
scores. Leave `reference_model_path` unset for CPO.

## Example

```json
{
  "model": "org/model",
  "data": "org/preference-dataset",
  "train": true,
  "train_type": "lora",
  "train_mode": "cpo",
  "dpo_cpo_loss_type": "sigmoid",
  "beta": 0.1,
  "iters": 100,
  "gradient_accumulation_steps": 4
}
```

Cached `efficient_long_context` is supported; see [memory.md](memory.md).
Quantized model loading is supported, but QAT is not enabled in the CPO
entrypoint. Read [quantization.md](quantization.md) for the distinction.

Compare preference accuracy and margins with held-out general behavior.
Without a frozen reference anchor, increasing the update budget can move the
policy beyond the desired preference repair even when training loss falls.
