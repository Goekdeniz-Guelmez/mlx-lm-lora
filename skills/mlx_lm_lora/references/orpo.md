# Odds Ratio Preference Optimization

Use `train_mode: "orpo"` to combine chosen-response NLL with an odds-ratio
preference term. ORPO does not load a frozen reference. Read
[datasets.md](datasets.md) for basic pairs and [config.md](config.md) for
shared options.

## ORPO-specific behavior

`beta` defaults to `0.1` and weights the preference term. `reward_scaling`
defaults to `1.0` and is accepted for compatibility, but is unused by the
current trainer. `dpo_cpo_loss_type` and `delta` do not select ORPO's loss.
Leave `reference_model_path` unset.

The ORPO loader handles plain response strings, structured response objects,
and response message lists. An optional numeric `preference_score` defaults
to `1.0` and scales the chosen-response score. It is a dataset field, not an
MCP config key. Check its scale when comparing runs.

```json
{
  "prompt": "Why is the sky blue?",
  "chosen": "Air molecules scatter shorter visible wavelengths more strongly.",
  "rejected": "The sky is painted blue.",
  "preference_score": 1.0
}
```

## Example

```json
{
  "model": "org/model",
  "data": "org/preference-dataset",
  "train": true,
  "train_type": "lora",
  "train_mode": "orpo",
  "beta": 0.1,
  "iters": 100
}
```

ORPO supports QAT and cached `efficient_long_context`; see
[quantization.md](quantization.md) and [memory.md](memory.md).

Compare chosen-response modeling and preference margins, then inspect held-out
generations. Consistent same-prompt pairs and calibrated `preference_score`
values matter more than treating the unused `reward_scaling` as a tuning knob.
