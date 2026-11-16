# GRPO variants

All variants use `train_mode: grpo`, GRPO prompt/answer data and registered
rewards. Put variant settings in `extra_config`; do not invent separate
train_mode values such as gspo or dapo. See [GRPO](grpo.md) for the full call.

| Variant | Extra settings | Implementation meaning |
| --- | --- | --- |
| GRPO | grpo_loss_type=grpo; importance_sampling_level=token | Normalize each completion by its valid-token count, then average completions |
| GSPO-style ratios | importance_sampling_level=sequence | Mean token log-ratio per completion before clipping |
| Dr. GRPO | grpo_loss_type=dr_grpo | Sample count × configured completion cap denominator |
| BNPO | grpo_loss_type=bnpo | Sum token losses and divide by the batch's total valid-token count |
| DAPO-style dual clipping | epsilon plus epsilon_high | Asymmetric clipping; not the entire DAPO training recipe |

## Best practices per variant

- GSPO-style: explicitly set sequence (token is the GRPO default); compare
  length, reward and KL on the same prompts. Do not equate a ratio option
  with reproducing every component of an external GSPO recipe.
- Dr. GRPO: hold max_completion_length fixed across comparisons because it
  enters the denominator. Inspect truncation and reward rather than assuming
  a change in normalized loss means better generations.
- BNPO: compare against GRPO with varied completion lengths: global token
  normalization weights long completions differently from a per-completion
  average. Keep length distributions and sampling settings comparable.
- DAPO-style: make the asymmetric upper/lower clips explicit and monitor KL
  and clipped ratios. A wider upper clip changes update behavior; it does
  not enable unrelated DAPO mechanisms.

## Memory efficiency

Variants retain GRPO's reference and grouped generation costs. Changing ratio
level or normalization is not a memory-saving switch. Reduce physical batch,
completion cap and unnecessary context; preserve group diversity and at least
two completions. Internal scoring microbatches do not eliminate rollout storage.
