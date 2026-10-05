# Exploratory Preference Optimization

Use `train_mode: "xpo"` for online DPO with an exploration term. It generates
candidate pairs, uses a pairwise model judge, and loads a frozen reference.
Read [online_dpo.md](online_dpo.md) for shared preference controls,
[judges.md](judges.md) for the judge contract, and [datasets.md](datasets.md)
for prompt records.

## Exploration schedule

`alpha` must be a nonempty JSON list of exploration weights. The MCP parser
default is `[0.00001]`. One value stays constant. With multiple values, the
current trainer divides the iteration budget into schedule segments and
selects the segment's value, capped at the final entry. This is an iteration
schedule, despite the older argument description calling it an epoch schedule.

Keep the list no longer than `iters`: `iters // len(alpha)` must be nonzero.
Choose nonnegative numeric weights when requesting an exploration bonus; this
list-length constraint is not checked by MCP config validation.

XPO also consumes `beta`, `dpo_cpo_loss_type`, `delta`, `judge`, `judge_config`,
`reference_model_path`, and `max_completion_length`. The entrypoint does not
pass a configurable `temperature` to XPO generation, so that field does not
control this mode's training sampler.

## Example

```json
{
  "model": "org/model",
  "data": "org/prompt-dataset",
  "train": true,
  "train_type": "lora",
  "train_mode": "xpo",
  "judge": "org/judge-model",
  "alpha": [0.00001, 0.000005],
  "beta": 0.1,
  "dpo_cpo_loss_type": "sigmoid",
  "max_completion_length": 256,
  "micro_batch_size": 1,
  "iters": 100
}
```

Use [config.md](config.md) for shared settings and [memory.md](memory.md) for
microbatches and recurrent chunks. QAT, `efficient_long_context`, and GRPO
reward/group controls are not used.

The current `test: true` XPO path forwards the complete `alpha` list to an
evaluation loss expecting a scalar, so post-training evaluation can fail.
If the user requests this evaluation, explain the entrypoint limitation; do
not silently remove `test` or convert the training schedule to a scalar.

Compare against online DPO under the same prompt pool and judge. Inspect the
exploration bonus, KL, and held-out task behavior; increasing `alpha` can
encourage movement that does not improve the task.
