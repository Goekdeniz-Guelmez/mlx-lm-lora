# Reinforcement learning and reward training

Use this reference for trainers that generate responses during optimization.
The supported reinforcement modes are:

| Method | `train_mode` | Feedback | Dataset shape |
| --- | --- | --- | --- |
| Group Relative Policy Optimization | `"grpo"` | Registered reward functions | `prompt`, `answer`; optional `system`, `type` |
| GSPO-style sequence importance sampling | `"grpo"` | Registered reward functions | Same as GRPO; set `importance_sampling_level: "sequence"` |
| Dr. GRPO | `"grpo"` | Registered reward functions | Same as GRPO; set `grpo_loss_type: "dr_grpo"` |
| BNPO | `"grpo"` | Registered reward functions | Same as GRPO; set `grpo_loss_type: "bnpo"` |
| DAPO-style dual clipping | `"grpo"` | Registered reward functions | Same as GRPO; set `epsilon_high` |
| RLHF REINFORCE with KL | `"rlhf_reinforce"` | Reward-model judge | Prompt-only data: `prompt` |
| Proximal Policy Optimization | `"ppo"` | Pairwise judge/reward model | Prompt-only data: `prompt` |

The repository also supports the online preference modes `"online_dpo"` and
`"xpo"`; their settings are included at the end of this reference because
they generate completions and require a judge.

## Common configuration

The examples use MCP field names. Always include `train: true`, keep `data` as
a Hugging Face dataset repository ID, and do not send CLI spellings such as
`--train-mode` in an MCP config.

These settings apply to the reinforcement modes unless a section below says
otherwise:

| Field | Default | Use |
| --- | ---: | --- |
| `train_type` | `"lora"` | `"lora"`, `"dora"`, or `"full"` |
| `optimizer` | `"adam"` | `"adam"`, `"adamw"`, or `"muon"` |
| `optimizer_config` | optimizer-specific empty mapping | Extra optimizer keyword arguments |
| `learning_rate` | `1e-5` | Positive optimizer learning rate |
| `lr_schedule` | `null` | MLX-LM schedule expression |
| `batch_size` | `4` | Prompt batch size; keep it divisible by worker count |
| `iters` | `null` | Number of optimizer iterations |
| `epochs` | `null` | Converted to iterations when `iters` is omitted |
| `gradient_accumulation_steps` | `1` | Accumulate minibatches before updating |
| `max_seq_length` | `2048` | Prompt/context token limit |
| `num_layers` | `-1` | Number of LoRA/DoRA layers; `-1` means all |
| `lora_parameters` | `rank: 8`, `dropout: 0.0`, `scale: 10.0` | LoRA/DoRA adapter settings |
| `grad_checkpoint` | `false` | Trade compute for lower activation memory |
| `val_batches` | `25` | Validation batches; `-1` means all |
| `steps_per_report` | `10` | Training-log interval |
| `steps_per_eval` | `200` | Validation interval |
| `save_every` | `100` | Adapter checkpoint interval |
| `resume_adapter_file` | `null` | Tenant-local adapter to resume from |
| `adapter_path` | server-selected | Tenant-local output directory |
| `reference_model_path` | `null` | Frozen reference; defaults to the base model for modes that use one |
| `seed` | `0` | Random seed |
| `wandb` | `null` | Optional Weights & Biases project name |
| `test` | `false` | Evaluate the test split after training |
| `test_batches` | `500` | Test batches; `-1` means all |
| `fuse` | `true` | Merge and save the trained adapter with the base model |

`load_in_4bits`, `load_in_6bits`, and `load_in_8bits` are independent model
loading choices; set at most one. QAT and `efficient_long_context` are not
wired into the GRPO or online-RL dispatch paths, so do not add those fields to
reinforcement requests.

## GRPO

GRPO generates `group_size` completions for every prompt, scores them with one
or more reward functions, normalizes rewards within each prompt group, and
updates the policy with a clipped objective.

```json
{
  "model": "Qwen/Qwen3.5-0.8B",
  "data": "org/prompt-answer-dataset",
  "train": true,
  "train_type": "lora",
  "train_mode": "grpo",
  "iters": 100,
  "group_size": 4,
  "max_completion_length": 512,
  "temperature": 0.8,
  "beta": 0.1,
  "epsilon": 0.0001,
  "reward_functions": "r1_accuracy_reward_func,r1_strict_format_reward_func",
  "reward_weights": "[0.8, 0.2]"
}
```

GRPO-specific settings are:

| Field | Default | Use |
| --- | ---: | --- |
| `group_size` | `4` | Number of sampled completions per prompt |
| `max_completion_length` | `512` | Maximum generated completion tokens |
| `temperature` | `0.8` | Sampling temperature |
| `beta` | `0.1` | KL penalty coefficient |
| `epsilon` | `1e-4` | Lower clipping bound |
| `epsilon_high` | `null` | Upper clipping bound; falls back to `epsilon` |
| `reference_model_path` | `null` | Frozen reference; defaults to the base model |
| `grpo_loss_type` | `"grpo"` | `"grpo"`, `"bnpo"`, or `"dr_grpo"` |
| `importance_sampling_level` | `"token"` | `"token"` or `"sequence"` |
| `reward_functions` | built-in defaults | Comma-separated registered reward names |
| `reward_functions_file` | `null` | Tenant-local Python file registering reward functions |
| `reward_weights` | equal weights | JSON-encoded list matching the function count |

The current default reward registry contains:

- `r1_accuracy_reward_func`
- `r1_int_reward_func`
- `r1_strict_format_reward_func`
- `r1_soft_format_reward_func`
- `r1_count_xml`

If `reward_functions` is omitted, all default functions are used. If a custom
file is supplied, it must register the names later passed in
`reward_functions`. A custom reward receives `prompts`, `completions`,
`answer`, and optional `types` lists and must return one numeric score or
`None` per completion. At least one reward must be valid for every completion.

`reward_weights` is shown as a string because the current CLI-to-trainer
adapter parses the value with `strip("[]")`. Use the same number of weights as
reward functions and make the scale intentional.

GRPO input rows use the `answer` field as reward-function reference data:

```json
{
  "prompt": "What is 2 + 2?",
  "answer": "4",
  "system": "Solve the problem and put the final result in <answer> tags.",
  "type": "arithmetic"
}
```

When no `system` field is supplied, the loader uses a default reasoning prompt
that asks for `<think>` and `<answer>` sections. The built-in reward functions
are designed around this XML-style format. There is no separate MCP setting
for the end-answer token; the current training path uses `</answer>`.

### GRPO loss variants

All variants still use `train_mode: "grpo"`:

| Variant | Configuration | Current implementation behavior |
| --- | --- | --- |
| Base GRPO | `grpo_loss_type: "grpo"` | Normalize by valid generated-token count |
| BNPO | `grpo_loss_type: "bnpo"` | Uses the same valid-token denominator in the current backend |
| Dr. GRPO | `grpo_loss_type: "dr_grpo"` | Normalize by generated-sample count × configured `max_completion_length` |

`dr_grpo` therefore avoids making the loss denominator depend on the sampled
completion lengths. It is still subject to the configured generation cap.

### GSPO-style sequence importance sampling

There is no separate `train_mode: "gspo"`. Select sequence-level importance
sampling on the GRPO path:

```json
{
  "model": "org/model",
  "data": "org/prompt-answer-dataset",
  "train": true,
  "train_type": "lora",
  "train_mode": "grpo",
  "importance_sampling_level": "sequence",
  "group_size": 4,
  "iters": 100
}
```

`"token"` retains one importance ratio per generated token. `"sequence"`
averages the valid-token log ratios per completion before clipping. Start with
the default token level and compare sequence-level runs on the same seed and
reward configuration.

### DAPO-style dual clipping

There is no separate DAPO mode. Set `epsilon_high` to use an asymmetric upper
clip with the normal GRPO path:

```json
{
  "model": "org/model",
  "data": "org/prompt-answer-dataset",
  "train": true,
  "train_type": "lora",
  "train_mode": "grpo",
  "epsilon": 0.0001,
  "epsilon_high": 0.01,
  "group_size": 4,
  "iters": 100
}
```

Leave `epsilon_high` unset for symmetric clipping. Do not describe
`epsilon_high` as a PPO setting; it is consumed by the GRPO loss.

## RLHF REINFORCE

Use the exact underscore spelling `"rlhf_reinforce"`. This mode generates two
completions per prompt, obtains scalar rewards from a judge model, and applies
a KL-regularized REINFORCE objective.

```json
{
  "model": "org/policy-model",
  "data": "org/prompt-dataset",
  "train": true,
  "train_type": "lora",
  "train_mode": "rlhf_reinforce",
  "judge": "org/reward-model",
  "reference_model_path": "org/policy-model",
  "judge_config": {"system_prompt": "Score helpfulness from 0 to 1."},
  "beta": 0.1,
  "max_completion_length": 512,
  "iters": 100
}
```

| Field | Default | Use |
| --- | ---: | --- |
| `judge` | required | Reward-model ID or tenant-approved local model path |
| `judge_config` | `{}` | Optional judge settings; `system_prompt` is consumed |
| `reference_model_path` | `null` | Frozen KL reference; defaults to the base model |
| `beta` | `0.1` | KL penalty coefficient |
| `max_completion_length` | `512` | Maximum tokens generated per prompt |

`alpha`, `epsilon`, `temperature`, reward-function fields, and
`dpo_cpo_loss_type` are not used by the current RLHF REINFORCE dispatch. The
dataset needs prompts, not pre-ranked chosen/rejected pairs.

## PPO

Use `train_mode: "ppo"` for the repository's PPO-style online preference
trainer. It generates a pair of responses, asks a human or model judge to
select the preferred response, and applies a clipped objective with a KL term.

```json
{
  "model": "org/policy-model",
  "data": "org/prompt-dataset",
  "train": true,
  "train_type": "lora",
  "train_mode": "ppo",
  "judge": "org/judge-model",
  "reference_model_path": "org/policy-model",
  "beta": 0.1,
  "epsilon": 0.2,
  "dpo_cpo_loss_type": "sigmoid",
  "temperature": 0.8,
  "max_completion_length": 512,
  "iters": 100
}
```

| Field | Default | Use |
| --- | ---: | --- |
| `judge` | required | Pairwise judge model; the trainer also has a human-judge path |
| `judge_config` | `{}` | Optional judge settings; `system_prompt` is consumed |
| `reference_model_path` | `null` | Frozen reference; defaults to the base model |
| `beta` | `0.1` | KL penalty coefficient |
| `epsilon` | `0.2` | PPO clipping bound |
| `temperature` | `0.8` | Sampling temperature |
| `max_completion_length` | `512` | Maximum generated tokens |
| `dpo_cpo_loss_type` | `"sigmoid"` | Scoring mode; `"ipo"` changes score normalization |
| `delta` | `50.0` | Accepted for DPOP-compatible scoring, but not otherwise used by PPO loss |

PPO consumes prompt rows such as `{"prompt": "Answer this question."}`. It
does not consume the GRPO `answer` field or reward-function registry.

## Online DPO and XPO

These modes are adjacent online preference trainers rather than GRPO variants.
They consume prompt-only rows and generate two completions per prompt.

### Online DPO

```json
{
  "model": "org/policy-model",
  "data": "org/prompt-dataset",
  "train": true,
  "train_type": "lora",
  "train_mode": "online_dpo",
  "judge": "org/judge-model",
  "reference_model_path": "org/policy-model",
  "beta": 0.1,
  "dpo_cpo_loss_type": "sigmoid",
  "delta": 50.0,
  "temperature": 0.8,
  "max_completion_length": 512,
  "iters": 100
}
```

Online DPO uses `beta`, `dpo_cpo_loss_type`, `delta`, `temperature`,
`max_completion_length`, `reference_model_path`, `judge`, and `judge_config`.
The loss choices are `sigmoid`, `hinge`, `ipo`, and `dpop`.

### XPO

XPO adds an exploration term to the online DPO objective:

```json
{
  "model": "org/policy-model",
  "data": "org/prompt-dataset",
  "train": true,
  "train_type": "lora",
  "train_mode": "xpo",
  "judge": "org/judge-model",
  "beta": 0.1,
  "alpha": [0.00001, 0.000005],
  "dpo_cpo_loss_type": "sigmoid",
  "delta": 50.0,
  "iters": 100
}
```

`alpha` is a list of exploration weights. With multiple values, the trainer
uses them by epoch and keeps the final value for later epochs. XPO also uses
`beta`, `dpo_cpo_loss_type`, `delta`, `reference_model_path`, `judge`,
`judge_config`, and `max_completion_length`. The current dispatch does not
pass a separate sampling temperature to XPO.

## Best practices by algorithm

### GRPO and its variants

- Define rewards on a small, inspectable sample before training. A reward that
  is always zero, always constant, or mostly invalid produces no useful group
  advantage.
- Keep reward functions aligned with the dataset's `answer` and the model's
  output format. Track per-function mean, standard deviation, and coverage;
  do not rely only on total reward.
- Start with `group_size: 4`, `grpo_loss_type: "grpo"`, token-level sampling,
  and symmetric clipping. Change one variant at a time for fair comparisons.
- Increase `group_size` only when reward variance is too noisy and memory allows
  it. Generation cost scales with prompts × group size × completion length.
- Use a low enough `temperature` to preserve task solvability but high enough
  to produce within-group variation. If every completion is identical, the
  normalized advantage is uninformative.
- Inspect `hit_max_tokens_ratio`; a high value means the completion cap or stop
  format is truncating trajectories and distorting rewards.
- Use `dr_grpo` or sequence-level importance sampling only with a controlled
  baseline. Compare reward, KL, clipping, completion length, and held-out task
  accuracy under the same seed.
- Treat reward weights as a model specification. Normalize incompatible reward
  scales before assigning weights, and keep the list length exactly aligned.
- Keep a frozen reference close to the starting policy and monitor KL. Rapidly
  increasing KL or clipping rates usually calls for a lower learning rate,
  smaller update budget, or stronger KL control.

### RLHF REINFORCE

- Use a reward model that is calibrated for the policy's task and output
  format. Test it independently on clearly good, bad, and ambiguous samples.
- Start with a conservative learning rate and `beta: 0.1`. Compare reward
  improvement against KL drift and general capability to catch reward hacking.
- Keep prompt data diverse and deduplicated. The trainer samples two responses
  per prompt, so repeated or near-identical prompts reduce effective coverage.
- Save frequent adapters and evaluate generations manually; scalar judge
  scores alone are insufficient for detecting degeneration.

### PPO

- Validate judge consistency before training. A noisy pairwise judge turns the
  clipped objective into a noisy target and can hide behind apparently stable
  losses.
- Start with `epsilon: 0.2`, `beta: 0.1`, moderate temperature, and short pilot
  runs. Tune the update size only after checking clipping and KL metrics.
- Keep the reference checkpoint fixed for the whole run and evaluate against a
  held-out prompt set that the judge never saw.
- Do not reuse GRPO reward-function settings or `epsilon_high`; PPO uses a
  pairwise judge and the single `epsilon` clipping field.

### Online DPO and XPO

- Use a fixed prompt pool and a deterministic judge configuration when
  comparing runs. Generated pair quality and judge drift otherwise confound
  the training signal.
- Start Online DPO with the sigmoid loss. Add XPO exploration only when the
  policy is too conservative and you have a strong KL/capability evaluation.
- For XPO, use a short `alpha` schedule and monitor exploration bonus and KL;
  excessive alpha can move the policy away from the reference too quickly.

## Operational checklist

Before starting a reinforcement job:

1. Confirm the exact mode spelling: `grpo`, `rlhf_reinforce`, `ppo`,
   `online_dpo`, or `xpo`.
2. Confirm the dataset schema: GRPO needs `prompt` and `answer`; all online
   judge modes need `prompt`.
3. Confirm reward function names or the tenant-local reward file before using a
   custom GRPO reward setup.
4. Confirm `group_size`, batch size, and completion length fit available memory.
5. Set `reference_model_path` explicitly when the reference must differ from
   the base model; otherwise the backend freezes the base model automatically
   for modes that use a reference.
6. Validate on held-out prompts and inspect generations before fusing the
   adapter.
