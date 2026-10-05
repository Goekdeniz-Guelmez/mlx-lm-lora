# KL-Regularized Policy Optimization

Use `train_mode: "klpo"` for one complete sampled response per prompt, terminal
rewards, and collection-sampler KL records. KLPO has no separate frozen
reference model or judge. It does not use GRPO group normalization or PPO
ratio clipping.

Read [datasets.md](datasets.md) for prompt/answer rows,
[reward_functions.md](reward_functions.md) for the shared GRPO/KLPO callback
contract, and [config.md](config.md) for common settings.

## Routes and estimators

| Field | MCP default | Values / meaning |
| --- | --- | --- |
| `klpo_route` | `"token"` | `"token"` or `"sequence"` regression |
| `klpo_kl_estimator` | `"mc"` | `"mc"`, `"topk"`, `"binary"`, or `"full"` |
| `klpo_mc_samples` | `128` | Positive auxiliary draws per visited prefix |
| `klpo_top_k` | `128` | Positive stored sampler head size for TopK-KL |
| `klpo_tail_floor` | `0.000001` | Finite probability floor strictly in (0, 1) |
| `beta` | `0.1` | Finite positive KL regularization strength |
| `temperature` | `1.0` | Finite positive collection temperature |
| `max_completion_length` | `512` | Maximum generated tokens |

Sequence regression with MC-KL requires `klpo_mc_samples >= 2`. Full KL stores
and evaluates vocabulary-wide records; estimator choice changes the memory
cost. MCP can validate numeric constraints without knowing model vocabulary
size or available memory.

## Example

```json
{
  "model": "org/model",
  "data": "org/prompt-answer-dataset",
  "train": true,
  "train_type": "lora",
  "train_mode": "klpo",
  "klpo_route": "sequence",
  "klpo_kl_estimator": "mc",
  "klpo_mc_samples": 128,
  "beta": 0.1,
  "temperature": 0.8,
  "reward_functions": "r1_accuracy_reward_func",
  "reward_weights": "[1.0]",
  "max_completion_length": 256,
  "iters": 100
}
```

Do not add `reference_model_path`, `judge`, `group_size`, GRPO loss settings,
or clipping fields to a KLPO-specific request. QAT, `efficient_long_context`,
and a user-controlled `micro_batch_size` are not wired into KLPO dispatch.
Read [memory.md](memory.md) for supported memory controls.

Inspect reward validity, generated answer quality, KL-estimator diagnostics,
and completion truncation. Compare routes/estimators with the same reward
setup, prompts, seed, and update budget.
