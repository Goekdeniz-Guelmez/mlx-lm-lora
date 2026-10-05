# KLPO reward training

Use `train_mode: "klpo"` for KL-Regularized Policy Optimization. It samples one
complete response per prompt and combines terminal rewards with KL records
from the collection sampler. It has no separate reference model or judge and
does not use GRPO group normalization or PPO clipping.

Use a Hugging Face dataset with `prompt` and `answer`, optionally `system` and
`type`. KLPO shares GRPO's registered reward callbacks, `reward_functions`,
`reward_functions_file`, and string-valued `reward_weights`. Read
[reinforcement_learning.md](reinforcement_learning.md) for reward formats.
Use `mlx_lm_lora_list_reward_functions` to discover registered names without
starting training or executing a custom reward file.

| Field | Default | Values |
| --- | --- | --- |
| `klpo_route` | `"token"` | `"token"` or `"sequence"` regression |
| `klpo_kl_estimator` | `"mc"` | `"mc"`, `"topk"`, `"binary"`, `"full"` |
| `klpo_mc_samples` | `128` | Positive auxiliary draws per prefix; at least 2 for sequence MC |
| `klpo_top_k` | `128` | Positive stored sampler head size for TopK-KL |
| `klpo_tail_floor` | `0.000001` | Finite probability floor strictly between 0 and 1 |
| `beta` | `0.1` | Finite positive KL regularization strength |
| `temperature` | `1.0` | Finite positive collection temperature |
| `max_completion_length` | `512` | Maximum generated tokens |

`temperature` reflects the MCP dictionary entrypoint's parser default. An
explicit value overrides it.

```json
{
  "model": "org/model",
  "data": "org/prompt-answer-dataset",
  "train": true,
  "train_mode": "klpo",
  "train_type": "lora",
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

Leave `reference_model_path`, `judge`, `group_size`, `epsilon`, `epsilon_high`,
`importance_sampling_level`, and `grpo_loss_type` out of KLPO-specific configs.
QAT and `efficient_long_context` are unsupported. Reduce the prompt batch,
completion cap, or recurrent chunk size to control memory; `micro_batch_size`
is not wired into KLPO dispatch.
