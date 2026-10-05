# PPO-style online preference training

Use `train_mode: "ppo"` for this backend's clipped online preference objective.
It generates response pairs, asks a pairwise model judge to choose the better
response, and uses a frozen reference. It is not an exposed actor/value-critic
training interface.

Read [datasets.md](datasets.md) for prompts, [judges.md](judges.md) for the
pairwise judge contract, and [config.md](config.md) for shared settings.

## Clipping and scoring

| Field | MCP default | Meaning |
| --- | --- | --- |
| `judge` | Required | Pairwise judge model |
| `judge_config` | `{}` | Judge template settings |
| `reference_model_path` | `null` | Frozen reference; null uses the starting model |
| `beta` | `0.1` | KL coefficient |
| `epsilon` | `0.0001` | PPO ratio clipping bound |
| `temperature` | `1.0` | Candidate sampling temperature |
| `max_completion_length` | `512` | Maximum generated tokens per candidate |
| `dpo_cpo_loss_type` | `"sigmoid"` | Score normalization selector; `"ipo"` uses mean scores |

The standalone PPO dataclass uses `epsilon: 0.2`, but the MCP dictionary
entrypoint supplies `0.0001`. Set `epsilon` explicitly when the user requests
a different clipping budget. `delta` is accepted by the entrypoint but does
not change the PPO loss; `epsilon_high` is a GRPO setting.

## Example

```json
{
  "model": "org/model",
  "data": "org/prompt-dataset",
  "train": true,
  "train_type": "lora",
  "train_mode": "ppo",
  "judge": "org/judge-model",
  "beta": 0.1,
  "epsilon": 0.2,
  "temperature": 0.8,
  "max_completion_length": 256,
  "micro_batch_size": 1,
  "iters": 100
}
```

Read [memory.md](memory.md) for scoring microbatches and recurrent chunks.
QAT, `efficient_long_context`, reward-function callbacks, and GRPO group
settings are not used by PPO dispatch.

Compare judge consistency, clipping, KL drift, and held-out generations.
Stable clipped training loss does not establish that the judge ranks pairs
correctly or that the model preserves general capability.
