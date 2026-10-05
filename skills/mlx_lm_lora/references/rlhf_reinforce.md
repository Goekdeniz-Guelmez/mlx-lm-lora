# RLHF REINFORCE with KL

Use the exact mode name `rlhf_reinforce`. This trainer generates two responses
per prompt, asks a model judge for numeric scores, and applies a KL-regularized
REINFORCE objective against a frozen reference.

Read [datasets.md](datasets.md) for prompts, [judges.md](judges.md) for the
numeric-score response contract, and [config.md](config.md) for shared options.
The judge is a generative scoring model used by `LLMPPOJudge`; the entrypoint
does not expose a separate scalar reward-head loader.

## Settings

| Field | MCP default | Meaning |
| --- | --- | --- |
| `judge` | Required | Hub or approved local scoring model |
| `judge_config` | `{}` | Numeric-score judge template settings |
| `reference_model_path` | `null` | Frozen KL reference; null uses the starting model |
| `beta` | `0.1` | KL penalty coefficient |
| `max_completion_length` | `512` | Maximum generated tokens per candidate |

`micro_batch_size` bounds trajectory scoring; see [memory.md](memory.md).
`alpha`, `epsilon`, `epsilon_high`, `temperature`, `dpo_cpo_loss_type`, and
reward-function callbacks are not consumed by the current REINFORCE dispatch.
QAT and `efficient_long_context` are also unsupported.

## Example

```json
{
  "model": "org/model",
  "data": "org/prompt-dataset",
  "train": true,
  "train_type": "lora",
  "train_mode": "rlhf_reinforce",
  "judge": "org/scoring-model",
  "beta": 0.1,
  "max_completion_length": 256,
  "micro_batch_size": 1,
  "iters": 100
}
```

Check score calibration and judge parsing on an inspectable sample. Compare
reward improvement with KL drift and held-out task behavior. A malformed
judge response can yield fallback scores rather than a hard job failure; use
the training log to detect this before trusting the rewards.
