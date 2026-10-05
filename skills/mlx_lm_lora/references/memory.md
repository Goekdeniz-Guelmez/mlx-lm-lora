# Training memory controls

The backend defaults to `batch_size: 1`. Preserve the user's requested batch
and distinguish scoring microbatches from optimizer accumulation.

| Field | Default | Scope and behavior |
| --- | --- | --- |
| `micro_batch_size` | `null` | Positive scoring batch for `online_dpo`, `xpo`, `rlhf_reinforce`, and `ppo`; null uses the prompt batch size |
| `recurrence_chunk_size` | `64` | Positive chunk size for automatic memory-safe recurrent fallbacks |
| `grad_checkpoint` | `false` | Recompute activations during backward |
| `gradient_accumulation_steps` | `1` | Accumulate minibatches before updating |
| `efficient_long_context` | `false` | Sequential 512-token cached chunks in SFT, DPO, CPO, ORPO |

`recurrence_chunk_size` controls recurrent training fallbacks when the model
uses a supported recurrence. It is independent of `max_seq_length` and the
512-token `efficient_long_context` path. There is no MCP toggle for a specific
architecture patch; the backend chooses the applicable fallback.

DSLA rejects `efficient_long_context`; FTPO, GRPO, KLPO, and online judge modes
do not wire it into training. Do not enable it for these modes.

For online judge training, lowering `micro_batch_size` bounds scoring memory
without changing the requested prompt batch. For GRPO, `group_size` multiplies
generated responses; KLPO generates one per prompt. Completion caps also
affect memory. Report out-of-memory failures from the job log, and preserve
the user's budget when proposing a revised config.
