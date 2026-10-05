# Memory controls and supported modes

Read this reference for long contexts, recurrent models, scoring batches, or
out-of-memory failures. Preserve the user's model and budget when proposing a
smaller-memory configuration.

## Controls

| Field | Default | Behavior |
| --- | --- | --- |
| `batch_size` | `1` | Training minibatch or prompt batch |
| `gradient_accumulation_steps` | `1` | Accumulate minibatches before an optimizer update |
| `grad_checkpoint` | `false` | Recompute activations during backward |
| `recurrence_chunk_size` | `64` | Positive chunk size for supported memory-safe recurrent fallbacks |
| `efficient_long_context` | `false` | Cached sequential processing in 512-token chunks |
| `micro_batch_size` | `null` | Positive bound on online candidate/trajectory scoring |
| `max_seq_length` | `2048` | Input/context token limit; mode-specific handling |
| `max_completion_length` | `512` | Generated-token cap for online/reward modes |

## Mode boundaries

Cached `efficient_long_context` processing is exposed for SFT, DPO, CPO, and
ORPO. DSLA rejects it because alignment needs pooled full-sequence states.
FTPO, GRPO, KLPO, and online judge modes do not enable it through dispatch.

`micro_batch_size` bounds preference-pair scoring in online DPO, XPO, and PPO;
null resolves to `batch_size` and values above that are capped. REINFORCE
bounds sampled trajectories instead; null resolves to `2 * batch_size` and
larger values are capped there. These workers also use bounded candidate
generation batches. The setting does not change the requested prompt batch
or optimizer accumulation count.

GRPO and KLPO have internal bounded generation/scoring paths, but the current
entrypoint does not forward a user-configurable `micro_batch_size` to them.
For GRPO, `group_size` also multiplies generated responses; KLPO samples one
complete response per prompt.

`recurrence_chunk_size` is independent of both the sequence limit and the
512-token cached path. The backend selects applicable recurrent/SSM fallbacks
for supported model architectures; MCP does not expose separate patch toggles.
SFT's `chunked_nll` only chunks loss evaluation; see [sft.md](sft.md).

## Diagnose memory failures

Read the job status and log to identify whether loading, generation, scoring,
or backward failed. A smaller input/completion cap changes available context
or generated answers, while checkpointing and scoring microbatches mainly
change execution memory. Propose the control that addresses the failing stage
and preserve supervised targets when reducing context length.

The server serializes jobs in one process because the host's Apple Silicon
memory and GPU are shared. Queueing additional jobs does not increase available
memory for the running job. Worker count and dataset-size constraints can
still invalidate an otherwise syntactically valid batch configuration.
