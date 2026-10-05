# Generated-response training overview

Use this overview when the user asks about reward training or online
preferences without selecting a method. These modes generate responses during
optimization; offline ranked pairs alone do not supply their feedback.

| Mode | Feedback | Frozen reference | Guide |
| --- | --- | --- | --- |
| `grpo` | Registered rewards over completion groups | Yes | [grpo.md](grpo.md) |
| `klpo` | Registered terminal rewards and sampler-conditioned KL records | No | [klpo.md](klpo.md) |
| `online_dpo` | Pairwise model judge | Yes | [online_dpo.md](online_dpo.md) |
| `xpo` | Pairwise model judge with exploration term | Yes | [xpo.md](xpo.md) |
| `ppo` | Pairwise model judge and clipped preference objective | Yes | [ppo.md](ppo.md) |
| `rlhf_reinforce` | Model judge returning numeric scores | Yes | [rlhf_reinforce.md](rlhf_reinforce.md) |

GSPO, BNPO, Dr. GRPO, and DAPO-style clipping are settings under
`train_mode: "grpo"`; they are not separate MCP mode names.

GRPO/KLPO require `prompt` and `answer` records and the callback setup in
[reward_functions.md](reward_functions.md). Online judge modes use prompt
records and the model/template contract in [judges.md](judges.md). Full schema
details are in [datasets.md](datasets.md).

Use [config.md](config.md) for shared controls, [memory.md](memory.md) for
generation/scoring bounds, and [quantization.md](quantization.md) for model
loading. None of these modes enables QAT or cached `efficient_long_context`
processing through the current MCP training entrypoint.

Compare held-out task behavior alongside rewards, judge reliability, KL drift,
clipping, and completion length. A higher training reward alone does not
establish better model quality.
