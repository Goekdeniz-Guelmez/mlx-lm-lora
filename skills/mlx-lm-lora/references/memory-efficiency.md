# Memory efficiency

Budget model weights, trainable gradients and optimizer state, activation/logit
tensors, generation KV caches, rollout records, reference/judge models and
export peaks separately. A small adapter does not imply a small training peak.

## Practical tuning order

1. Start with LoRA, batch size 1 and a task-appropriate short context. Lower
   `num_layers` and adapter rank when appropriate; preserve the user's choices
   and explain quality/capacity tradeoffs. DoRA and full training are separate
   choices, not guaranteed memory improvements over LoRA.
2. Prefer a compatible pre-quantized policy when loading itself is the problem.
   On-the-fly `quantize_bits` first loads the source weights and can exceed
   memory before quantization. See [quantization](quantization.md).
3. Reduce sequence padding and unnecessary length. For preferences, keep usable
   chosen/rejected responses; for DSLA, preserve nonempty prompt and answer.
   In generated modes also reduce `max_completion_length`; prompt limits alone
   do not bound generation caches. Do not truncate away the supervised signal.
4. Enable `grad_checkpoint` where wired to trade recomputation for activations.
   Use gradient accumulation to retain a desired effective batch after reducing
   the physical batch; accumulation alone does not shrink a fixed minibatch.
   Inspect each trainer's update/iteration semantics before changing budgets.
5. For online judge modes, set `micro_batch_size` for scoring/backprop.
   This does not eliminate rollout generation or model-residency memory.
   GRPO/KLPO use internal scoring microbatches; their dispatch does not forward
   the configurable `micro_batch_size` field.
6. Keep `fuse: false` during pilots. Fusion/export can add another peak and
   dequantization can be expensive. Save adapters, assess the run, then plan
   an explicit export with enough headroom.

Read peak-memory reports in `training_log` and the system's unified-memory
pressure. Compare the same data/lengths across changes. On OOM, keep saved
checkpoints, identify whether loading, rollout, backward or export failed,
adjust the relevant setting, and report a new run ID. Avoid repeated blind
retries or claims of a universal safe model size for a given RAM amount.

## Controls actually forwarded by the current dispatcher

All 12 modes receive gradient checkpointing and gradient accumulation settings.
Recurrent chunking (`recurrence_chunk_size`, backend default 64) targets models
using the repository's recurrence patch; it is not a general attention-window knob.

| Modes | `efficient_long_context` | Configurable `micro_batch_size` | QAT wired |
| --- | --- | --- | --- |
| SFT | Yes, fixed 512-token sequential chunks | No | Yes |
| DPO, ORPO | Yes, fixed 512-token sequential chunks | No | Yes |
| CPO | Yes, fixed 512-token sequential chunks | No | No |
| DSLA | Explicitly rejected | No | Yes |
| FTPO | Not forwarded | No | No |
| GRPO, KLPO | Not forwarded | No; internal microbatching | No |
| Online DPO, XPO, PPO, RLHF REINFORCE | Not forwarded | Yes | No |

Sequential long-context chunks use cached state and are an implementation
choice, not a promise of identical full-sequence gradients. Test task quality
before relying on them. SFT `chunked_nll` chunks loss evaluation in 256-token
pieces; it is a different control from sequential model execution.

## Model and algorithm costs

| Modes | Additional resident models | Main extra pressure |
| --- | --- | --- |
| SFT, CPO, ORPO | No frozen reference loaded | Token logits and activations |
| DPO, FTPO | Frozen reference | Paired scoring; FTPO vocabulary/logit comparisons |
| DSLA-DPO | Frozen reference | Hidden states plus pair scoring |
| DSLA-CPO/ORPO | No frozen reference | Hidden states and pair scoring |
| GRPO | Frozen reference loaded by dispatcher | Group rollouts and KV caches |
| KLPO | No separate reference loaded | Behavior-policy records and KL estimator tensors |
| Online DPO, XPO, PPO, RLHF REINFORCE | Frozen reference plus judge | Paired generation, judge scoring, training |

The policy's quantization flag does not quantize separately loaded reference
or judge models. Supply compatible explicitly quantized IDs/paths if desired.
The judge can reuse an explicitly matching reference path; do not assume
implicit/default references are deduplicated. GRPO loads its reference even
when beta is zero; zero beta does not guarantee saving reference residency.

For GRPO, group size scales total generated work and rollout storage even
though internal scoring microbatches avoid a simple group-size multiplier
on backward peak. Keep group size at least 2 for meaningful within-group
advantages. For KLPO, `full` stores full-vocabulary behavior distributions;
`mc`/`topk` reduce stored auxiliaries but still compute current full log-probs.
`binary` avoids those stored auxiliary distributions; none eliminates the
model's vocabulary logits or guarantees a particular peak reduction.
