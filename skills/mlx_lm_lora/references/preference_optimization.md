# Offline preference overview

Use this overview when the user has ranked responses or asks which offline
preference method fits the task. If the algorithm is already named, read its
guide directly. Preserve an explicit algorithm choice.

| Mode | Training signal | Frozen reference | Guide |
| --- | --- | --- | --- |
| `dpo` | Chosen/rejected sequence preferences relative to a reference | Yes | [dpo.md](dpo.md) |
| `cpo` | Chosen/rejected policy score contrast | No | [cpo.md](cpo.md) |
| `orpo` | Chosen-response NLL plus odds-ratio preferences | No | [orpo.md](orpo.md) |
| `dsla` | DPO, ORPO, or CPO plus hidden-state alignment | DPO objective only | [dsla.md](dsla.md) |
| `ftpo` | Final-token preference repair on Antidoom-style records | Yes | [ftpo.md](ftpo.md) |

DPO, CPO, ORPO, and DSLA use `prompt`, `chosen`, and `rejected` records. FTPO
uses a distinct token-level schema. Read [datasets.md](datasets.md) for shared
record formats and split requirements.

Reference-using modes load a frozen copy of the starting `model` unless
`reference_model_path` specifies another Hub model or approved local model.
Reference-free modes do not load this model even if the field is present.

All modes use the shared controls in [config.md](config.md). Read
[quantization.md](quantization.md) for quantized loading or QAT, and
[memory.md](memory.md) for checkpointing, accumulation, and recurrent chunks.

Online DPO, XPO, PPO, and RLHF REINFORCE generate responses and need a judge;
use [reinforcement_learning.md](reinforcement_learning.md) for those methods.
SFT datasets without ranked pairs are described in [sft.md](sft.md).
