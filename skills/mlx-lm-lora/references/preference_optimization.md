# Offline preference training

Choose the requested objective; do not silently replace it with SFT or a
reference-free method to fit memory. All examples in algorithm references are
flat MCP calls; specialized settings go in the JSON string `extra_config`.

| Mode | Dataset | Frozen reference | Guide |
| --- | --- | --- | --- |
| DPO | prompt/chosen/rejected | Yes | [DPO](dpo.md) |
| CPO | prompt/chosen/rejected | No | [CPO](cpo.md) |
| ORPO | prompt/chosen/rejected, optional preference_score | No | [ORPO](orpo.md) |
| FTPO | Antidoom token-repair fields | Yes | [FTPO](ftpo.md) |
| DSLA | prompt/chosen/rejected with usable prompt/answer boundaries | For dsla_loss=dpo only | [DSLA](dsla.md) |

Use a Hub ID or local JSONL directory. SFT text/completion rows alone do not
make a preference dataset. Inspect pairs for ties, inverted labels, duplicates,
incompatible message formats and truncation. Keep an evaluation split that
tests preference gains and general capabilities. Choose an explicit step budget;
the current MCP server cannot execute epoch-only budgets correctly.

DPO/CPO margin variants use `dpo_cpo_loss_type`: sigmoid, hinge, ipo or dpop;
`delta` is the DPOP penalty, and `beta` scales the preference objective.
DSLA selects its underlying objective with `dsla_loss` separately. Prefer one
controlled baseline before changing loss variants, QAT or long-context behavior.
See the per-algorithm best practices and [memory matrix](memory-efficiency.md).
