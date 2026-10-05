# Dataset sources and schemas

Use this reference when choosing a dataset, checking its columns/splits, or
diagnosing loader failures. MCP `data` accepts a Hugging Face dataset repository
ID such as `org/dataset`. Local JSONL/CSV paths, `tenant://` dataset paths, and
HTTP URLs are rejected, even though the direct training CLI has local loaders.

## Splits and column names

The MCP data path loads `train`, `valid`, and `test` splits by those exact
names. It does not map `validation` to `valid` or create held-out splits.
Training requires a nonempty `train`; `test: true` requires a nonempty `test`.
Check validation availability for the chosen run.

The loader uses the default columns below. Custom column mappings,
`hf_dataset` collections, subset names, and split expressions are not exposed
by the current MCP config. Do not invent config keys to map another schema.

## SFT

Use one of these record shapes:

```json
{"prompt": "Explain gradient descent.", "completion": "It updates parameters along the negative gradient."}
```

```json
{"messages": [{"role": "user", "content": "Explain gradient descent."}, {"role": "assistant", "content": "It updates parameters along the negative gradient."}]}
```

```json
{"text": "A complete continued-pretraining example."}
```

Chat rows may include `tools`, passed to the tokenizer's chat template.
`mask_prompt: true` masks the prompt in prompt/completion rows and everything
before the final assistant message in chat rows. Plain `text` rejects prompt
masking and receives an EOS token if needed. See [sft.md](sft.md).

## Offline preference pairs

DPO, CPO, DSLA, and ORPO accept the basic pair shape:

```json
{
  "prompt": "Why is the sky blue?",
  "chosen": "Air molecules scatter shorter visible wavelengths more strongly.",
  "rejected": "The sky is painted blue.",
  "system": "Answer accurately and concisely."
}
```

`system` is optional. DPO, CPO, and DSLA render responses with the policy
chat template. DSLA requires the generation prompt to be an exact token prefix
of both responses. ORPO also handles structured responses and optional
`preference_score`; see [orpo.md](orpo.md). FTPO's Antidoom schema is maintained
in [ftpo.md](ftpo.md).

## GRPO and KLPO

```json
{"prompt": "What is 2 + 2?", "answer": "4", "system": "Put the result in <answer> tags.", "type": "arithmetic"}
```

`prompt` and `answer` are required; `system` and `type` are optional. Without
`system`, the loader requests `<think>` and `<answer>` sections. The `answer`
is reference data for reward callbacks, not a pre-ranked response. Read
[reward_functions.md](reward_functions.md) to match feedback to the format.

## Online judge modes

```json
{"prompt": "Explain gradient descent."}
```

`prompt` may also hold role/content message lists. Online DPO, XPO, PPO, and
RLHF REINFORCE generate candidate pairs from these prompts. A separate
`system` column is not consumed by this loader; include system messages inside
the `prompt` message list when needed. See [judges.md](judges.md).
