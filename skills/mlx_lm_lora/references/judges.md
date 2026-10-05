# Model judges for online training

Read this reference for `online_dpo`, `xpo`, `ppo`, or `rlhf_reinforce`.
Shared prompt schemas are in [datasets.md](datasets.md).

## Model and reference roles

`judge` is required and names a Hugging Face model or tenant-approved local
model. The entrypoint loads a frozen generative judge model and tokenizer.
`reference_model_path` separately names the frozen policy reference; when
omitted, the starting policy model is used. The backend can reuse the loaded
reference as the judge when their explicit paths match.

Do not translate a user request for an interactive human judge into
`judge: "human"` for MCP. Some standalone trainers contain that branch, but
the current entrypoint still attempts to load `human` as a model identifier.
Report that limitation rather than claiming human judging works through MCP.

## Pairwise responses

Online DPO, XPO, and PPO use `LLMPairwiseJudge`. The judge must emit exactly
`0` or `1`, selecting one candidate. Extra explanation or formatting is invalid.
Candidate order is shuffled and judgments are mapped back to the original
order.

`judge_config` defaults to `{}`. The exposed override is `system_prompt`, a
Python format template using `{prompt}`, `{response0}`, and `{response1}`.
The current worker does not forward other constructor settings such as
`enable_reasoning` from `judge_config`.

```json
{
  "judge_config": {
    "system_prompt": "Question: {prompt}\nResponse 0: {response0}\nResponse 1: {response1}\nChoose the more helpful and accurate response. Return only 0 or 1."
  }
}
```

Despite the key name, this template becomes a user message passed through the
judge tokenizer's chat template. Retain all three placeholders so the judge
sees the question and both candidates. Escape literal braces as `{{` and `}}`
in custom templates that embed JSON.

## Numeric-score responses

RLHF REINFORCE uses `LLMPPOJudge` for numeric scores on candidate pairs. Its
judge must emit JSON with a `scores` list, string model identifiers `"0"` and
`"1"`, and numeric `score` values:

```json
{"scores": [{"model_identifier": "0", "score": 0.9}, {"model_identifier": "1", "score": 0.2}]}
```

Use the backend default template unless a supplied scoring template preserves
this contract. A scalar classification/reward-head checkpoint is not an
alternative to a generative model emitting this JSON.

Inspect judge outputs on known-good, known-bad, and ambiguous candidates.
Parsing failures can produce skipped pairwise judgments or fallback numeric
scores instead of terminating the job. Read the logs before interpreting
training rewards as reliable feedback.
