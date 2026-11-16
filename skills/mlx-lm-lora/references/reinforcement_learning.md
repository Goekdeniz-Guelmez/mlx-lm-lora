# Generated-response training

| Mode | Required data/feedback | Guide |
| --- | --- | --- |
| GRPO | prompt, answer; registered rewards | [GRPO](grpo.md), [variants](grpo_variants.md) |
| KLPO | prompt, answer; registered rewards | [KLPO](klpo.md) |
| Online DPO | prompt; model judge | [Online DPO](online_dpo.md) |
| XPO | prompt; model judge | [XPO](xpo.md) |
| PPO | prompt; model judge | [PPO](ppo.md) |
| RLHF REINFORCE | prompt; scalar-reward model judge | [REINFORCE](rlhf_reinforce.md) |

Reward modes use registered reward names, a comma-separated `reward_functions`
string and, optionally, a JSON-list-encoded `reward_weights` string, all inside
`extra_config`. The builtin registry includes `r1_accuracy_reward_func`,
`r1_int_reward_func`, `r1_strict_format_reward_func`,
`r1_soft_format_reward_func`, `r1_count_xml`. Discover available names rather
than inventing functions. Optional `reward_functions_file` is an existing local
Python file registering the requested names; no tenant URI resolver exists.

Judge modes require `judge` in `extra_config`; the current training entrypoint
loads it as a model. Interactive `judge: human` is not a working background-MCP
workflow: the detached trainer has closed stdin and the entrypoint attempts a
model load. `judge_config` carries judge options such as `system_prompt`.

`max_seq_length` limits prompts/inputs; `max_completion_length` separately
limits rollouts. Validate reward diversity or judge consistency before long
runs. Read rewards, KL, length/truncation and clipping from logs, not only
generic status loss. Hold out prompts and examine generations to detect reward
hacking. QAT and `efficient_long_context` are not wired for these modes.
See [memory efficiency](memory-efficiency.md) for model copies and microbatching.
