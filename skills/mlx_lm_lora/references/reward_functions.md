# GRPO and KLPO reward callbacks

Use this reference for reward selection, weighting, or custom Python callbacks.
Only GRPO and KLPO consume this registry through the MCP entrypoint. Online
judge modes use [judges.md](judges.md) instead.

## Discovery and configuration

Call `mlx_lm_lora_list_reward_functions` to obtain registered names and default
names without training or loading a custom file. Do not submit
`list_reward_functions: true`; MCP rejects that option in training requests.

| Field | Default | Format |
| --- | --- | --- |
| `reward_functions` | `null` | Comma-separated registered names; null uses all five default callbacks |
| `reward_weights` | `null` | String of comma-separated numeric weights, optionally bracketed |
| `reward_functions_file` | `null` | Approved local Python file that registers custom callbacks |

Default names are `r1_accuracy_reward_func`, `r1_int_reward_func`,
`r1_strict_format_reward_func`, `r1_soft_format_reward_func`, and `r1_count_xml`.
They target reasoning/answer XML formats; do not assume they fit every task.
Keep weights ordered and the same length as the selected callback list.

```json
{
  "reward_functions": "r1_accuracy_reward_func,r1_strict_format_reward_func",
  "reward_weights": "[0.8, 0.2]"
}
```

The backend parses weights with string operations, so a JSON numeric array is
not interchangeable with this string. Omitted weights leave the trainer's
unweighted reward combination in place.

## Custom rewards

A custom file is executed by the training worker and must register the names
selected in `reward_functions`. Use approved paths from
[multi-tenant.md](multi-tenant.md), such as `tenant://inputs/reward.py`.
Discovery does not execute that file, so newly defined names are not listed
until registration occurs. Confirm names from the supplied source.

```python
from mlx_lm_lora.trainer.grpo_reward_functions import register_reward_function

@register_reward_function("exact_match")
def exact_match(prompts, completions, answer, types=None):
    return [float(text.strip() == target.strip())
            for text, target in zip(completions, answer)]
```

The callback receives aligned lists of prompts, completion strings, reference
answers, and optional types. Return one numeric score or `None` per completion;
at least one callback must provide a valid reward for each completion. Keep
score scales intentional before assigning weights.

The example compares raw text and is appropriate only when raw exact matching
is the requested signal. XML-wrapped answers need extraction before comparison.

## Check the signal

Inspect an example's generated text, reference answer, per-function scores,
and reward coverage. Mostly invalid rewards, constant rewards, and wrong
output-format assumptions can eliminate useful learning signal. A failed
custom import or unknown name may be printed before the backend returns
without training; inspect logs and checkpoint production before treating a
successful job status as evidence that the intended reward run completed.
