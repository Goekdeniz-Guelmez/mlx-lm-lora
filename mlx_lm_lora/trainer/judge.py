import json
from typing import Optional

import mlx.nn as nn
import numpy as np
from mlx_lm.generate import generate
from tqdm import tqdm

RAW_TRAINING_SYSTEM_PROMPT = """You are a binary‑preference evaluator.  
For each interaction you will receive:

```json
{
    "instruction": "prompt",
}

{
    {
        "model_identifier": "0",
        "output": "response0"
    },
    {
        "model_identifier": "1",
        "output": "response1"
    }
}
```

Your ONLY output must be a single digit:

* `0` – if you judge **model_identifier‑0** to be the better (more helpful, truthful, safe, and pleasant) response for the user.  
* `1` – if you judge **model_identifier‑1** to be the better response.

**Do NOT** add spaces, newlines, punctuation, explanations, or any other characters.  
If you cannot determine a clear winner, choose the answer that is **safer** or **more accurate**; if both are equally good, default to `0`.

### Evaluation Guidelines

1. **Helpfulness** – Does the answer directly address the user’s request and provide useful information?
2. **Truthfulness** – Is the content fact‑correct and free of hallucination?
3. **Clarity & Tone** – Is the language clear, polite, and appropriate for a wide audience?
4. **Conciseness** – Does the answer give the needed information without unnecessary verbosity?

### Decision Process (internal, you do not output it)

- Compare 0 and 1 on the five criteria above, ranking each criterion 1 (better) / 0 (worse) for each answer.
- Sum the scores; the answer with the higher total wins.
- In case of a tie, prefer the answer with the higher safety score.
- If still tied, default to `0` (pick model_identifier‑0).

### Prohibited Output

- Anything other than the single character `0` or `1`.
- Any mention of the evaluation process, reasons, or meta‑information.

---

**Example (for illustration only, not to be emitted):**
{
    "instruction": "How do I reset my router?",
}

{
    {
        "model_identifier": "0",
        "output": "Unplug it, wait 30 seconds, plug it back in."
    },
    {
        "model_identifier": "1",
        "output": "Press the reset button for 10 seconds, then log into the admin panel to configure Wi‑Fi settings."
    }
}

→ Output: 1

---

Remember: **Your response is always exactly one digit, `0` or `1`.**
"""


DEFAULT_PAIRWISE_SYSTEM_PROMPT = '''I require a leaderboard for various large language models. I'll provide you with prompts given to these models and their corresponding outputs. Your task is to assess these responses, and select the model that produces the best output from a human perspective.

## Instruction

{{
    "instruction": """{prompt}""",
}}

## Model Outputs

Here are the unordered outputs from the models. Each output is associated with a specific model, identified by a unique model identifier.

{{
    {{
        "model_identifier": "0",
        "output": """{response0}"""
    }},
    {{
        "model_identifier": "1",
        "output": """{response1}"""
    }}
}}

## Task

Evaluate the models on the basis of the quality and relevance of their results, and select the model that generated the best result. Reply with the identifier of the best model. Our evaluation will only take into account the first character of your answer, so make sure it contains only one of the identifiers and nothing else (no quotation marks, no spaces, no new lines, ...).
'''

PPO = '''You are an impartial judge assessing the quality of responses to a given prompt.

## Instruction

{{
    "instruction": """{prompt}""",
}}

## Model Outputs

Here are the outputs from the models. Each output is associated with a specific model, identified by a unique model identifier.

{{
    {{
        "model_identifier": "0",
        "output": """{response0}"""
    }},
    {{
        "model_identifier": "1",
        "output": """{response1}"""
    }}
}}

## Task

Evaluate each response independently on a continuous scale based on quality, relevance, and helpfulness. For each model, output a JSON object with the model identifier and its numerical score. Use the following format:

{{
    "scores": [
        {{"model_identifier": "0", "score": <float>}},
        {{"model_identifier": "1", "score": <float>}}
    ]
}}

Provide scores that reflect the relative quality of each response. The scores should be between 0 and 10, with higher being better, so make sure it contains only one of the json and nothing else (no quotation marks, no other text, no new lines, ...).
'''

DEFAULT_PAIRWISE_HUMAN_PROMPT = '''## Instruction

{{
    "instruction": """{prompt}""",
}}

## Model Outputs

{{
    {{
        "model_identifier": "0",
        "output": """{response0}"""
    }},
    {{
        "model_identifier": "1",
        "output": """{response1}"""
    }}
}}

'''


class LLMPairwiseJudge:
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Optional[str] = None,
        system_prompt: Optional[str] = None,
        enable_reasoning: bool = False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.enable_reasoning = enable_reasoning
        self.system_prompt = system_prompt or DEFAULT_PAIRWISE_SYSTEM_PROMPT

    def judge(
        self,
        prompts: list[str],
        completions: list[list[str]],
        shuffle_order: bool = True,
    ) -> list[int]:
        if shuffle_order:
            flip_mask = np.random.randint(0, 2, (len(prompts),)).astype(bool)
            completions = [
                pair[::-1] if flip else pair
                for flip, pair in zip(flip_mask, completions)
            ]

        def get_rank(prompt, candidates):
            content = self.system_prompt.format(
                prompt=prompt, response0=candidates[0], response1=candidates[1]
            )
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                enable_thinking=self.enable_reasoning,
                add_generation_prompt=True,
            )
            response = generate(self.model, self.tokenizer, prompt, max_tokens=16)
            if response in ["0", "1"]:
                return int(response)
            else:
                tqdm.write(
                    f"Invalid response from the judge model: '{response}'. Returning -1."
                )
                return -1

        ranks = []
        for prompt, completion in zip(prompts, completions):
            ranks.append(get_rank(prompt, completion))

        if shuffle_order:
            ranks = [
                ranks[i] if not flip else 1 - ranks[i]
                for i, flip in enumerate(flip_mask)
            ]

        return ranks


class LLMPPOJudge:
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Optional[str] = None,
        system_prompt: Optional[str] = None,
        enable_reasoning: bool = False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.enable_reasoning = enable_reasoning
        self.system_prompt = system_prompt or PPO

    def judge(
        self,
        prompts: list[str],
        completions: list[list[str]],
        shuffle_order: bool = True,
    ) -> list[list[float]]:
        if shuffle_order:
            flip_mask = np.random.randint(0, 2, (len(prompts),)).astype(bool)
            completions = [
                pair[::-1] if flip else pair
                for flip, pair in zip(flip_mask, completions)
            ]

        def get_scores(prompt, candidates):
            content = self.system_prompt.format(
                prompt=prompt, response0=candidates[0], response1=candidates[1]
            )
            messages = [{"role": "user", "content": content}]
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                enable_thinking=self.enable_reasoning,
                add_generation_prompt=True,
            )
            response = generate(
                self.model,
                self.tokenizer,
                prompt_text,
                max_tokens=200,
            )

            # Try to extract JSON from response
            try:
                # Find JSON object in response
                start_idx = response.find("{")
                end_idx = response.rfind("}")
                if start_idx == -1 or end_idx == -1:
                    raise ValueError("No JSON found in response")

                json_str = response[start_idx : end_idx + 1]
                score_data = json.loads(json_str)

                # Build score dictionary
                score_dict = {}
                for item in score_data["scores"]:
                    model_id = item["model_identifier"]
                    score_dict[model_id] = float(item["score"])

                return [score_dict.get("0", 0.5), score_dict.get("1", 0.5)]
            except Exception as e:
                tqdm.write(
                    f"Error parsing judge response: {e}\nResponse: {response}\nUsing fallback scores."
                )
                return [0.5, 1.0]  # Neutral fallback

        scores_list = []
        for prompt, completion in zip(prompts, completions):
            scores = get_scores(prompt, completion)
            scores_list.append(scores)

        if shuffle_order:
            # Unshuffle scores by reversing when order was flipped
            scores_list = [
                [s[1], s[0]] if flip else s for s, flip in zip(scores_list, flip_mask)
            ]

        return scores_list


class HumanPairwiseJudge:
    def __init__(
        self,
        prompt: Optional[str] = None,
    ):
        self.prompt = prompt or DEFAULT_PAIRWISE_HUMAN_PROMPT

    def judge(
        self,
        prompts: list[str],
        completions: list[list[str]],
        shuffle_order: bool = True,
    ) -> list[int]:
        if shuffle_order:
            flip_mask = np.random.randint(0, 2, (len(prompts),)).astype(bool)
            completions = [
                pair[::-1] if flip else pair
                for flip, pair in zip(flip_mask, completions)
            ]

        def get_rank(prompt, candidates):
            content = self.prompt.format(
                prompt=prompt, response0=candidates[0], response1=candidates[1]
            )
            tqdm.write(content)
            response = input(f"\nChoose with one is better (0, 1): ")
            if response in ["0", "1"]:
                return int(response)
            else:
                tqdm.write(
                    f"Invalid response from the judge model: '{response}'. Returning -1."
                )
                return -1

        ranks = []
        for prompt, completion in zip(prompts, completions):
            ranks.append(get_rank(prompt, completion))

        if shuffle_order:
            ranks = [
                ranks[i] if not flip else 1 - ranks[i]
                for i, flip in enumerate(flip_mask)
            ]

        return ranks
