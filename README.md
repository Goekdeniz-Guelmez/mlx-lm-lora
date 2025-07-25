<h1 align="center">MLX-LM-LORA</h1>

<p align="center">
  <img src="https://github.com/Goekdeniz-Guelmez/mlx-lm-lora/blob/main/logo.png" alt="logo" width="300"/>
</p>

Train Large Language Models localy on Apple Silicon using MLX. Training works with all the model that are supported with MLX-LM, for example:

- Llama, 3, 4
- Phi2, 3
- Mistral
- Mixtral
- Qwen2, 2.5, 3
- Qwen3 MoE
- Gemma1, 2, 3
- OLMo, OLMoE
- MiniCPM, MiniCPM3
- and more ...

Supported training algorithms include:

- LoRA, DoRA, and full-precision fine-tuning
- Supervised Fine-Tuning (SFT)
- Direct Preference Optimization (DPO)
- Online Direct Preference Optimization (Online DPO)
- eXtended Preference Optimization (XPO)
- Contrastive Preference Optimization (CPO)
- Odds Ratio Preference Optimization (ORPO)
- Reinforcement Learning from Human Feedback (RLHF)
- Group Relative Policy Optimization (GRPO)

### 📓 Notebooks Examples

* [🧪 LoRA Fine-Tuning (SFT)](examples/custom_sft_lora.ipynb) – Shows how to fine-tune a model using LoRA on a standard SFT dataset.
* [🧠 Full-Precision SFT](examples/custom_sft.ipynb) – Uses full model weights instead of LoRA for supervised fine-tuning.
* [⚖️ ORPO Training](examples/custom_orpo_lora.ipynb) – Monolithic preference optimization without the need for a reference model.
* [📈 CPO Training](examples/custom_cpo_lora.ipynb) – Contrastive fine-tuning to improve model decision boundaries.
* [👥 GRPO Training](examples/custom_grpo_lora.ipynb) – Group-based reinforcement training with multiple completions per prompt.
* [🧬 Pretraining](examples/pretrain_fineweb-200k.ipynb) – Pretrains a language model from scratch using a 200k-sample subset of the FineWeb dataset.
* [🚀 Training a model fully from scratch with Pre/Post-traininng](examples/qwen3_moe_from_scratch.ipynb) - Fully trains a Qwen3-MoE model from scratch, including both pretraining and preference-stage fine-tuning.


## Contents

- [Install](#Install)
- [Run](#Run)
  - [LoRA or Full-Precision](#Lora-or-Full-Precision)
  - [SFT](#SFT-Training)
  - [ORPO-Training](#ORPO-Training)
  - [DPO-Training](#DPO-Training)
  - [CPO-Training](#CPO-Training)
  - [GRPO-Training](#GRPO-Training)
  - [Evaluate](#Evaluate)
  - [Generate](#Generate)
- [Memory Issues](#Memory-Issues)

---

## Install

```shell
pip install mlx-lm-lora
```

## Run

The main command is `mlx_lm_lora.train`. To see a full list of command-line options run:

```shell
mlx_lm_lora.train --help
```

Note, in the following the `--model` argument can be any compatible Hugging
Face repo or a local path to a converted model.

You can also specify a YAML config with `-c`/`--config`. For more on the format see the
[example YAML](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora/blob/main/examples/example_lora.yaml). For example:

```shell
mlx_lm_lora.train --config /path/to/config.yaml
```

If command-line flags are also used, they will override the corresponding
values in the config.

---

### LoRA or Full-Precision

To fine-tune a model use:

```shell
mlx_lm_lora.train \
    --model <path_to_model> \
    --train \
    --data <path_to_data> \
    --iters 600
```

To fine-tune the full model weights, add the `--train-type full` flag.
Currently supported training types are `lora` (default), `dora`, and `full`.

The `--data` argument must specify a path to a `train.jsonl`, `valid.jsonl`
when using `--train` and a path to a `test.jsonl` when using `--test`. For more

If `--model` points to a quantized model, then the training will use QLoRA,
otherwise it will use regular LoRA.

By default, the adapter config and learned weights are saved in `adapters/`.
You can specify the output location with `--adapter-path`.

You can resume fine-tuning with an existing adapter with
`--resume-adapter-file <path_to_adapters.safetensors>`.

---

### SFT-Training

Supervised Fine-Tuning (SFT) trains a model using pairs of prompts and expected completions. This is the most common form of instruction tuning.

To run SFT:

mlx_lm_lora.train \
    --model <path_to_model> \
    --train \
    --data <path_to_data>

You can set the training type explicitly using --train-type lora, dora, or full. By default, LoRA is used.

Data Format

The data should be in JSONL format with one of the following structures:

Chat-style (preferred for chat models):

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "Paris."}
  ]
}
```

Prompt-completion style:

```json
{"prompt": "What is the capital of France?", "completion": "Paris."}
```

You can mix system messages or use multi-turn chat formatting depending on the target model’s capabilities.

Additional Options
	•	--use-chat-template: Format messages using the model’s chat template (default: False)
	•	--mask-prompt: Apply loss only on the assistant’s output instead of the full sequence
	•	--train-type: Choose lora, dora, or full for LoRA, DoRA, or full-weight fine-tuning
	•	--resume-adapter-file: Resume training from a previously saved adapter

Example with chat template and prompt masking:

```text
mlx_lm_lora.train \
    --model <path_to_model> \
    --train \
    --data <path_to_data> \
    --use-chat-template True \
    --mask-prompt
```

---

### ORPO-Training

Odds Ratio Preference Optimization (ORPO) was introduced in ORPO: Monolithic Preference Optimization without Reference Model by Jiwoo Hong, Noah Lee, and James Thorne. Usage:

```shell
mlx_lm_lora.train \
 --model <path_to_model> \
 --train \
 --train-mode orpo \
 --data <path_to_data> \
 --beta 0.1
```

Parameters:

- `--beta`: Temperature for logistic function (default: 0.1)

Data format (JSONL):

```jsonl
# Basic format with string responses
{"prompt": "User prompt", "chosen": "Preferred response", "rejected": "Less preferred response"}

# With custom preference score
{"prompt": "User prompt", "chosen": "Preferred response", "rejected": "Less preferred response", "preference_score": 8.0}

# With system message
{"prompt": "User prompt", "chosen": "Preferred response", "rejected": "Less preferred response", "system": "System instruction"}

# With full conversation objects
{
  "prompt": "User prompt",
  "chosen": {
    "messages": [
      {"role": "system", "content": "System instruction"},
      {"role": "user", "content": "User message"},
      {"role": "assistant", "content": "Assistant response"}
    ]
  },
  "rejected": {
    "messages": [
      {"role": "system", "content": "System instruction"},
      {"role": "user", "content": "User message"},
      {"role": "assistant", "content": "Assistant response"}
    ]
  }
}
```

The trainer assigns binary rewards (1.0 chosen, 0.0 rejected) if no explicit rewards provided via `preference_score`.

---

### DPO-Training

Direct Preference Optimization (DPO) training allows you to fine-tune models using human preference data, as described in the paper Direct Preference Optimization: Your Language Model is Secretly a Reward Model by Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn. To use DPO training, set the training mode to 'dpo':

```shell
mlx_lm.lora \
    --model <path_to_model> \
    --train \
    --train-mode dpo \
    --data <path_to_data> \
    --beta 0.1
```

The DPO training accepts the following additional parameters:

- `--beta`: Controls the strength of the DPO loss (default: 0.1)
- `--dpo-loss-type`: Choose between "sigmoid" (default), "hinge", "ipo", or "dpop" loss functions
- `--delta`: Margin parameter for hinge loss (default: 50.0)
- `--reference-model-path`: Path to a reference model for DPO training

For DPO training, the data should be in JSONL format with the following structure:

```jsonl
{"prompt": "User prompt", "chosen": "Preferred response", "rejected": "Less preferred response"}
```

if the Prompt template accept a system message, you can extend the Dataset with a additional "system" field.

```jsonl
{"system": "You are a helpfull assistant", "prompt": "User prompt", "chosen": "Preferred response", "rejected": "Less preferred response"}
```

---

### CPO-Training

Contrastive Preference Optimization (CPO) as introduced in the paper Contrastive Preference Optimization: Pushing the Boundaries of LLM Performance in Machine Translation by Haoran Xu, Amr Sharaf, Yunmo Chen, Weiting Tan, Lingfeng Shen, Benjamin Van Durme, Kenton Murray, and Young Jin Kim. At a high-level, CPO trains models to avoid generating adequate, but not perfect translations in Machine Translation (MT) tasks. However, CPO is a general approximation of the DPO loss and can be applied to other domains, such as chat.

CPO aims to mitigate two fundamental shortcomings of SFT. First, SFT’s methodology of minimizing the discrepancy between predicted outputs and gold-standard references inherently caps model performance at the quality level of the training data. Secondly, SFT lacks a mechanism to prevent the model from rejecting mistakes in translations. The CPO objective is derived from the DPO objective.
To use CPO training, set the training mode to 'cpo':

```shell
mlx_lm.lora \
    --model <path_to_model> \
    --train \
    --train-mode cpo \
    --data <path_to_data> \
    --beta 0.1
```

The CPO training accepts the following additional parameters:

- `--beta`: Controls the strength of the DPO loss (default: 0.1)
- `--dpo-loss-type`: Choose between "sigmoid" (default), "hinge", "ipo", or "dpop" loss functions
- `--delta`: Margin parameter for hinge loss (default: 50.0)

The CPO training uses teh exact same Dataset format defined in DPO.

---

### GRPO-Training
#### Overview

Group Relative Policy Optimization (GRPO) is a fine-tuning method that optimizes language models by generating multiple responses per prompt and learning from their relative quality. This approach helps improve response quality through comparative learning. As described in the paper DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models by Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Mingchuan Zhang, Y. K. Li, Y. Wu, Daya Guo.

#### Dataset Format

GRPO requires a dataset in JSONL format (one JSON object per line) with the following structure:

```json
{"prompt": "Your question or instruction here", "answer": "The expected response"}
```

Each entry must contain:
- `prompt`: The input text for the model to respond to
- `answer`: The target/reference response

Optional fields:
- `system`: A system message providing context or instructions for the model

Example entries:
```json
{"prompt": "Gerald spends $100 a month on baseball supplies. His season is 4 months long. He wants to use the months he's not playing baseball to save up by raking, shoveling, and mowing lawns. He charges $10 for each. How many chores does he need to average a month to save up for his supplies?", "answer": "5"}
{"prompt": "Ann is cutting fabric to make curtains. She cuts a 4 foot by 6 foot rectangle for the living room, and a 2 foot by 4 foot rectangle for the bedroom. If the bolt of fabric is 16 feet by 12 feet, how much fabric is left in square feet?", "answer": "160"}
{"prompt": "Arnel had ten boxes of pencils with the same number of pencils in each box. He kept ten pencils and shared the remaining pencils equally with his five friends. If his friends got eight pencils each, how many pencils are in each box?", "answer": "5", "system": "You are a helpful math tutor."}
```

#### Usage

To fine-tune a model using GRPO:

```shell
mlx_lm.lora \
    --model <path_to_model> \
    --train \
    --data <path_to_data> \
    --fine-tune-type grpo \
    --group-size 4
```

#### GRPO-Specific Arguments

- `--group-size`: Number of responses generated per prompt (default: 4)
- `--beta`: KL penalty coefficient for policy optimization (default: 0.1)
- `--epsilon`: Small constant for numerical stability (default: 1e-4)
- `--max-completion-length`: Maximum length of generated completions (default: 512)
- `--reference-model-path`: Path to reference model weights. If not specified, uses the same model
- `--temperature`: Sampling temperature for generations. Higher values increase randomness (default: 1.0)
- `--reward-weights`: Optional list of weights for multiple reward functions. Must match number of reward functions. If not specified, all rewards weighted equally with 1.0
- `--use-chat-template`: Whether to use the model's chat template for formatting prompts (default: False)
- `--use-prompt`: Whether to use the prompt as part of the input for generation (default: False)

#### Training Process

During GRPO training, the model:
1. Takes each prompt from the dataset
2. Generates multiple responses (specified by `--group-size`)
3. Evaluates these responses against the reference answer
4. Optimizes the policy based on the relative quality of the responses

#### Resource Considerations

GRPO requires more compute resources than standard LoRA training since it generates multiple responses per prompt. Consider:
- Reducing batch size
- Using gradient checkpointing
- Adjusting `--group-size` to balance between quality and resource usage

If running into memory issues, you can also try:
- Reducing `--max-completion-length`
- Using a smaller model for initial experiments

#### Example Command with Full Options

```shell
mlx_lm.lora \
    --model <path_to_model> \
    --train \
    --data <path_to_data> \
    --fine-tune-type grpo \
    --group-size 4 \
    --beta 0.1 \
    --epsilon 1e-4 \
    --max-completion-length 512 \
    --reference-model-path <optional_path_to_reference_model> \
    --temperature 1.0 \
    --reward-weights 1.0 1.0 \
    --use-chat-template False \
    --use-prompt False \
    --batch-size 4 \
    --learning-rate 1e-5 \
    --num-epochs 3
```

---

### Evaluate

To compute test set perplexity use:

```shell
mlx_lm_lora.train \
    --model <path_to_model> \
    --adapter-path <path_to_adapters> \
    --data <path_to_data> \
    --test
```

### Generate

For generation use `mlx-lm` with `mlx_lm.generate`:

```shell
mlx_lm.generate \
    --model <path_to_model> \
    --adapter-path <path_to_adapters> \
    --prompt "<your_model_prompt>"
```

#### Prompt Masking

The default training computes a loss for every token in the sample. You can
ignore the prompt and compute loss for just the completion by passing
`--mask-prompt`. Note this is only supported for `chat` and `completion`
datasets. For `chat` datasets the final message in the message list is
considered the completion.

---

## Memory Issues

Fine-tuning a large model with LoRA requires a machine with a decent amount
of memory. Here are some tips to reduce memory use should you need to do so:

1. Try quantization (QLoRA). You can use QLoRA by generating a quantized model
   with `convert.py` and the `-q` flag. See the [Setup](#setup) section for
   more details.

2. Try using a smaller batch size with `--batch-size`. The default is `4` so
   setting this to `2` or `1` will reduce memory consumption. This may slow
   things down a little, but will also reduce the memory use.

3. Reduce the number of layers to fine-tune with `--num-layers`. The default
   is `16`, so you can try `8` or `4`. This reduces the amount of memory
   needed for back propagation. It may also reduce the quality of the
   fine-tuned model if you are fine-tuning with a lot of data.

4. Longer examples require more memory. If it makes sense for your data, one thing
   you can do is break your examples into smaller
   sequences when making the `{train, valid, test}.jsonl` files.

5. Gradient checkpointing lets you trade-off memory use (less) for computation
   (more) by recomputing instead of storing intermediate values needed by the
   backward pass. You can use gradient checkpointing by passing the
   `--grad-checkpoint` flag. Gradient checkpointing will be more helpful for
   larger batch sizes or sequence lengths with smaller or quantized models.


---

## Citing MLX-LM-LoRA

The MLX-LM-LoRA software suite was developed by Gökdeniz Gülmez. If you find MLX-LM-LoRA useful in your research and wish to cite it, please use the following BibTex entry:

```
@software{
  MLX-LM-LoRA,
  author = {Gökdeniz Gülmez},
  title = {{MLX-LM-LoRA}: Train LLMs on Apple silicon with MLX and the Hugging Face Hub.},
  url = {https://github.com/Goekdeniz-Guelmez/mlx-lm-lora},
  version = {0.1.0},
  year = {2025},
}
```
