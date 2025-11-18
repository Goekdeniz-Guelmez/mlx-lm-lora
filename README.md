<p align="center">
  <img src="https://github.com/Goekdeniz-Guelmez/mlx-lm-lora/blob/main/logos/mlx_lm_lora.png" alt="logo" width="100%"/>
</p>

# MLX-LM-LORA

[![image](https://img.shields.io/pypi/v/mlx-lm-lora.svg)](https://pypi.python.org/pypi/mlx-lm-lora)

With MLX-LM-LoRA you can, train Large Language Models locally on Apple Silicon using MLX. Training works with all models supported by MLX-LM, including:

- Llama 3, 4
- Phi 2, 3
- Mistral
- Mixtral
- Qwen 2, 2.5, 3
- Qwen3 MoE
- Qwen3 Next
- Gemma 1, 2, 3
- OLMo, OLMoE
- MiniCPM, MiniCPM3
- and more...

## Supported Training Methods

**Training Types:**

- **LoRA**: Low-Rank Adaptation for efficient fine-tuning
- **DoRA**: Weight-Decomposed Low-Rank Adaptation
- **Full-precision**: Train all model parameters
- **Quantized training**: QLoRA with 4-bit, 6-bit, or 8-bit quantization

**Training Algorithms:**

- **SFT**: Supervised Fine-Tuning
- **DPO**: Direct Preference Optimization
- **CPO**: Contrastive Preference Optimization
- **ORPO**: Odds Ratio Preference Optimization
- **GRPO**: Group Relative Policy Optimization
- **GSPO**: Group Sequence Policy Optimization
- **Dr. GRPO**: Dr. Group Relative Policy Optimization
- **DAPO**: Decoupled Clip and Dynamic Sampling Policy Optimization
- **Online DPO**: Online Direct Preference Optimization
- **XPO**: Extended Preference Optimization
- **RLHF Reinforce**: Reinforcement Learning from Human Feedback

## New Features

**Synthetic Dataset Creation:**

- **SFT**: Create a synthetic sft dataset using a teacher model
- **Preference**: Create a synthetic preference dataset using a base and a teacher model

**Training Your Custom Preference Model:**

- You can now train a custom preference model for online preference training

## 📓 Example Notebooks

- [🧪 LoRA Fine-Tuning (SFT)](examples/custom_sft_lora.ipynb) – Shows how to fine-tune a model using LoRA on a standard SFT dataset.
- [🧠 Full-Precision SFT](examples/custom_sft.ipynb) – Uses full model weights instead of LoRA for supervised fine-tuning.
- [⚖️ ORPO Training](examples/custom_orpo_lora.ipynb) – Monolithic preference optimization without the need for a reference model.
- [📈 CPO Training](examples/custom_cpo_lora.ipynb) – Contrastive fine-tuning to improve model decision boundaries.
- [👥 GRPO Training](examples/custom_grpo_lora.ipynb) – Group-based reinforcement training with multiple completions per prompt.
- [🧬 Pretraining](examples/pretrain_fineweb-200k.ipynb) – Pretrains a language model from scratch using a 200k-sample subset of the FineWeb dataset.
- [🚀 Training a model fully from scratch with Pre/Post-training](examples/qwen3_moe_from_scratch.ipynb) - Fully trains a Qwen3-MoE model from scratch, including both pretraining and preference-stage fine-tuning.

## Contents

- [Install](#install)
- [Quick Start](#quick-start)
- [Training Methods](#training-methods)
  - [Supervised Fine-Tuning (SFT)](#supervised-fine-tuning-sft)
  - [Direct Preference Optimization (DPO)](#direct-preference-optimization-dpo)
  - [Contrastive Preference Optimization (CPO)](#contrastive-preference-optimization-cpo)
  - [Odds Ratio Preference Optimization (ORPO)](#odds-ratio-preference-optimization-orpo)
  - [Group Relative Policy Optimization (GRPO)](#group-relative-policy-optimization-grpo)
  - [Group Sequence Policy Optimization (GSPO)](#group-sequence-policy-optimization-gspo)
  - [Decoupled Reward Group Relative Policy Optimization (Dr. GRPO)](#decoupled-reward-group-relative-policy-optimization-dr-grpo)
  - [Decoupled Clip and Dynamic Sampling Policy Optimization (DAPO)](#decoupled-clip-and-dynamic-sampling-policy-optimization-dapo)
  - [Online DPO](#online-dpo)
  - [eXtended Preference Optimization (XPO)](#extended-preference-optimization-xpo)
  - [Reinforcement Learning from Human Feedback Reinforce (RLHF Reinforce)](#reinforcement-learning-from-human-feedback-rlhf-reinforce)
- [Other Features](#other-features)
  - [Synthetic Dataset Creation](#synthetic-dataset-creation)
    - [SFT](#synthetic-sft-dataset-creation)
    - [Preference](#synthetic-preference-dataset-creation)
  - [Training Your Custom Preference Model](#training-your-custom-preference-model)
- [Configuration](#configuration)
- [Dataset Formats](#dataset-formats)
- [Memory Optimization](#memory-optimization)
- [Evaluation & Generation](#evaluation--generation)

---

## Install

```shell
pip install -U mlx-lm-lora
```

## Quick Start

The main command is `mlx_lm_lora.train`. To see all options:

```shell
mlx_lm_lora.train --help
```

Basic training command:

```shell
mlx_lm_lora.train \
--model Goekdeniz-Guelmez/Josiefied-Qwen2.5-0.5B-Instruct-abliterated-v1 \
--train \
--data mlx-community/wikisql \
--iters 600
```

You can specify a YAML config with `-c`/`--config`:

```shell
mlx_lm_lora.train --config /path/to/config.yaml
```

Command-line flags will override corresponding values in the config file.

---

## Training Methods

### Supervised Fine-Tuning (SFT)

Standard instruction tuning using prompt-completion pairs.

```shell
mlx_lm_lora.train \
--model Goekdeniz-Guelmez/Josiefied-Qwen2.5-0.5B-Instruct-abliterated-v1 \
--train \
--train-mode sft \
--data mlx-community/hermes-3 \
--batch-size 4 \
--learning-rate 1e-5 \
--iters 1000
```

**Key Parameters:**

- `--train-type`: Choose `lora` (default), `dora`, or `full`
- `--mask-prompt`: Apply loss only to assistant responses
- `--max-seq-length`: Maximum sequence length (default: 2048)
- `--gradient-accumulation-steps`: Accumulate gradients over multiple steps

**Dataset Format:**

```jsonl
{"messages": [{"role": "user", "content": "What is AI?"}, {"role": "assistant", "content": "AI is..."}]}
{"prompt": "Explain quantum computing", "completion": "Quantum computing uses..."}
{"text": "Complete text for language modeling"}
```

---

### Direct Preference Optimization (DPO)

Train models using preference pairs without a separate reward model.

```shell
mlx_lm_lora.train \
--model Goekdeniz-Guelmez/Josiefied-Qwen2.5-0.5B-Instruct-abliterated-v1 \
--train \
--train-mode dpo \
--data mlx-community/Human-Like-DPO \
--beta 0.1 \
--dpo-cpo-loss-type sigmoid \
--reference-model-path Goekdeniz-Guelmez/Josiefied-Qwen2.5-0.5B-Instruct-abliterated-v1
```

**Key Parameters:**

- `--beta`: KL penalty strength (default: 0.1)
- `--dpo-cpo-loss-type`: Loss function - `sigmoid`, `hinge`, `ipo`, or `dpop`
- `--delta`: Margin for hinge loss (default: 50.0)
- `--reference-model-path`: Reference model path (uses main model if not specified)

**Dataset Format:**

```jsonl
{"prompt": "User question", "chosen": "Good response", "rejected": "Bad response"}
{"system": "You are helpful", "prompt": "Question", "chosen": "Good", "rejected": "Bad"}
```

---

### Contrastive Preference Optimization (CPO)

Variant of DPO designed for machine translation and other structured tasks.

```shell
mlx_lm_lora.train \
--model Goekdeniz-Guelmez/Josiefied-Qwen2.5-0.5B-Instruct-abliterated-v1 \
--train \
--train-mode cpo \
--data mlx-community/Human-Like-DPO \
--beta 0.1 \
--dpo-cpo-loss-type sigmoid
```

**Key Parameters:**
Same as DPO. Uses identical dataset format to DPO.

---

### Odds Ratio Preference Optimization (ORPO)

Monolithic preference optimization without requiring a reference model.

```shell
mlx_lm_lora.train \
--model Goekdeniz-Guelmez/Josiefied-Qwen2.5-0.5B-Instruct-abliterated-v1 \
--train \
--train-mode orpo \
--data mlx-community/Human-Like-DPO \
--beta 0.1 \
--reward-scaling 1.0
```

**Key Parameters:**

- `--beta`: Temperature for logistic function (default: 0.1)
- `--reward-scaling`: Reward scaling factor (default: 1.0)

**Dataset Format:**

```jsonl
{"prompt": "Question", "chosen": "Good response", "rejected": "Bad response"}
{"prompt": "Question", "chosen": "Good", "rejected": "Bad", "preference_score": 8.0}
{"prompt": "Question", "chosen": {"messages": [...]}, "rejected": {"messages": [...]}}
```

---

### Group Relative Policy Optimization (GRPO)

Generate multiple responses per prompt and learn from their relative quality.

```shell
mlx_lm_lora.train \
--model Goekdeniz-Guelmez/Josiefied-Qwen2.5-0.5B-Instruct-abliterated-v1 \
--train \
--train-mode grpo \
--data mlx-community/gsm8k \
--group-size 4 \
--epsilon 1e-4 \
--max-completion-length 512 \
--temperature 0.8 \
--reward-functions "accuracy_reward,format_reward" \
--reward-weights "[0.7, 0.3]"
```

**Key Parameters:**

- `--group-size`: Number of generations per prompt (default: 4)
- `--epsilon`: Numerical stability constant (default: 1e-4)
- `--max-completion-length`: Max generation length (default: 512)
- `--temperature`: Sampling temperature (default: 0.8)
- `--reward-functions`: Comma-separated reward function names
- `--reward-functions-file`: Path to custom reward functions file
- `--reward-weights`: JSON list of weights for each reward function
- `--grpo-loss-type`: Loss variant - `grpo`, `bnpo`, or `dr_grpo`

**Dataset Format:**

```jsonl
{"prompt": "Math problem", "answer": "42"}
{"prompt": "Question", "answer": "Response", "system": "You are helpful"}
{"prompt": "Question", "answer": "Response", "type": "math"}
```

**Custom Reward Functions:**
Create a Python file with reward functions:

```python
# my_rewards.py
from mlx_lm_lora.reward_functions import register_reward_function

@register_reward_function()
def my_custom_reward(prompt, completion, reference_answer, **kwargs):
    """Custom reward function"""
    # Your logic here
    return score  # float between 0 and 1
```

Then use: `--reward-functions-file ./my_rewards.py --reward-functions "my_custom_reward"`

---

### Group Sequence Policy Optimization (GSPO)

GSPO extends GRPO with importance sampling at token or sequence level for improved sample efficiency.

```shell
mlx_lm_lora.train \
--model Goekdeniz-Guelmez/Josiefied-Qwen2.5-0.5B-Instruct-abliterated-v1 \
--train \
--train-mode grpo \
--grpo-loss-type grpo \
--importance-sampling-level token \
--group-size 4 \
--epsilon 1e-4 \
--temperature 0.8
```

**Key Parameters:**

- `--importance-sampling-level`: Choose `token`, `sequence`, or `None` (default: None)
- All other GRPO parameters apply

**Dataset Format:** Same as GRPO

---

### Decoupled Reward Group Relative Policy Optimization (Dr. GRPO)

Dr. GRPO decouples the reward computation from the policy optimization for more stable training.

```shell
mlx_lm_lora.train \
--model Goekdeniz-Guelmez/Josiefied-Qwen2.5-0.5B-Instruct-abliterated-v1 \
--train \
--train-mode grpo \
--grpo-loss-type dr_grpo \
--group-size 4 \
--epsilon 1e-4 \
--temperature 0.8
```

**Key Parameters:**

- `--grpo-loss-type dr_grpo`: Enables Dr. GRPO variant
- All other GRPO parameters apply

**Dataset Format:** Same as GRPO

---

### Decoupled Clip and Dynamic Sampling Policy Optimization (DAPO)

DAPO uses dual epsilon values for more flexible clipping in policy optimization.

```shell
mlx_lm_lora.train \
--model Goekdeniz-Guelmez/Josiefied-Qwen2.5-0.5B-Instruct-abliterated-v1 \
--train \
--train-mode grpo \
--epsilon 1e-4 \
--epsilon-high 1e-2 \
--group-size 4 \
--temperature 0.8
```

**Key Parameters:**

- `--epsilon`: Lower bound for clipping (default: 1e-4)
- `--epsilon-high`: Upper bound for clipping (uses epsilon value if not specified)
- All other GRPO parameters apply

**Dataset Format:** Same as GRPO

---

### Online DPO

Online preference optimization using a judge model or human feedback.

```shell
mlx_lm_lora.train \
--model Goekdeniz-Guelmez/Josiefied-Qwen2.5-0.5B-Instruct-abliterated-v1 \
--train \
--train-mode online_dpo \
--data ./online_data \
--judge mlx-community/Josiefied-Qwen2.5-7B-Instruct-abliterated-v2-4-bit \
--alpha 1e-5
```

**Key Parameters:**

- `--judge`: Judge model ID or "human" for human feedback
- `--alpha`: Learning rate for online updates (default: 1e-5)
- `--judge-config`: Additional configuration for judge model

**Dataset Format:**

```jsonl
{"prompt": [{"role": "user", "content": "Question"}]}
{"messages": [{"role": "user", "content": "Question"}]}
```

---

### eXtended Preference Optimization (XPO)

XPO extends online DPO with additional preference learning mechanisms.

```shell
mlx_lm_lora.train \
--model Goekdeniz-Guelmez/Josiefied-Qwen2.5-0.5B-Instruct-abliterated-v1 \
--train \
--train-mode xpo \
--data ./xpo_data \
--judge mlx-community/Josiefied-Qwen2.5-7B-Instruct-abliterated-v2-4-bit \
--alpha 1e-5 \
--beta 0.1
```

**Key Parameters:**

- `--judge`: Judge model ID or "human"
- `--alpha`: Online learning rate (default: 1e-5)
- `--beta`: KL penalty strength (default: 0.1)
- `--judge-config`: Additional judge configuration

**Dataset Format:** Same as Online DPO

---

### Reinforcement Learning from Human Feedback REINFORCE (RLHF REINFORCE)

Full RLHF REINFORCE pipeline with reward model and policy optimization Ziegler style.

```shell
mlx_lm_lora.train \
--model Goekdeniz-Guelmez/Josiefied-Qwen2.5-0.5B-Instruct-abliterated-v1 \
--train \
--train-mode rlhf-reinforce \
--data ./rlhf_data \
--judge mlx-community/reward-model \
--alpha 1e-5 \
--beta 0.1 \
--group-size 4
```

**Key Parameters:**

- `--judge`: Reward model ID
- `--alpha`: Policy learning rate (default: 1e-5)
- `--beta`: KL penalty strength (default: 0.1)
- `--group-size`: Number of samples for policy optimization (default: 4)

**Dataset Format:** Same as Online DPO

---

## Other Features

### Synthetic Dataset Creation

This feature makes it able to use mlx-lm's powerfull batch genebrate to create a synthetic datasets using a teacher model, this can be used for knowledge distiliation, etc., and is a powerfull tool to create custom model, fuly locally.

#### SFT

With this you can create a synthetic SFT dataset using a teacher model. this creates multible files, the first file is a JSONL file that has the generated samples in it, the next ones are parquet verison for HF compatibility. Example:

```shell
python -m mlx_lm_lora.synthetic_sft \
--dataset-path Goekdeniz-Guelmez/Josiefication-prompts-online-po \
--model mlx-community/Josiefied-Qwen3-4B-Instruct-2507-abliterated-v1-8bit \
--output-dir ./sft_dataset \
--num-samples 1000 \
--valid-split 0.01 \
--batch-size 16 \
--max-tokens 4096
```

**Dataset Format:**

```jsonl
{"prompt": "Question"}
{"prompt": "Question"}
{"prompt": "Question"}
```

#### Preference

With this you can create a synthetic DPO flatt-dataset using a teacher model. this creates multible files just like sft. Example:

```shell
python -m mlx_lm_lora.synthetic_dpo \
--dataset-path Goekdeniz-Guelmez/Josiefication-prompts-online-po \
--base-model mlx-community/Qwen3-4B-Instruct-2507-4bit \
--teacher-model mlx-community/Qwen3-4B-Instruct-2507-4bit \
--system-promtp "can be a normal string or the path to a .txt file for longer prompts"t \
--output-dir ./dpo_dataset \
--num-samples 10000 \
--valid-split 0.0001 \
--test-split 0.2 \
--batch-size 16 \
--max-tokens 8192
```

**Dataset Format:** Same as abouve

### Training Your Custom Preference Model

This feature adds a second training stage on top of the judge (preference) stage. A reward model thats scores the policy’s generations and the policy is updated with a KL‑penalised PPO‑style loss.

1. Collect preference data  →  judge‑mode (online DPO) →  reward model
2. Run RLHF (policy optimisation) using the reward model → final policy

```shell
python -m mlx_lm_lora.train_judge \
--model Goekdeniz-Guelmez/Josiefied-Qwen3-0.6B-abliterated-v1 \
--train-type full \
--optimizer adamw \
--steps-per-report 1 \
--iters 50 \
--max-seq-length 1024 \
--adapter-path /Users/Goekdeniz.Guelmez@computacenter.com/Library/CloudStorage/OneDrive-COMPUTACENTER/Desktop/test \
--data mlx-community/Human-Like-DPO \
--gradient-accumulation-steps 1
```

**Dataset Format:** Same as DPO (with `prompt`, `chosen`, and `rejected` pairs).

---

## Configuration

### Core Training Parameters

```shell
# Model and data
--model <model_path>              # Model path or HF repo
--data <data_path>                # Dataset path or HF dataset name
--train-type lora                 # lora, dora, or full
--train-mode sft                  # sft, dpo, cpo, orpo, grpo, etc.

# Training schedule
--batch-size 4                    # Batch size
--iters 1000                      # Training iterations
--epochs 3                        # Training epochs (ignored if iters set)
--learning-rate 1e-5              # Learning rate
--gradient-accumulation-steps 1   # Gradient accumulation

# Model architecture
--num-layers 16                   # Layers to fine-tune (-1 for all)
--max-seq-length 2048            # Maximum sequence length

# LoRA parameters
--lora-parameters '{"rank": 8, "dropout": 0.0, "scale": 10.0}'

# Optimization
--optimizer adam                  # adam, adamw, qhadam, muon
--lr-schedule cosine             # Learning rate schedule
--grad-checkpoint                # Enable gradient checkpointing

# Quantization
--load-in-4bits                  # 4-bit quantization
--load-in-6bits                  # 6-bit quantization  
--load-in-8bits                  # 8-bit quantization

# Monitoring
--steps-per-report 10            # Steps between loss reports
--steps-per-eval 200             # Steps between validation
--val-batches 25                 # Validation batches (-1 for all)
--wandb project_name             # WandB logging

# Checkpointing
--adapter-path ./adapters        # Save/load path for adapters
--save-every 100                 # Save frequency
--resume-adapter-file <path>     # Resume from checkpoint
--fuse                           # Fuse and save trained model
```

### Algorithm-Specific Parameters

**Preference Optimization Methods:**

**DPO/CPO:**

```shell
--beta 0.1                        # KL penalty strength
--dpo-cpo-loss-type sigmoid       # sigmoid, hinge, ipo, dpop
--delta 50.0                      # Margin for hinge loss
--reference-model-path <path>     # Reference model path
```

**ORPO:**

```shell
--beta 0.1                        # Temperature parameter
--reward-scaling 1.0              # Reward scaling factor
```

**Group-Based Methods:**

**GRPO (Base):**

```shell
--group-size 4                    # Generations per prompt
--epsilon 1e-4                    # Numerical stability constant
--temperature 0.8                 # Sampling temperature
--max-completion-length 512       # Max generation length
--reward-functions "func1,func2"  # Comma-separated reward functions
--reward-functions-file <path>    # Custom reward functions file
--reward-weights "[0.5, 0.5]"    # JSON list of reward weights
--grpo-loss-type grpo             # grpo, bnpo, dr_grpo
```

**GSPO (GRPO + Importance Sampling):**

```shell
--importance-sampling-level token # token, sequence, or None
# Plus all GRPO parameters
```

**Dr. GRPO (Decoupled Rewards):**

```shell
--grpo-loss-type dr_grpo         # Enable Dr. GRPO variant
# Plus all GRPO parameters
```

**DAPO (Dynamic Clipping):**

```shell
--epsilon 1e-4                   # Lower bound for clipping
--epsilon-high 1e-2              # Upper bound for clipping
# Plus all GRPO parameters
```

**Online Methods:**

**Online DPO:**

```shell
--judge <model_id>               # Judge model or "human"
--alpha 1e-5                     # Online learning rate
--beta 0.1                       # KL penalty strength
--judge-config '{}'              # Additional judge configuration
```

**XPO (Extended Preference Optimization):**

```shell
--judge <model_id>               # Judge model or "human"
--alpha 1e-5                     # Online learning rate
--beta 0.1                       # KL penalty strength
--judge-config '{}'              # Judge configuration
# Plus additional XPO-specific parameters
```

**RLHF Reinforce (Full Pipeline):**

```shell
--judge <reward_model_id>        # Reward model
--alpha 1e-5                     # Policy learning rate
--beta 0.1                       # KL penalty strength
--group-size 4                   # Samples for policy optimization
--judge-config '{}'              # Reward model configuration
```

---

## Dataset Formats

### Local Datasets

Place JSONL files in a directory:

```text
data/
├── train.jsonl
├── valid.jsonl
└── test.jsonl
```

### Hugging Face Datasets

```shell
mlx_lm_lora.train --data "Goekdeniz-Guelmez/Josiefied-Qwen2.5-0.5B-Instruct-abliterated-v1" --train
```

### Custom Dataset Keys

Configure custom field names:

```shell
--text-feature "content"          # For text datasets
--chat-feature "conversation"     # For chat datasets
--prompt-feature "question"       # For prompt-completion
--completion-feature "answer"     # For prompt-completion
--chosen-feature "preferred"      # For preference datasets
--rejected-feature "dispreferred" # For preference datasets
--system-feature "instruction"    # For system messages
```

### Dataset Examples by Training Mode

**SFT - Chat Format:**

```jsonl
{"messages": [
  {"role": "system", "content": "You are helpful"},
  {"role": "user", "content": "What is 2+2?"},
  {"role": "assistant", "content": "4"}
]}
```

**SFT - Completion Format:**

```jsonl
{"prompt": "What is 2+2?", "completion": "2+2 equals 4"}
```

**SFT - Text Format:**

```jsonl
{"text": "The complete text for language modeling"}
```

**DPO/CPO Format:**

```jsonl
{"prompt": "Explain AI", "chosen": "AI is artificial intelligence", "rejected": "AI is magic"}
```

**ORPO Format:**

```jsonl
{"prompt": "What is AI?", "chosen": "Good explanation", "rejected": "Bad explanation", "preference_score": 0.8}
```

**GRPO Format:**

```jsonl
{"prompt": "Solve: 2+2=?", "answer": "4", "system": "You are a math tutor"}
```

**RLHF Format:**

```jsonl
{"prompt": [{"role": "user", "content": "Question"}]}
```

or:

```jsonl
{"prompt": "Question"}
```

---

## Memory Optimization

### Quantization (QLoRA)

Use quantized models to reduce memory usage:

```shell
# 4-bit quantization (most memory efficient)
mlx_lm_lora.train --model <model> --load-in-4bits --train

# 6-bit quantization (balanced)
mlx_lm_lora.train --model <model> --load-in-6bits --train

# 8-bit quantization (higher quality)
mlx_lm_lora.train --model <model> --load-in-8bits --train
```

### Other Memory Reduction Techniques

```shell
# Reduce batch size
--batch-size 1

# Train fewer layers
--num-layers 8

# Enable gradient checkpointing
--grad-checkpoint

# Reduce sequence length
--max-seq-length 1024

# Use gradient accumulation
--gradient-accumulation-steps 4 --batch-size 1
```

### LoRA Configuration for Memory

```shell
# Smaller LoRA rank
--lora-parameters '{"rank": 4, "dropout": 0.1, "scale": 10.0}'

# Train specific layers only
--num-layers 8
```

---

## Evaluation & Generation

### Evaluation

Evaluate on test set:

```shell
mlx_lm_lora.train \
--model <model_path> \
--adapter-path <adapter_path> \
--data <data_path> \
--test \
--test-batches 500
```

### Generation

Use `mlx-lm` for generation with trained adapters:

```shell
mlx_lm.generate \
--model <model_path> \
--adapter-path <adapter_path> \
--prompt "Your prompt here" \
--max-tokens 100 \
--temperature 0.7
```

### Fusing Adapters

Merge LoRA weights into base model:

```shell
mlx_lm_lora.train \
--model <model_path> \
--adapter-path <adapter_path> \
--fuse
```

---

## Advanced Features

### Learning Rate Schedules

```shell
--lr-schedule cosine              # Cosine annealing
--lr-schedule linear              # Linear decay
--lr-schedule constant            # Constant rate
```

### Multiple Optimizers

```shell
--optimizer adam                  # Adam optimizer
--optimizer adamw                 # AdamW with weight decay
--optimizer qhadam               # Quasi-hyperbolic Adam
--optimizer muon                 # Muon optimizer
```

### Reward Function System (GRPO)

List available reward functions:

```shell
mlx_lm_lora.train --list-reward-functions
```

Use multiple reward functions:

```shell
--reward-functions "accuracy_reward,format_reward,length_reward" \
--reward-weights "[0.5, 0.3, 0.2]"
```

### WandB Integration

```shell
--wandb my_project_name
```

---

## Training Method Comparison

| Method | Type | Reference Model | Judge Model | Multiple Generations | Key Benefit |
|--------|------|-----------------|-------------|---------------------|-------------|
| SFT | Supervised | ❌ | ❌ | ❌ | Simple, fast training |
| DPO | Preference | ✅ | ❌ | ❌ | No reward model needed |
| CPO | Preference | ✅ | ❌ | ❌ | Better for structured tasks |
| ORPO | Preference | ❌ | ❌ | ❌ | Monolithic optimization |
| GRPO | Policy | ❌ | ❌ | ✅ | Group-based learning |
| GSPO | Policy | ❌ | ❌ | ✅ | Importance sampling |
| Dr. GRPO | Policy | ❌ | ❌ | ✅ | Decoupled rewards |
| DAPO | Policy | ❌ | ❌ | ✅ | Dynamic clipping |
| Online DPO | Online RL | ❌ | ✅ | ❌ | Real-time feedback |
| XPO | Online RL | ❌ | ✅ | ❌ | Extended preferences |
| RLHF Reinforce | Online RL | ❌ | ✅ | ✅ | Full RL pipeline |

---

## Example Commands for All Methods

### Basic Methods

```shell
# SFT
mlx_lm_lora.train --model <model> --train-mode sft --data <data>

# DPO
mlx_lm_lora.train --model <model> --train-mode dpo --data <data> --beta 0.1

# CPO
mlx_lm_lora.train --model <model> --train-mode cpo --data <data> --beta 0.1

# ORPO
mlx_lm_lora.train --model <model> --train-mode orpo --data <data> --beta 0.1
```

### Group-Based Methods

```shell
# GRPO
mlx_lm_lora.train --model <model> --train-mode grpo --data <data> --group-size 4

# GSPO (GRPO with importance sampling)
mlx_lm_lora.train --model <model> --train-mode grpo --data <data> \
--importance-sampling-level token --group-size 4

# Dr. GRPO
mlx_lm_lora.train --model <model> --train-mode grpo --data <data> \
--grpo-loss-type dr_grpo --group-size 4

# DAPO
mlx_lm_lora.train --model <model> --train-mode grpo --data <data> \
--epsilon 1e-4 --epsilon-high 1e-2 --group-size 4
```

### Online Methods

```shell
# Online DPO
mlx_lm_lora.train --model <model> --train-mode online_dpo --data <data> \
--judge <judge_model> --alpha 1e-5

# XPO
mlx_lm_lora.train --model <model> --train-mode xpo --data <data> \
--judge <judge_model> --alpha 1e-5

# RLHF Reinforce
mlx_lm_lora.train --model <model> --train-mode rlhf-reinforce --data <data> \
--judge <reward_model> --alpha 1e-5 --group-size 4
```

---

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size, use quantization, enable gradient checkpointing
2. **Slow Training**: Increase batch size, reduce validation frequency
3. **Poor Quality**: Increase LoRA rank, train more layers, check data quality
4. **Convergence Issues**: Adjust learning rate, try different optimizers

### Memory Usage Guidelines

| Model Size | Recommended Settings |
|------------|---------------------|
| 1-3B | `--batch-size 4 --num-layers 16` |
| 7B | `--batch-size 2 --num-layers 8 --load-in-8bits` |
| 13B+ | `--batch-size 1 --num-layers 4 --load-in-4bits --grad-checkpoint` |

---

## Example Configurations

### Basic LoRA Fine-tuning

```yaml
model: Goekdeniz-Guelmez/Josiefied-Qwen2.5-0.5B-Instruct-abliterated-v1
train: true
data: ./my_data
train_type: lora
train_mode: sft
batch_size: 4
learning_rate: 1e-5
iters: 1000
lora_parameters:
  rank: 8
  dropout: 0.0
  scale: 10.0
```

### DPO Training

```yaml
model: Goekdeniz-Guelmez/Josiefied-Qwen2.5-0.5B-Instruct-abliterated-v1
train: true
data: ./preference_data
train_mode: dpo
beta: 0.1
dpo_cpo_loss_type: sigmoid
batch_size: 2
learning_rate: 5e-6
iters: 500
```

### GRPO with Custom Rewards

```yaml
model: Goekdeniz-Guelmez/Josiefied-Qwen2.5-0.5B-Instruct-abliterated-v1
train: true
data: ./grpo_data
train_mode: grpo
group_size: 4
temperature: 0.8
reward_functions: "accuracy_reward,format_reward"
reward_weights: [0.7, 0.3]
max_completion_length: 512
```

---

## MLX-LM-LoRA is trusted by teams and industry leaders such as:

<p align="center">
  <img src="https://github.com/Goekdeniz-Guelmez/mlx-lm-lora/blob/main/logos/macpaw.png" alt="logo" width="300"/>
</p>

---

## Citing MLX-LM-LoRA

```bibtex
@software{MLX-LM-LoRA,
  author = {Gökdeniz Gülmez},
  title = {{MLX-LM-LoRA}: Train LLMs on Apple silicon with MLX and the Hugging Face Hub},
  url = {https://github.com/Goekdeniz-Guelmez/mlx-lm-lora},
  version = {0.1.0},
  year = {2025},
}
```
