# Supervised fine-tuning

Use `train_mode: "sft"` for next-token supervised fine-tuning. The SFT
implementation supports three loss algorithms selected with
`sft_loss_type`:

| Algorithm | `sft_loss_type` | Objective |
| --- | --- | --- |
| Standard next-token NLL | `"nll"` | Masked cross-entropy over valid target tokens |
| Chunked NLL | `"chunked_nll"` | Same NLL objective, accumulated over fixed 256-token loss chunks |
| Dynamic Fine-Tuning | `"dft"` | NLL weighted by detached target-token probabilities |

## Basic configuration

The examples use MCP field names. Keep `data` as a Hugging Face dataset
repository ID and include `train: true`.

```json
{
  "model": "Qwen/Qwen3.5-0.8B",
  "data": "mlx-community/wikisql",
  "train": true,
  "train_type": "lora",
  "train_mode": "sft",
  "sft_loss_type": "nll",
  "iters": 100,
  "batch_size": 4,
  "learning_rate": 1e-5,
  "max_seq_length": 2048
}
```

Map natural-language “steps” or “iterations” to `iters`. Use `epochs` only
when an epoch budget is requested; if both are present, `iters` takes
precedence.

## SFT settings

| Field | Default | Use |
| --- | ---: | --- |
| `sft_loss_type` | `"nll"` | `"nll"`, `"chunked_nll"`, or `"dft"` |
| `mask_prompt` | `false` | Score only assistant/completion tokens when `true` |
| `train_type` | `"lora"` | `"lora"`, `"dora"`, or `"full"` |
| `optimizer` | `"adam"` | `"adam"`, `"adamw"`, or `"muon"` |
| `optimizer_config` | optimizer-specific empty mapping | Extra optimizer keyword arguments |
| `learning_rate` | `1e-5` | Positive optimizer learning rate |
| `lr_schedule` | `null` | MLX-LM schedule expression |
| `batch_size` | `4` | Per-step batch size; keep it divisible by worker count |
| `iters` | `null` | Number of optimizer iterations |
| `epochs` | `null` | Converted to iterations when `iters` is omitted |
| `gradient_accumulation_steps` | `1` | Accumulate minibatches before updating |
| `max_seq_length` | `2048` | Maximum tokenized example length |
| `num_layers` | `-1` | Number of LoRA/DoRA layers; `-1` means all |
| `lora_parameters` | `rank: 8`, `dropout: 0.0`, `scale: 10.0` | LoRA/DoRA adapter settings |
| `grad_checkpoint` | `false` | Trade compute for lower activation memory |
| `efficient_long_context` | `false` | Process examples through 512-token sequential cached chunks |
| `val_batches` | `25` | Validation batches; `-1` means all |
| `steps_per_report` | `10` | Training-log interval |
| `steps_per_eval` | `200` | Validation interval |
| `save_every` | `100` | Adapter checkpoint interval |
| `resume_adapter_file` | `null` | Tenant-local adapter to resume from |
| `adapter_path` | server-selected | Tenant-local output directory |
| `seed` | `0` | Random seed |
| `wandb` | `null` | Optional Weights & Biases project name |
| `test` | `false` | Evaluate the test split after training |
| `test_batches` | `500` | Test batches; `-1` means all |
| `fuse` | `true` | Merge and save the trained adapter with the base model |

For quantized loading, set at most one of `load_in_4bits`, `load_in_6bits`, or
`load_in_8bits` to `true`. These control model loading and are independent of
`train_type`.

## Loss algorithms

### Standard NLL

`sft_loss_type: "nll"` is masked next-token cross-entropy. It is the default
and the baseline against which the other SFT losses should be compared.

```json
{
  "model": "org/model",
  "data": "org/instruction-dataset",
  "train": true,
  "train_type": "lora",
  "train_mode": "sft",
  "sft_loss_type": "nll",
  "mask_prompt": true,
  "iters": 100
}
```

With `mask_prompt: true`, the loss is applied only to assistant/completion
tokens for chat and prompt/completion records. Do not set prompt masking for a
plain `text` dataset; the loader rejects that combination.

### Chunked NLL

`sft_loss_type: "chunked_nll"` computes the same masked NLL while accumulating
the loss in fixed 256-token chunks. The chunk size is an implementation
constant; there is no MCP setting for changing it. This can reduce the memory
pressure from loss evaluation on long sequences, but it does not change the
training target or tokenizer behavior.

```json
{
  "model": "org/long-context-model",
  "data": "org/long-instruction-dataset",
  "train": true,
  "train_type": "lora",
  "train_mode": "sft",
  "sft_loss_type": "chunked_nll",
  "max_seq_length": 8192,
  "efficient_long_context": true,
  "grad_checkpoint": true,
  "batch_size": 1,
  "gradient_accumulation_steps": 8,
  "iters": 100
}
```

For very long sequences, combine chunked NLL with a small batch, gradient
accumulation, gradient checkpointing, and `efficient_long_context`. Establish
an ordinary NLL baseline before attributing a quality change to chunking.

### Dynamic Fine-Tuning (DFT)

`sft_loss_type: "dft"` multiplies each target-token NLL contribution by the
detached probability assigned to that target. The weight is detached from the
gradient path, so DFT changes the contribution of examples/tokens without
creating a second trainable objective.

```json
{
  "model": "org/model",
  "data": "org/instruction-dataset",
  "train": true,
  "train_type": "lora",
  "train_mode": "sft",
  "sft_loss_type": "dft",
  "mask_prompt": true,
  "learning_rate": 1e-5,
  "iters": 100
}
```

DFT has no additional algorithm-specific configuration fields. Use the same
data and seed as the NLL baseline when comparing losses.

## Dataset formats

The default SFT loader accepts one of these record shapes:

### Prompt and completion

```json
{
  "prompt": "Explain gradient descent.",
  "completion": "Gradient descent is an optimization method..."
}
```

The loader wraps the values as a user message and an assistant message using
the model's chat template. `mask_prompt: true` scores only the completion.

### Chat messages

```json
{
  "messages": [
    {"role": "user", "content": "Explain gradient descent."},
    {"role": "assistant", "content": "Gradient descent is..."}
  ]
}
```

The optional `tools` field is passed to the tokenizer chat template. With
`mask_prompt: true`, all messages except the final assistant message are
excluded from the loss.

### Plain text

```json
{
  "text": "A complete language-modeling training example."
}
```

Plain text is trained as a continuous language-modeling sequence and always
includes the full sequence in the loss. It cannot be combined with
`mask_prompt: true`.

The tokenizer adds an EOS token to plain-text examples when one is not already
present. Keep records compatible with the model tokenizer and avoid mixing
incompatible chat templates in one run.

## Quantization-aware training

QAT is available on the SFT path when the deployment target is quantized:

| Field | Default | Use |
| --- | ---: | --- |
| `qat_enable` | `false` | Enable straight-through fake-quantized linear forwards |
| `qat_bits` | `8` | Projection bit width; implementation accepts 2–16 |
| `qat_group_size` | `64` | Last-dimension group size; smaller groups are more local |
| `qat_mode` | `"affine"` | The only supported mode |
| `qat_start_step` | `1` | First optimizer step with QAT projection |
| `qat_interval` | `1` | Apply the projection every N optimizer steps |

```json
{
  "model": "org/model",
  "data": "org/instruction-dataset",
  "train": true,
  "train_type": "lora",
  "train_mode": "sft",
  "sft_loss_type": "nll",
  "qat_enable": true,
  "qat_bits": 4,
  "qat_group_size": 64,
  "qat_mode": "affine",
  "qat_start_step": 1,
  "qat_interval": 1,
  "iters": 100
}
```

QAT keeps optimizer weights in full precision while exposing fake-quantized
weights in forward passes. Compare a non-QAT run and evaluate the actually
quantized deployment artifact; QAT is not a substitute for deployment testing.

## Best practices by SFT algorithm

### Standard NLL

- Use NLL as the reference baseline for data cleaning, learning-rate tuning,
  sequence length, and adapter capacity.
- Decide explicitly whether prompt tokens should contribute to loss. For
  instruction tuning, `mask_prompt: true` often makes the objective clearer;
  for continued pretraining, use plain text without masking.
- Keep examples complete, consistently templated, and within the intended
  context budget. Monitor token counts and validation perplexity, not only
  training loss.
- Start with LoRA for inexpensive experiments and use `full` only when the
  hardware budget and required capacity justify it.

### Chunked NLL

- Choose chunked NLL for long examples when ordinary NLL is memory-constrained;
  it is an efficiency variant, not a different supervision signal.
- Use `batch_size: 1`, gradient accumulation, checkpointing, and
  `efficient_long_context` together when needed, then increase throughput one
  control at a time.
- Verify that truncation is not silently removing the completion. A smaller
  `max_seq_length` can improve stability while eliminating the target signal.
- Compare loss and generation quality against ordinary NLL at the same seed and
  data order before adopting it as the default.

### DFT

- Treat DFT as an experimental weighting strategy. Establish a clean NLL
  baseline first and compare validation loss, instruction-following, and
  calibration on held-out data.
- Use conservative learning rates and watch for under-training of difficult or
  low-probability tokens; detached probability weighting can reduce their
  contribution.
- Keep the data distribution and seed fixed when comparing DFT with NLL. Do
  not add an unimplemented DFT-specific parameter to the MCP config.

### SFT with QAT

- First reach the target quality without QAT, then enable QAT and compare the
  post-quantization model. This separates training issues from quantization
  issues.
- Match `qat_bits` and `qat_group_size` to the deployment quantizer. Smaller
  groups may improve fidelity but increase overhead.
- If QAT destabilizes training, delay it with `qat_start_step`, apply it less
  frequently with `qat_interval`, or lower the learning rate.

## Operational checklist

Before starting an SFT job:

1. Confirm the mode is `train_mode: "sft"` and choose exactly one supported
   `sft_loss_type`.
2. Confirm the dataset uses `messages`, `prompt`/`completion`, or `text` and
   that `mask_prompt` is compatible with the chosen format.
3. Set `max_seq_length` high enough to retain the supervised completion, then
   use memory controls if the run does not fit.
4. Choose `iters` or `epochs`; if both are provided, `iters` wins.
5. Keep `batch_size` divisible by the distributed worker count.
6. Evaluate held-out perplexity and task behavior before fusing the adapter.
