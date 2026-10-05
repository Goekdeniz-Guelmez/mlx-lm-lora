# Supervised fine-tuning

Use `train_mode: "sft"` for next-token supervision. Read
[datasets.md](datasets.md) for `messages`, `prompt`/`completion`, and plain
`text` records, and [config.md](config.md) for shared training settings.

## Loss selection

| `sft_loss_type` | Behavior |
| --- | --- |
| `"nll"` | Default masked next-token cross-entropy |
| `"chunked_nll"` | Same NLL, accumulating cross-entropy in fixed 256-token chunks |
| `"dft"` | NLL weighted by detached target-token probabilities |

`mask_prompt` defaults to `false`. Set it to `true` when the user wants
completion-only or final-assistant supervision. Plain `text` data cannot be
combined with prompt masking.

Chunked NLL still computes a model forward and logits for its input. Its loss
chunk size is not configurable through MCP, and it is distinct from cached
long-context processing or recurrent chunk size. DFT has no additional
algorithm-specific fields; probability weighting reduces low-probability
contributions and should be compared with an NLL baseline.

## Example

```json
{
  "model": "org/model",
  "data": "org/instruction-dataset",
  "train": true,
  "train_type": "lora",
  "train_mode": "sft",
  "sft_loss_type": "nll",
  "mask_prompt": true,
  "iters": 100,
  "max_seq_length": 2048
}
```

## Memory, quantization, and evaluation

SFT supports QAT and `efficient_long_context`; read
[quantization.md](quantization.md) and [memory.md](memory.md) for their
independent settings. Retain the supervised completion when setting the
sequence limit; lowering memory by truncating away targets defeats the task.

Compare held-out loss/perplexity and task generations under the same data,
seed, and token budget. For DFT, compare the same evaluation objective across
runs rather than interpreting its weighted training loss as ordinary NLL.
Use `fuse: false` when the requested output is an adapter rather than a merged
model; the default fusion behavior is explained in [config.md](config.md).
