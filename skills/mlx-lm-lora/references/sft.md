# SFT: pretraining and supervised fine-tuning

`train_mode: "sft"` runs causal language-model training and supports both
pretraining on text and supervised fine-tuning on instruction/chat data.
For continued or domain pretraining, use plain `text` rows with
`sft_loss_type: "nll"` and no prompt masking. Use `train_type: "full"` when
all model weights should train, or the requested adapter method. The MCP
loads an existing checkpoint; this does not imply random-initialization
training support. Never reject a pretraining request merely because the
mode is named SFT.

## Data and settings

Rows may contain `text`, `messages`, or `prompt`/`completion`. Use `mask_prompt` only with chat/completion rows, not plain text. Set `sft_loss_type` to `nll` (default), `chunked_nll` or `dft` in `extra_config`.

## MCP example

Pass these flat arguments to `start_training`; use the same arguments with
`check_training_config` first, and require successful `validate_model_support`
and `inspect_dataset` preflights before starting. Hub IDs below are illustrative.

```json
{
  "model": "org/policy-model",
  "data": "org/training-dataset",
  "train_mode": "sft",
  "train_type": "lora",
  "iters": 100,
  "batch_size": 1,
  "num_layers": 1,
  "max_seq_length": 512,
  "fuse": false,
  "extra_config": "{\"sft_loss_type\": \"nll\", \"steps_per_report\": 10}"
}
```

## Best practices

Use NLL as the baseline. Check tokenizer/chat-template compatibility, response termination and the amount of supervised text retained after truncation. Compare validation loss and task generations, not training loss alone. For chunked NLL, check equivalence on a short baseline before depending on its memory behavior. DFT weights NLL by detached target probability; compare it with NLL using the same data, seed and budget because hard low-probability tokens receive less weight.

## Memory efficiency

Use LoRA, a small physical batch, suitable rank/layers and gradient checkpointing. `chunked_nll` evaluates loss in fixed 256-token chunks; it does not bound every model activation. `efficient_long_context` separately selects fixed 512-token sequential chunks and may change gradient flow. QAT is wired but is not a low-memory loading mode.

See the [shared memory guide](memory-efficiency.md) for tuning order and the
[MCP contract](config.md) for defaults, monitoring and iteration-budget limits.
