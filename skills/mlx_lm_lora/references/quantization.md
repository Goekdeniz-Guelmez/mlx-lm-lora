# Quantized loading and QAT

Read this reference when translating a quantization request. Quantized loading,
adapter type, and quantization-aware training are independent choices.

## Quantized model loading

| User wording | Config | Backend loading settings |
| --- | --- | --- |
| 4-bit / 4bit | `load_in_4bits: true` | 4 bits, group size 128 |
| 6-bit / 6bit | `load_in_6bits: true` | 6 bits, group size 128 |
| 8-bit / 8bit | `load_in_8bits: true` | 8 bits, group size 128 |
| MXFP4 / mixed FP4 | `load_in_mxfp4: true` | 4 bits, group size 32, mode `mxfp4` |

Set at most one loading flag. Preserve the user's model ID; if it clearly names
an already-quantized checkpoint, do not add a second quantization request.
LoRA and DoRA still map to `train_type`, not to a loading flag.

```json
{
  "model": "org/model",
  "data": "org/instruction-dataset",
  "train": true,
  "train_mode": "sft",
  "train_type": "lora",
  "load_in_mxfp4": true,
  "iters": 100
}
```

## Quantization-aware training

The MCP entrypoint enables QAT for SFT, DPO, ORPO, and DSLA. Other modes do not
wire QAT into training. QAT exposes fake-quantized weights in forward passes
with straight-through gradients while preserving optimizer weights.

| Field | Default | Meaning |
| --- | --- | --- |
| `qat_enable` | `false` | Enable QAT hooks |
| `qat_bits` | `8` | Integer in [2, 16] |
| `qat_group_size` | `64` | Nonnegative integer; 0 means per-tensor |
| `qat_mode` | `"affine"` | Only accepted enum value |
| `qat_start_step` | `1` | Install hooks after reaching this optimizer update |
| `qat_interval` | `1` | Accepted shared field; currently unused for hook timing |

The implementation uses symmetric fake quantization despite the accepted
`qat_mode` label. Once installed, hooks quantize eligible linear forwards;
`qat_interval` does not make them intermittent. DSLA scopes hooks to policy
modules and restores them after the run; its reference stays unprojected.
Do not describe QAT as producing an automatically validated deployment format.

```json
{
  "model": "org/model",
  "data": "org/instruction-dataset",
  "train": true,
  "train_mode": "sft",
  "train_type": "lora",
  "qat_enable": true,
  "qat_bits": 4,
  "qat_group_size": 64,
  "qat_start_step": 10,
  "iters": 100
}
```

Choose bit width and grouping for the requested deployment target. Compare
against a non-QAT run and evaluate the quantized deployment artifact itself.
For output/fusion defaults, see [config.md](config.md).
