# Quantization

Map explicit quantization requests to the boolean fields understood by the
training backend:

| User wording | MCP field |
| --- | --- |
| 4-bit / 4bit | `load_in_4bits: true` |
| 6-bit / 6bit | `load_in_6bits: true` |
| 8-bit / 8bit | `load_in_8bits: true` |
| MXFP4 / mixed FP4 | `load_in_mxfp4: true` |

Preserve the model identifier exactly as the user gave it. If it already
clearly identifies a pre-quantized model, do not add another quantization
field. If the user gives an unquantized model ID and explicitly requests a
quantization level, set the matching field.

Do not set more than one of the `load_in_4bits`, `load_in_6bits`,
`load_in_8bits`, and `load_in_mxfp4` fields. Do not confuse LoRA (`train_type`) with quantization;
they are independent settings.

MXFP4 loads with 4 bits, group size 32, and mode `mxfp4`; ordinary bit-width
loading uses group size 128. QAT simulates quantization during optimization
and is supported for SFT, DPO, ORPO, and DSLA. Use `qat_enable`, `qat_bits`
(2–16), `qat_group_size` (0 for per-tensor), `qat_mode: "affine"`,
`qat_start_step`, and `qat_interval`. DSLA has its own projection schedule;
see [dsla.md](dsla.md).
