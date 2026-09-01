# Quantization

Map explicit quantization requests to the boolean fields understood by the
training backend:

| User wording | MCP field |
| --- | --- |
| 4-bit / 4bit | `load_in_4bits: true` |
| 6-bit / 6bit | `load_in_6bits: true` |
| 8-bit / 8bit | `load_in_8bits: true` |

Preserve the model identifier exactly as the user gave it. If it already
clearly identifies a pre-quantized model, do not add another quantization
field. If the user gives an unquantized model ID and explicitly requests a
quantization level, set the matching field.

Do not set more than one of the `load_in_4bits`, `load_in_6bits`, and
`load_in_8bits` fields. Do not confuse LoRA (`train_type`) with quantization;
they are independent settings.
