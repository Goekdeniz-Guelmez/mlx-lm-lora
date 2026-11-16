# Quantized loading and QAT

Map explicit policy loading requests to flat `quantize_bits`: 4, 6 or 8.
Use 0 for no additional conversion, including a model already quantized.
Preserve the user's model ID. The backend rejects re-quantizing a checkpoint
whose config already reports quantization. Loading conversion uses group size
128 for these three flags; it happens after loading, so it cannot solve a
source-model loading peak. Prefer a compatible pre-quantized model in that case.

`extra_config` also passes backend booleans `load_in_4bits`, `load_in_6bits`,
`load_in_8bits`, `load_in_mxfp4`. MXFP4 uses group size 32. Use one loading
scheme only; do not combine contradictory flags with `quantize_bits`.
Quantization is independent of `train_type`, and frozen low-bit bases normally
pair with LoRA/DoRA. Do not promise ordinary full-weight optimization of
packed quantized weights. Test architecture and projection compatibility.

## QAT best practices

QAT simulates quantized projections during training; it is not a substitute
for low-bit weight storage or an automatic memory reduction. It is wired for
SFT, DPO, ORPO and DSLA. The current CPO, FTPO and generated-mode dispatches
do not enable it. Put `qat_enable`, `qat_bits`, `qat_group_size`, `qat_mode`,
`qat_start_step`, `qat_interval` in `extra_config`.

Backend defaults are false, 8, 64, affine, 1, 1 respectively. The implemented
fake quantizer accepts 2–16 bits. Establish a non-QAT baseline, match the
deployment target, then evaluate the actually exported quantized model.
Delay QAT or lower learning rate when fake-quantized projections destabilize
training. Policy loading flags do not apply to separately loaded references
or judges. See [memory efficiency](memory-efficiency.md) for residency/export costs.
