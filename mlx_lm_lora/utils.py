import math
from pathlib import Path
from typing import Any, Optional, Tuple

import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.gguf import convert_to_gguf
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.tuner.utils import dequantize, linear_to_lora_layers, load_adapters
from mlx_lm.utils import (
    load,
    save_config,
    save_model,
)


def calculate_iters(train_set, batch_size, epochs) -> int:
    num_samples = len(train_set)
    batches_per_epoch = math.ceil(num_samples / batch_size)
    iters = epochs * batches_per_epoch
    print(
        f"[INFO] Calculated {iters} iterations from {epochs} epochs (dataset size: {num_samples}, batch size: {batch_size})"
    )
    return iters


def fuse_and_save_model(
    model: nn.Module,
    tokenizer: TokenizerWrapper,
    save_path: str = "fused_model",
    adapter_path: Optional[str] = None,
    de_quantize: Optional[bool] = False,
    export_gguf: Optional[bool] = False,
    gguf_path: Optional[str] = "ggml-model-f16.gguf",
) -> None:
    """
    Fuse fine-tuned adapters into the base model.

    Args:
        model: The MLX model to fuse adapters into.
        tokenizer: The tokenizer wrapper.
        save_path: The path to save the fused model.
        adapter_path: Path to the trained adapter weights and config.
        de_quantize: Generate a de-quantized model.
        export_gguf: Export model weights in GGUF format.
        gguf_path: Path to save the exported GGUF format model weights.
    """
    model.freeze()

    if adapter_path is not None:
        print(f"Loading adapters from {adapter_path}")
        model = load_adapters(model, adapter_path)

    args = vars(model.args)

    fused_linears = [
        (n, m.fuse(de_quantize=de_quantize))
        for n, m in model.named_modules()
        if hasattr(m, "fuse")
    ]

    if fused_linears:
        model.update_modules(tree_unflatten(fused_linears))

    if de_quantize:
        print("De-quantizing model")
        model = dequantize(model)
        args.pop("quantization", None)

    save_path_obj = Path(save_path)
    save_model(save_path_obj, model, donate_model=True)
    save_config(args, config_path=save_path_obj / "config.json")
    tokenizer.save_pretrained(save_path_obj)

    if export_gguf:
        model_type = args["model_type"]
        if model_type not in ["llama", "mixtral", "mistral"]:
            raise ValueError(
                f"Model type {model_type} not supported for GGUF conversion."
            )
        weights = dict(tree_flatten(model.parameters()))
        convert_to_gguf(save_path, weights, args, str(save_path_obj / gguf_path))


def from_pretrained(
    model: str,
    adapter_path: Optional[str] = None,
    lora_config: Optional[dict] = None,
    quantized_load: Optional[dict] = None,
) -> Tuple[nn.Module, Any]:
    """
    Load a model with LoRA adapters and optional quantization.
    Args:
        model: The base MLX model to load.
        lora_config: Configuration for LoRA adapters.
        quantized_load: If provided, the model will be loaded with quantization.
    Returns:
        Tuple[nn.Module, tokenizer]: The model with LoRA adapters loaded, and tokenizer.
    """
    print(f"Loading model {model}")
    model, tokenizer = load(model, adapter_path=adapter_path)
    args = vars(model.args) if hasattr(model, "args") else {}

    if lora_config is not None:
        print(f"Loading LoRA adapters with config: {lora_config}")
        rank = lora_config.get("rank", 8)
        dropout = lora_config.get("dropout", 0.0)
        scale = lora_config.get("scale", 10.0)
        use_dora = lora_config.get("use_dora", False)

        model.freeze()
        linear_to_lora_layers(
            model=model,
            num_layers=lora_config.get("num_layers", None),
            config={
                "rank": rank,
                "dropout": dropout,
                "scale": scale,
                "use_dora": use_dora,
            },
            use_dora=use_dora,
        )

    if quantized_load is not None:
        print(f"Quantizing model with {quantized_load['bits']} bits")
        if "quantization" in args:
            raise ValueError("Cannot quantize already quantized model")

        bits = quantized_load.get("bits", 4)
        group_size = quantized_load.get("group_size", 128)

        nn.quantize(model, bits=bits, group_size=group_size)

        if hasattr(model, "args"):
            model.args.quantization = {"group_size": group_size, "bits": bits}
            model.args.quantization_config = model.args.quantization

    return model, tokenizer
