import os
import math
import datetime
from pathlib import Path
from typing import Any, Optional, Tuple

import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.gguf import convert_to_gguf
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.tuner.utils import linear_to_lora_layers, load_adapters
from mlx_lm.utils import (
    load,
    save_config,
    save_model,
    dequantize_model
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
    from ._version import __version__
    model.freeze()
  
    if adapter_path is not None:
        print(f"Loading adapters from {adapter_path}")
        model = load_adapters(model, adapter_path)
  
    args = vars(model.args)
  
    fused_linears = [
        (n, m.fuse())
        for n, m in model.named_modules()
        if hasattr(m, "fuse")
    ]
  
    if fused_linears:
        model.update_modules(tree_unflatten(fused_linears))
  
    if de_quantize:
        print("De-quantizing model")
        model = dequantize_model(model)
        args.pop("quantization", None)
        args.pop("quantization_config", None)
  
    save_path_obj = Path(save_path)
    save_model(save_path_obj, model, donate_model=True)
    save_config(args, config_path=save_path_obj / "config.json")
    tokenizer.save_pretrained(save_path_obj)
    
    readme_content = f"""# MLX-LM-LoRA Model

This model was fine-tuned using [mlx-lm-lora](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora) version {__version__}.

## Model Details

- Base model: {args.get('model_name', 'Unknown')}
- Model type: {args.get('model_type', 'Unknown')}
- Training method: LoRA fine-tuning with MLX
- Fusion date: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Usage

This model can be loaded and used with the MLX framework.
"""
    
    with open(save_path_obj / "README.md", "w") as f:
        f.write(readme_content)
    
    print(f"Created README.md in {save_path}")
  
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
        mode = quantized_load.get("mode", "affine")

        nn.quantize(model, bits=bits, group_size=group_size, mode=mode)

        if hasattr(model, "args"):
            model.args.quantization = {"group_size": group_size, "bits": bits, "quant_method": mode}
            model.args.quantization_config = model.args.quantization

    return model, tokenizer


def push_to_hf(
    model_path: str,
    hf_repo: str,
    api_key: str,
    private: bool = False,
    commit_message: Optional[str] = None
) -> None:
    """
    Push the fused model to the Hugging Face Hub.

    Args:
        model_path: Local path of the model to upload.
        hf_repo: Name of the HF repo (format: username/repo_name).
        api_key: Hugging Face API token.
        private: Whether to create a private repository.
        commit_message: Custom commit message for the upload.
    """
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        raise ImportError(
            "The huggingface_hub package is required to push to the Hugging Face Hub. "
            "Please install it with `pip install huggingface_hub`."
        )

    print(f"Pushing model to {hf_repo}...")
    
    # Set the API token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = api_key
    api = HfApi()
    
    # Create the repo if it doesn't exist
    try:
        create_repo(hf_repo, private=private, token=api_key)
    except Exception as e:
        print(f"Repository creation failed or repository already exists: {e}")
    
    # Set default commit message if not provided
    if commit_message is None:
        commit_message = f"Upload fused MLX model {Path(model_path).name}"
    
    # Upload the model files
    api.upload_folder(
        folder_path=model_path,
        repo_id=hf_repo,
        commit_message=commit_message
    )
    
    print(f"✅ Model successfully pushed to https://huggingface.co/{hf_repo}")