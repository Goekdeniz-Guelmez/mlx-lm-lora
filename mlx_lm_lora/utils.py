import datetime
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.gguf import convert_to_gguf
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.tuner.utils import linear_to_lora_layers, load_adapters
from mlx_lm.utils import dequantize_model, load, save_config, save_model
from transformers import AutoProcessor


def calculate_iters(train_set, batch_size, epochs) -> int:
    num_samples = len(train_set)
    batches_per_epoch = math.ceil(num_samples / batch_size)
    iters = epochs * batches_per_epoch
    print(
        f"[INFO] Calculated {iters} iterations from {epochs} epochs (dataset size: {num_samples}, batch size: {batch_size})"
    )
    return iters


def find_lmstudio_models_path() -> Path:
    """
    Find the LM Studio models directory.

    Returns:
        Path: The path to the LM Studio models directory.
    """
    lm = Path.home() / ".lmstudio" / "models"

    if not lm.exists():
        raise FileNotFoundError(f"LM Studio models root not found at {lm}")

    return lm


def save_pretrained(
    model: nn.Module,
    tokenizer: TokenizerWrapper,
    save_path: str = "fused_model",
    export_gguf: Optional[bool] = False,
    gguf_path: Optional[str] = "ggml-model-f16.gguf",
    remove_adapters: Optional[bool] = False,
) -> None:
    """
    Fuse fine-tuned adapters into the base model.

    Args:
        model: The MLX model to fuse adapters into.
        tokenizer: The tokenizer wrapper.
        save_path: The path to save the fused model.
        export_gguf: Export model weights in GGUF format.
        gguf_path: Path to save the exported GGUF format model weights.
        remove_adapters: Whether to remove adapter files from the saved model directory.
    """
    from ._version import __version__

    save_path_obj = Path(save_path)
    save_model(save_path_obj, model, donate_model=True)
    save_config(vars(model.args), config_path=save_path_obj / "config.json")
    tokenizer.save_pretrained(save_path_obj)

    readme_content = f"""# MLX-LM-LoRA Model

This model was fine-tuned using [mlx-lm-lora](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora) version {__version__}.

## Model Details

- Base model: {vars(model.args).get('model_name', 'Unknown')}
- Model type: {vars(model.args).get('model_type', 'Unknown')}
- Training method: LoRA fine-tuning with MLX
- Fusion date: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Usage

This model can be loaded and used with the MLX framework.
"""
    with open(save_path_obj / "README.md", "w") as f:
        f.write(readme_content)

    print(f"Created README.md in {save_path}")

    if remove_adapters:
        adapter_config_file = save_path_obj / "adapter_config.json"
        if adapter_config_file.exists():
            adapter_config_file.unlink()
            print(f"Removed {adapter_config_file}")

        adapter_patterns = ["adapters*.safetensors", "*adapters.safetensors"]
        for pattern in adapter_patterns:
            for adapter_file in save_path_obj.glob(pattern):
                adapter_file.unlink()
                print(f"Removed {adapter_file}")

    if export_gguf:
        model_type = model.args["model_type"]
        if model_type not in ["llama", "mixtral", "mistral"]:
            raise ValueError(
                f"Model type {model_type} not supported for GGUF conversion."
            )
        weights = dict(tree_flatten(model.parameters()))
        convert_to_gguf(save_path, weights, model.args, str(save_path_obj / gguf_path))


def save_pretrained_merged(
    model: nn.Module,
    tokenizer: TokenizerWrapper,
    save_path: str = "fused_model",
    adapter_path: Optional[str] = None,
    de_quantize: Optional[bool] = False,
    export_gguf: Optional[bool] = False,
    gguf_path: Optional[str] = "ggml-model-f16.gguf",
    remove_adapters: Optional[bool] = False,
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
        remove_adapters: Whether to remove adapter files from the saved model directory.
    """
    from ._version import __version__

    model.freeze()

    if adapter_path is not None:
        print(f"Loading adapters from {adapter_path}")
        model = load_adapters(model, adapter_path)

    args = vars(model.args)

    fused_linears = [
        (n, m.fuse()) for n, m in model.named_modules() if hasattr(m, "fuse")
    ]

    if fused_linears:
        model.update_modules(tree_unflatten(fused_linears))

    if de_quantize:
        print("De-quantizing model")
        model = dequantize_model(model)
        args.pop("quantization", None)
        args.pop("quantization_config", None)

    save_pretrained(
        model=model,
        tokenizer=tokenizer,
        save_path=save_path,
        export_gguf=export_gguf,
        gguf_path=gguf_path,
        remove_adapters=remove_adapters,
    )


def from_pretrained(
    model: str,
    adapter_path: Optional[str] = None,
    new_adapter_path: Optional[str] = None,
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
        Tuple[nn.Module, tokenizer, Optional[str]]: The model with LoRA adapters loaded, the tokenizer, and the adapter path if provided.
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
        group_size = quantized_load.get("group_size", 64)
        mode = quantized_load.get("mode", "affine")

        nn.quantize(model, bits=bits, group_size=group_size, mode=mode)

        if hasattr(model, "args"):
            model.args.quantization = {
                "group_size": group_size,
                "bits": bits,
                "mode": mode,
            }
            model.args.quantization_config = model.args.quantization

    if new_adapter_path is not None:
        args = (
            {
                "lora_parameters": lora_config,
                "num_layers": lora_config.get("num_layers", None),
            }
            if lora_config is not None
            else {} | args
        )
        new_adapter_path = Path(new_adapter_path)
        new_adapter_path.mkdir(parents=True, exist_ok=True)
        new_adapter_file = new_adapter_path / "adapters.safetensors"
        save_config(args, new_adapter_path / "adapter_config.json")

    return model, tokenizer, new_adapter_file if new_adapter_path is not None else None


def push_to_hub(
    model_path: str,
    hf_repo: str,
    api_key: str,
    private: bool = False,
    commit_message: Optional[str] = None,
    remove_adapters: Optional[bool] = False,
) -> None:
    """
    Push the fused model to the Hugging Face Hub.

    Args:
        model_path: Local path of the model to upload.
        hf_repo: Name of the HF repo (format: username/repo_name).
        api_key: Hugging Face API token.
        private: Whether to create a private repository.
        commit_message: Custom commit message for the upload.
        remove_adapters: Whether to remove adapters before pushing.
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
    os.environ["HF_TOKEN"] = api_key
    api = HfApi()

    # Create the repo if it doesn't exist
    try:
        create_repo(hf_repo, private=private, token=api_key, repo_type="model")
    except Exception as e:
        print(f"Repository creation failed or repository already exists: {e}")

    # Set default commit message if not provided
    if commit_message is None:
        commit_message = f"Upload fused MLX model {Path(model_path).name}"

    # Upload the model files
    api.upload_folder(
        folder_path=model_path,
        repo_id=hf_repo,
        commit_message=commit_message,
        ignore_patterns=(
            ["adapters*.safetensors", "adapters*.json"] if remove_adapters else None
        ),
    )

    print(f"✅ Model successfully pushed to https://huggingface.co/{hf_repo}")


def save_to_lmstudio_merged(
    model: nn.Module,
    tokenizer: TokenizerWrapper,
    new_model_name: str = "mlx_lm_lora_model",
    de_quantize: Optional[bool] = True,
) -> None:
    """
    Fuse fine-tuned adapters into the base model.

    Args:
        model: The MLX model to fuse adapters into.
        tokenizer: The tokenizer wrapper.
        new_model_name: The name of the new fused model.
        de_quantize: Generate a de-quantized model.
    """

    lmstudio_models_root = find_lmstudio_models_path()

    lmstudio_models_path = lmstudio_models_root / "mlx_lm_lora"
    lmstudio_models_path.mkdir(parents=True, exist_ok=True)

    model_path = lmstudio_models_path / new_model_name

    print(f"LM Studio models directory found at: {lmstudio_models_root}")

    save_pretrained_merged(
        model=model,
        tokenizer=tokenizer,
        save_path=str(model_path),
        de_quantize=de_quantize,
    )

    print(f"Model successfully sent to LM Studio at {model_path}")


def save_pretrained_merged_vision(
    model_name: str,
    text_model: nn.Module,
    output_path: Union[str, Path],
    de_quantize: bool = True,
) -> None:
    """Merge trained text model weights back into the full VLM and save.

    Works entirely with safetensors on disk – no need to instantiate the full
    VLM in memory.  Only requires ``huggingface_hub`` and ``transformers``
    (no ``mlx_vlm``).

    Args:
        model_name: HuggingFace repo id or local path of the original VLM.
        text_model: The fine-tuned MLX text sub-model (may contain LoRA layers).
        output_path: Directory where the merged model will be saved.
        de_quantize: Whether to de-quantize the text model before merging.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    model_path = Path(model_name)
    if not model_path.exists():
        model_path = Path(snapshot_download(model_name))
    print(f"[INFO] VLM source: {model_path}")

    text_model.freeze()
    fused_linears = [
        (n, m.fuse()) for n, m in text_model.named_modules() if hasattr(m, "fuse")
    ]
    if fused_linears:
        text_model.update_modules(tree_unflatten(fused_linears))
    if de_quantize:
        text_model = dequantize_model(text_model)

    trained_weights = dict(tree_flatten(text_model.parameters()))

    index_file = model_path / "model.safetensors.index.json"
    if index_file.exists():
        with open(index_file) as f:
            weight_map = json.load(f)["weight_map"]
        shard_files = sorted(set(weight_map.values()))
    else:
        shard_files = [
            p.name
            for p in sorted(model_path.glob("*.safetensors"))
            if "adapter" not in p.name.lower()
        ]
        weight_map = None

    if not shard_files:
        raise FileNotFoundError(f"No safetensors files found in {model_path}")

    vlm_keys: set = set()
    for sf in shard_files:
        shard = mx.load(str(model_path / sf))
        vlm_keys.update(shard.keys())
        del shard

    PREFIXES = [
        "",
        "model.language_model.model.",
        "model.language_model.",
        "language_model.model.",
        "language_model.",
        "model.text_model.",
        "text_model.",
        "model.",
    ]

    def _strip_prefix(key: str) -> str:
        """Strip the first matching known prefix to get the bare weight name."""
        for p in PREFIXES[1:]:
            if key.startswith(p):
                return key[len(p) :]
        return key

    bare_to_vlm: dict[str, str] = {}
    for vk in vlm_keys:
        bare = _strip_prefix(vk)
        bare_to_vlm[bare] = vk

    key_mapping: dict[str, str] = {}
    for tkey in trained_weights:
        if tkey in vlm_keys:
            key_mapping[tkey] = tkey
            continue
        bare = _strip_prefix(tkey)
        if bare in bare_to_vlm:
            key_mapping[bare_to_vlm[bare]] = tkey

    if not key_mapping:
        raise ValueError(
            f"No weights matched between text model and VLM.\n"
            f"  Text keys sample: {list(trained_weights.keys())[:5]}\n"
            f"  VLM keys sample:  {sorted(vlm_keys)[:5]}"
        )
    print(
        f"[INFO] Merging {len(key_mapping)}/{len(trained_weights)} text weights into VLM"
    )

    new_index: dict = {"metadata": {}, "weight_map": {}}
    total_size = 0
    shard_count = len(shard_files)

    for i, sf in enumerate(shard_files):
        shard = dict(mx.load(str(model_path / sf)))

        for vlm_key in list(shard.keys()):
            if vlm_key in key_mapping:
                shard[vlm_key] = trained_weights[key_mapping[vlm_key]]

        out_name = (
            f"model-{i + 1:05d}-of-{shard_count:05d}.safetensors"
            if shard_count > 1
            else "model.safetensors"
        )
        mx.save_safetensors(
            str(output_path / out_name), shard, metadata={"format": "mlx"}
        )

        for k, v in shard.items():
            new_index["weight_map"][k] = out_name
            total_size += v.nbytes
        del shard

    new_index["metadata"]["total_size"] = total_size
    new_index["weight_map"] = dict(sorted(new_index["weight_map"].items()))

    with open(output_path / "model.safetensors.index.json", "w") as f:
        json.dump(new_index, f, indent=4)

    for pattern in ["config.json", "*.json", "*.txt", "*.model", "*.tiktoken"]:
        for src in model_path.glob(pattern):
            if src.name == "model.safetensors.index.json":
                continue  # we wrote our own
            dst = output_path / src.name
            if not dst.exists():
                shutil.copy2(src, dst)

    try:
        processor = AutoProcessor.from_pretrained(str(model_path))
        processor.save_pretrained(str(output_path))
    except Exception as e:
        print(f"[WARN] Could not save processor ({e}); config files were still copied.")

    for adapter_file in output_path.glob("*adapter*"):
        adapter_file.unlink()
        print(f"[INFO] Removed adapter artifact: {adapter_file.name}")

    print(f"✓ Merged VLM saved to {output_path}")
