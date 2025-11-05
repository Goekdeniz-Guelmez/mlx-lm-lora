from mlx_lm_lora.train import main

args = {
    "model": "mlx-community/Josiefied-Qwen3-0.6B-abliterated-v1-4bit",
    "data": "mlx-community/wikisql",
    "train": True,
    "train_mode": "sft",
    "train_type": "lora",  # dora or "full"
    "optimizer": "muon",  # or "adamw", etc.
    "iters": 100,
    "batch_size": 1,
    "steps_per_report": 10,
    "steps_per_eval": 50,
    "wandb": "mlx-lm-lora-sft-example",
    "mask_prompt": False,
    "adapter_path": "path/to/save/adaper",
    "save_every": 10,
}

main(args)
