import argparse
import importlib.util
import math
import re
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import yaml
from mlx_lm.tokenizer_utils import load_tokenizer
from mlx_lm.tuner.callbacks import WandBCallback
from mlx_lm.tuner.utils import (
    build_schedule,
    linear_to_lora_layers,
    load_adapters,
    print_trainable_parameters,
)
from mlx_lm.utils import load, save_config

from .trainer.datasets import CacheDataset, load_dataset
from .trainer.sft_trainer import (
    SFTTrainingArgs,
    TrainingCallback,
    evaluate_sft,
    train_sft,
)
from .utils import from_pretrained, fuse_and_save_model

yaml_loader = yaml.SafeLoader
yaml_loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        """^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$""",
        re.X,
    ),
    list("-+0123456789."),
)

CONFIG_DEFAULTS = {
    "model": "mlx_model",
    "load_in_4bits": False,
    "load_in_6bits": False,
    "load_in_8bits": False,
    "optimizer": "adam",
    "optimizer_config": {
        "adam": {},
        "adamw": {},
        "muon": {},
    },
    "data": "data/",
    "seed": 0,
    "num_layers": 16,
    "batch_size": 4,
    "iters": None,
    "epochs": None,
    "gradient_accumulation_steps": 1,
    "val_batches": 25,
    "learning_rate": 1e-5,
    "steps_per_report": 10,
    "steps_per_eval": 200,
    "resume_adapter_file": None,
    "adapter_path": "adapters",
    "save_every": 100,
    "test": False,
    "test_batches": 500,
    "max_seq_length": 2048,
    "config": None,
    "grad_checkpoint": False,
    "lr_schedule": None,
    "lora_parameters": {"rank": 8, "dropout": 0.0, "scale": 10.0},
    "mask_prompt": False,
    "fuse": True,
}


def load_reward_functions_from_file(file_path):
    """Load reward functions from a Python file"""
    if not file_path or not Path(file_path).exists():
        return None

    try:
        print(f"Loading custom reward functions from {file_path}")
        spec = importlib.util.spec_from_file_location("custom_rewards", file_path)
        custom_rewards = importlib.util.module_from_spec(spec)
        sys.modules["custom_rewards"] = custom_rewards
        spec.loader.exec_module(custom_rewards)
        print("Successfully loaded custom reward functions")
        return True
    except Exception as e:
        print(f"Error loading custom reward functions: {e}")
        return None


def calculate_iters(train_set, batch_size, epochs) -> int:
    num_samples = len(train_set)
    batches_per_epoch = math.ceil(num_samples / batch_size)
    iters = epochs * batches_per_epoch
    print(
        f"[INFO] Calculated {iters} iterations from {epochs} epochs (dataset size: {num_samples}, batch size: {batch_size})"
    )
    return iters


def build_parser():
    parser = argparse.ArgumentParser(description="LoRA or QLoRA finetuning.")
    parser.add_argument(
        "--model",
        type=str,
        help="The path to the local model directory or Hugging Face repo.",
    )
    parser.add_argument(
        "--load-in-4bits",
        action="store_true",
        help="Load the model in 4-bit quantization.",
        default=None,
    )
    parser.add_argument(
        "--load-in-6bits",
        action="store_true",
        help="Load the model in 6-bit quantization.",
        default=None,
    )
    parser.add_argument(
        "--load-in-8bits",
        action="store_true",
        help="Load the model in 8-bit quantization.",
        default=None,
    )

    # Training args
    parser.add_argument(
        "--data",
        type=str,
        help=(
            "Directory with {train, valid, test}.jsonl files or the name in the DPO-format "
            "of a Hugging Face dataset (e.g., 'mlx-community/orpo-dpo-mix-40k-flat-mlx')"
        ),
    )
    parser.add_argument(
        "--train-type",
        type=str,
        choices=["lora", "dora", "full"],
        help="Type of fine-tuning to perform: lora, dora, or full.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["adam", "adamw", "qhadam", "muon"],
        default=None,
        help="Optimizer to use for training: adam or adamw",
    )
    parser.add_argument(
        "--mask-prompt",
        action="store_true",
        help="Mask the prompt in the loss when training",
        default=None,
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        help="Number of layers to fine-tune. Default is 16, use -1 for all.",
    )
    parser.add_argument("--batch-size", type=int, help="Minibatch size.")
    parser.add_argument("--iters", type=int, help="Iterations to train for.")
    parser.add_argument(
        "--epochs",
        type=int,
        help="Epochs to train for. Ignored if --iters is provided.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        help="Number of gradient accumulation steps.",
        default=1,
    )
    parser.add_argument(
        "--val-batches",
        type=int,
        help="Number of validation batches, -1 uses the entire validation set.",
    )
    parser.add_argument("--learning-rate", type=float, help="Adam learning rate.")
    parser.add_argument(
        "--steps-per-report",
        type=int,
        help="Number of training steps between loss reporting.",
    )
    parser.add_argument(
        "--steps-per-eval",
        type=int,
        help="Number of training steps between validations.",
    )
    parser.add_argument(
        "--resume-adapter-file",
        type=str,
        help="Load path to resume training from the given fine-tuned weights.",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        help="Save/load path for the fine-tuned weights.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        help="Save the model every N iterations.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Evaluate on the test set after training",
        default=None,
    )
    parser.add_argument(
        "--test-batches",
        type=int,
        help="Number of test set batches, -1 uses the entire test set.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        help="Maximum sequence length.",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="A YAML configuration file with the training options",
    )
    parser.add_argument(
        "--grad-checkpoint",
        action="store_true",
        help="Use gradient checkpointing to reduce memory use.",
        default=None,
    )
    parser.add_argument(
        "--wandb",
        type=str,
        default=None,
        help="WandB project name to report training metrics. Disabled if None.",
    )
    parser.add_argument("--seed", type=int, help="The PRNG seed")
    parser.add_argument(
        "--fuse",
        action="store_true",
        help="Fuse and save the trained model.",
        default=None,
    )
    return parser


def train_model(
    args,
    model: nn.Module,
    tokenizer,
    train_set,
    valid_set,
    training_callback: TrainingCallback = None,
):
    mx.random.seed(args.seed)

    if args.iters is None and args.epochs is not None:
        args.iters = calculate_iters(
            train_set=train_set, batch_size=args.batch_size, epochs=args.epochs
        )

    model.freeze()
    if args.num_layers > len(model.layers):
        raise ValueError(
            f"Requested to train {args.num_layers} layers "
            f"but the model only has {len(model.layers)} layers."
        )

    if args.train_type == "full":
        for l in model.layers[-max(args.num_layers, 0) :]:
            l.unfreeze()
    elif args.train_type in ["lora", "dora"]:
        # Convert linear layers to lora/dora layers and unfreeze in the process
        linear_to_lora_layers(
            model,
            args.num_layers,
            args.lora_parameters,
            use_dora=(args.train_type == "dora"),
        )
    else:
        raise ValueError(f"Received unknown train-type {args.train_type}")

    # Resume from weights if provided
    if args.resume_adapter_file is not None:
        print(f"Loading fine-tuned weights from {args.resume_adapter_file}")
        model.load_weights(args.resume_adapter_file, strict=False)

    print_trainable_parameters(model)

    adapter_path = Path(args.adapter_path)
    adapter_path.mkdir(parents=True, exist_ok=True)

    adapter_file = adapter_path / "adapters.safetensors"
    save_config(vars(args), adapter_path / "adapter_config.json")

    # Initialize the selected optimizer
    lr = build_schedule(args.lr_schedule) if args.lr_schedule else args.learning_rate

    optimizer_name = args.optimizer.lower()
    optimizer_config = args.optimizer_config.get(optimizer_name, {})

    if optimizer_name == "adam":
        opt_class = optim.Adam
    elif optimizer_name == "adamw":
        opt_class = optim.AdamW
    elif optimizer_name == "muon":
        opt_class = optim.Muon
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    opt = opt_class(learning_rate=lr, **optimizer_config)

    sft_training_args = SFTTrainingArgs(
        batch_size=args.batch_size,
        iters=args.iters,
        val_batches=args.val_batches,
        steps_per_report=args.steps_per_report,
        steps_per_eval=args.steps_per_eval,
        steps_per_save=args.save_every,
        adapter_file=adapter_file,
        max_seq_length=args.max_seq_length,
        grad_checkpoint=args.grad_checkpoint,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    train_sft(
        model=model,
        args=sft_training_args,
        optimizer=opt,
        train_dataset=CacheDataset(train_set),
        val_dataset=CacheDataset(valid_set),
        training_callback=training_callback,
    )


def evaluate_model(args, model: nn.Module, tokenizer, test_set):
    test_loss = evaluate_sft(
        model=model,
        dataset=CacheDataset(test_set),
        batch_size=args.batch_size,
        num_batches=args.test_batches,
        max_seq_length=args.max_seq_length,
    )

    test_ppl = math.exp(test_loss)

    print(f"Test loss {test_loss:.3f}, Test ppl {test_ppl:.3f}.")


def run(args, training_callback: TrainingCallback = None):
    np.random.seed(args.seed)
    args.train_mode = "judge"
    args.train = True

    if args.wandb is not None:
        training_callback = WandBCallback(
            project_name=args.wandb,
            log_dir=args.adapter_path,
            config=vars(args),
            wrapped_callback=training_callback,
        )

    if args.load_in_4bits:
        quanziation_config = {"bits": 4, "group_size": 64}
    elif args.load_in_6bits:
        quanziation_config = {"bits": 6, "group_size": 64}
    elif args.load_in_8bits:
        quanziation_config = {"bits": 8, "group_size": 64}
    else:
        quanziation_config = None

    model, tokenizer = from_pretrained(
        model=args.model,
        quantized_load=quanziation_config,
    )

    print("Loading datasets")
    train_set, valid_set, test_set = load_dataset(
        args,
        tokenizer,
    )

    if args.test and not args.train:
        if args.adapter_path != "":
            load_adapters(model, args.adapter_path)

    print("Training")
    train_model(args, model, tokenizer, train_set, valid_set, training_callback)

    if args.test:
        print("Testing")
        evaluate_model(args, model, tokenizer, test_set)

    if args.fuse:
        print("Fusing model")
        fuse_and_save_model(
            model=model,
            tokenizer=tokenizer,
            save_path=args.adapter_path,
            adapter_path=None,
            de_quantize=True,
            export_gguf=False,
        )


def main(args=None):
    import os
    import types

    import yaml

    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    if args is None:
        parser = build_parser()
        args = parser.parse_args()
    elif isinstance(args, dict):
        # Allow programmatic overrides from notebook
        default_args = vars(build_parser().parse_args([]))
        default_args.update(args)
        args = types.SimpleNamespace(**default_args)

    if args.config:
        with open(args.config, "r") as f:
            config_args = yaml.load(f, Loader=yaml_loader)
            for k, v in config_args.items():
                if getattr(args, k, None) is None:
                    setattr(args, k, v)

    # Set all None args to defaults
    for k, v in CONFIG_DEFAULTS.items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)

    run(args)


if __name__ == "__main__":
    main()
