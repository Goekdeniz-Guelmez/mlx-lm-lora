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

from .trainer.cpo_trainer import CPOTrainingArgs, evaluate_cpo, train_cpo
from .trainer.datasets import CacheDataset, load_dataset
from .trainer.dpo_trainer import DPOTrainingArgs, evaluate_dpo, train_dpo
from .trainer.grpo_reward_functions import (
    get_default_reward_functions,
    get_reward_function,
    list_available_reward_functions,
)
from .trainer.grpo_trainer import GRPOTrainingArgs, evaluate_grpo, train_grpo
from .trainer.online_dpo_trainer import (
    OnlineDPOTrainingArgs,
    evaluate_online_dpo,
    train_online_dpo,
)
from .trainer.orpo_trainer import ORPOTrainingArgs, evaluate_orpo, train_orpo
from .trainer.rflhf_trainer import RLHFTrainingArgs, evaluate_rlhf, train_rlhf
from .trainer.sft_trainer import (
    SFTTrainingArgs,
    TrainingCallback,
    evaluate_sft,
    train_sft,
)
from .trainer.xpo_trainer import XPOTrainingArgs, evaluate_xpo, train_xpo
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
    "train": True,
    "load_in_4bits": False,
    "load_in_6bits": False,
    "load_in_8bits": False,
    "train_type": "lora",
    "train_mode": "sft",
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
    # ORPO args
    "beta": 0.1,
    "reward_scaling": 1.0,
    # DPO args
    "dpo_cpo_loss_type": "sigmoid",
    "delta": 50.0,
    "reference_model_path": None,
    # Online DPO & XPO
    "judge": None,
    "judge_config": {},
    "alpha": 1e-5,
    # GRPO args
    "group_size": 4,
    "epsilon": 1e-4,
    "epsilon_high": None,  # DAPO
    "max_completion_length": 512,
    "temperature": 0.8,
    "reward_weights": None,
    "reward_functions": None,
    "reward_functions_file": None,
    "grpo_loss_type": "grpo",
    "importance_sampling_level": None,  # GSPO
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
        "--train",
        action="store_true",
        help="Do training",
        default=None,
    )
    parser.add_argument(
        "--data",
        type=str,
        help=(
            "Directory with {train, valid, test}.jsonl files or the name "
            "of a Hugging Face dataset (e.g., 'mlx-community/wikisql')"
        ),
    )
    parser.add_argument(
        "--train-type",
        type=str,
        choices=["lora", "dora", "full"],
        help="Type of fine-tuning to perform: lora, dora, or full.",
    )
    parser.add_argument(
        "--train-mode",
        type=str,
        default="sft",
        choices=["sft", "dpo", "cpo", "orpo", "grpo", "online_dpo", "xpo", "rlhf"],
        help="Training mode: sft, dpo, rlhf, online_dpo, xpo, cpo, orpo, or grpo, default is sft",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["adam", "adamw", "muon"],
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
    parser.add_argument("--learning-rate", type=float, help="Optimizer learning rate.")
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

    # ORPO args
    parser.add_argument(
        "--beta",
        type=float,
        help="Temperature parameter for ORPO training.",
        default=0.1,
    )
    parser.add_argument(
        "--reward-scaling",
        type=float,
        help="Reward scaling factor for ORPO training, not implemented.",
        default=1.0,
    )

    # DPO args
    parser.add_argument(
        "--dpo-cpo-loss-type",
        type=str,
        help="DPO loss type: 'sigmoid', 'hinge', 'ipo', or 'dpop'.",
        choices=["sigmoid", "hinge", "ipo", "dpop"],
        default="sigmoid",
    )
    parser.add_argument(
        "--delta", type=float, help="Delta parameter for DPOP loss type.", default=50.0
    )
    parser.add_argument(
        "--reference-model-path",
        type=str,
        help="Path to reference model weights. If None, uses the same model.",
        default=None,
    )

    # Online DPO & XPO args
    parser.add_argument(
        "--judge",
        type=str,
        help="Judge to use can be a model ID or 'human'.",
        default="mlx-community/Josiefied-Qwen2.5-7B-Instruct-abliterated-v2-4-bit",
    )
    parser.add_argument(
        "--alpha",
        type=list[float],
        help="Judge to use can be a model ID or 'human'.",
        default=[1e-5],
    )

    # GRPO args
    parser.add_argument(
        "--group-size",
        type=int,
        help="Number of generations.",
        default=4,
    )
    parser.add_argument(
        "--max-completion-length",
        type=int,
        help="Maximum length of the prompt. If the prompt is longer than this value, it will be truncated left.",
        default=512,
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        help="The Epsilon for numerical stability.",
        default=1e-4,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Temperature for sampling. The higher the temperature, the more random the completions.",
        default=1.0,
    )
    parser.add_argument(
        "--reward-weights",
        type=str,
        help="Weights for each reward function. Must match the number of reward functions and be in this format [0.1, 0.2, 0.3, 0.4, 0.5]. If not given, all rewards are weighted equally with weight `1.0`.",
        default=None,
    )
    parser.add_argument(
        "--reward-functions",
        type=str,
        help=(
            "Comma-separated list of reward function names to use. These must be registered in the reward_functions registry. "
            "Use --list-reward-functions to see available functions. "
            "Example: r1_accuracy_reward_func,action_format_reward_func"
        ),
        default=None,
    )
    parser.add_argument(
        "--reward-functions-file",
        type=str,
        help=(
            "Path to a Python file containing custom reward functions. "
            "The file should define functions decorated with @register_reward_function(). "
            "Example: path/to/my_reward_functions.py"
        ),
        default=None,
    )
    parser.add_argument(
        "--list-reward-functions",
        action="store_true",
        help="List all available reward functions and exit",
    )

    parser.add_argument(
        "--grpo-loss-type",
        type=str,
        help="GRPO loss type: 'grpo', 'bnpo', or 'dr_grpo'.",
        choices=["grpo", "bnpo", "dr_grpo"],
        default="grpo",
    )

    # DAPO args
    parser.add_argument(
        "--epsilon-high",
        type=float,
        help="Upper-bound epsilon value for clipping. If not specified, it defaults to the same value as the lower-bound specified in argument epsilon.",
        default=None,
    )

    # GSPO args
    parser.add_argument(
        "--importance-sampling-level",
        type=str,
        choices=["token", "sequence", None],
        default=None,
        help=(
            "Level of importance sampling to use. "
            "'token' uses token-level importance sampling, 'sequence' uses sequence-level, and None (default) disables it."
        ),
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

    if args.train_mode == "orpo":
        orpo_training_args = ORPOTrainingArgs(
            batch_size=args.batch_size,
            iters=args.iters,
            val_batches=args.val_batches,
            steps_per_report=args.steps_per_report,
            steps_per_eval=args.steps_per_eval,
            steps_per_save=args.save_every,
            adapter_file=adapter_file,
            max_seq_length=args.max_seq_length,
            grad_checkpoint=args.grad_checkpoint,
            beta=args.beta,
            reward_scaling=args.reward_scaling,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )

        train_orpo(
            model=model,
            optimizer=opt,
            train_dataset=CacheDataset(train_set),
            val_dataset=CacheDataset(valid_set),
            args=orpo_training_args,
            training_callback=training_callback,
        )
    elif args.train_mode == "dpo":
        dpo_training_args = DPOTrainingArgs(
            batch_size=args.batch_size,
            iters=args.iters,
            val_batches=args.val_batches,
            steps_per_report=args.steps_per_report,
            steps_per_eval=args.steps_per_eval,
            steps_per_save=args.save_every,
            adapter_file=adapter_file,
            max_seq_length=args.max_seq_length,
            grad_checkpoint=args.grad_checkpoint,
            beta=args.beta,
            loss_type=args.dpo_cpo_loss_type,
            delta=args.delta,
            reference_model_path=args.reference_model_path,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )

        print("Loading pretrained reference model")
        if args.reference_model_path:
            reference_model, _ = load(args.reference_model_path)
        else:
            reference_model, _ = load(args.model)

        train_dpo(
            model=model,
            ref_model=reference_model.freeze(),
            optimizer=opt,
            train_dataset=CacheDataset(train_set),
            val_dataset=CacheDataset(valid_set),
            args=dpo_training_args,
            training_callback=training_callback,
        )

    elif args.train_mode == "online_dpo":
        online_dpo_training_args = OnlineDPOTrainingArgs(
            batch_size=args.batch_size,
            iters=args.iters,
            val_batches=args.val_batches,
            steps_per_report=args.steps_per_report,
            steps_per_eval=args.steps_per_eval,
            steps_per_save=args.save_every,
            adapter_file=adapter_file,
            max_seq_length=args.max_seq_length,
            grad_checkpoint=args.grad_checkpoint,
            beta=args.beta,
            loss_type=args.dpo_cpo_loss_type,
            delta=args.delta,
            reference_model_path=args.reference_model_path,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            judge=args.judge,
            max_completion_length=args.max_completion_length,
            temperature=args.temperature,
        )

        print("Loading pretrained reference model")
        if args.reference_model_path:
            reference_model, _ = load(args.reference_model_path)
        else:
            reference_model, _ = load(args.model)

        print("Loading pretrained judge model")
        if args.judge:
            if args.judge == args.reference_model_path:
                judge_model = reference_model
                judge_tokenizer = load_tokenizer(args.judge)
            else:
                judge_model, judge_tokenizer = load(args.judge)
        else:
            judge_model, judge_tokenizer = load(args.judge)

        train_online_dpo(
            model=model,
            tokenizer=tokenizer,
            ref_model=reference_model.freeze(),
            judge_model=judge_model.freeze(),
            judge_tokenizer=judge_tokenizer,
            judge_config=args.judge_config,
            optimizer=opt,
            train_dataset=CacheDataset(train_set),
            val_dataset=CacheDataset(valid_set),
            args=online_dpo_training_args,
            training_callback=training_callback,
        )

    elif args.train_mode == "rlhf":
        online_dpo_training_args = RLHFTrainingArgs(
            batch_size=args.batch_size,
            iters=args.iters,
            val_batches=args.val_batches,
            steps_per_report=args.steps_per_report,
            steps_per_eval=args.steps_per_eval,
            steps_per_save=args.save_every,
            adapter_file=adapter_file,
            max_seq_length=args.max_seq_length,
            grad_checkpoint=args.grad_checkpoint,
            beta=args.beta,
            reference_model_path=args.reference_model_path,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            judge=args.judge,
            max_completion_length=args.max_completion_length,
        )

        print("Loading pretrained reference model")
        if args.reference_model_path:
            reference_model, _ = load(args.reference_model_path)
        else:
            reference_model, _ = load(args.model)

        print("Loading pretrained judge model")
        if args.judge:
            if args.judge == args.reference_model_path:
                judge_model = reference_model
                judge_tokenizer = load_tokenizer(args.judge)
            else:
                judge_model, judge_tokenizer = load(args.judge)
        else:
            judge_model, judge_tokenizer = load(args.judge)

        train_rlhf(
            model=model,
            tokenizer=tokenizer,
            ref_model=reference_model.freeze(),
            judge_model=judge_model.freeze(),
            judge_tokenizer=judge_tokenizer,
            judge_config=args.judge_config,
            optimizer=opt,
            train_dataset=CacheDataset(train_set),
            val_dataset=CacheDataset(valid_set),
            args=online_dpo_training_args,
            training_callback=training_callback,
        )

    elif args.train_mode == "xpo":
        xpo_training_args = XPOTrainingArgs(
            batch_size=args.batch_size,
            iters=args.iters,
            val_batches=args.val_batches,
            steps_per_report=args.steps_per_report,
            steps_per_eval=args.steps_per_eval,
            steps_per_save=args.save_every,
            adapter_file=adapter_file,
            max_seq_length=args.max_seq_length,
            grad_checkpoint=args.grad_checkpoint,
            beta=args.beta,
            loss_type=args.dpo_cpo_loss_type,
            delta=args.delta,
            reference_model_path=args.reference_model_path,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            alpha=args.alpha,
            judge=args.judge,
            max_completion_length=args.max_completion_length,
        )

        print("Loading pretrained reference model")
        if args.reference_model_path:
            reference_model, _ = load(args.reference_model_path)
        else:
            reference_model, _ = load(args.model)

        print("Loading pretrained judge model")
        if args.judge:
            if args.judge == args.reference_model_path:
                judge_model = reference_model
                judge_tokenizer = load_tokenizer(args.judge)
            else:
                judge_model, judge_tokenizer = load(args.judge)
        else:
            judge_model, judge_tokenizer = load(args.judge)

        train_xpo(
            model=model,
            tokenizer=tokenizer,
            ref_model=reference_model.freeze(),
            judge_config=args.judge_config,
            judge_model=judge_model.freeze(),
            judge_tokenizer=judge_tokenizer,
            optimizer=opt,
            train_dataset=CacheDataset(train_set),
            val_dataset=CacheDataset(valid_set),
            args=xpo_training_args,
            training_callback=training_callback,
        )

    elif args.train_mode == "cpo":
        cpo_training_args = CPOTrainingArgs(
            batch_size=args.batch_size,
            iters=args.iters,
            val_batches=args.val_batches,
            steps_per_report=args.steps_per_report,
            steps_per_eval=args.steps_per_eval,
            steps_per_save=args.save_every,
            adapter_file=adapter_file,
            max_seq_length=args.max_seq_length,
            grad_checkpoint=args.grad_checkpoint,
            beta=args.beta,
            loss_type=args.dpo_cpo_loss_type,
            delta=args.delta,
            reference_model_path=args.reference_model_path,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )

        train_cpo(
            model=model,
            optimizer=opt,
            train_dataset=CacheDataset(train_set),
            val_dataset=CacheDataset(valid_set),
            args=cpo_training_args,
            training_callback=training_callback,
        )

    elif args.train_mode == "grpo":
        if args.reward_functions_file:
            load_reward_functions_from_file(args.reward_functions_file)

        reward_funcs = get_default_reward_functions()
        if args.reward_functions:
            func_names = [name.strip() for name in args.reward_functions.split(",")]
            try:
                reward_funcs = [get_reward_function(name) for name in func_names]
                print(f"Using custom reward functions: {', '.join(func_names)}")
            except KeyError as e:
                print(f"Error: {str(e)}")
                print(
                    f"Available reward functions: {list_available_reward_functions()}"
                )
                return

        grpo_training_args = GRPOTrainingArgs(
            batch_size=args.batch_size,
            iters=args.iters,
            val_batches=args.val_batches,
            steps_per_report=args.steps_per_report,
            steps_per_eval=args.steps_per_eval,
            steps_per_save=args.save_every,
            adapter_file=adapter_file,
            max_seq_length=args.max_seq_length,
            max_completion_length=args.max_completion_length,
            grad_checkpoint=args.grad_checkpoint,
            beta=args.beta,
            group_size=args.group_size,
            epsilon=args.epsilon,
            epsilon_high=args.epsilon_high,
            reference_model_path=args.reference_model_path,
            temperature=args.temperature,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            reward_weights=(
                [float(x) for x in args.reward_weights.strip("[]").split(",")]
                if args.reward_weights
                else None
            ),
            importance_sampling_level=args.importance_sampling_level,
            grpo_loss_type=args.grpo_loss_type,
        )

        print("Loading pretrained reference model")
        if args.reference_model_path:
            reference_model, _ = load(args.reference_model_path)
        elif args.beta == 0:
            reference_model = None
        else:
            reference_model, _ = load(args.model)

        train_grpo(
            model=model,
            ref_model=reference_model.freeze() if reference_model else None,
            tokenizer=tokenizer,
            optimizer=opt,
            train_dataset=train_set,
            val_dataset=valid_set,
            reward_funcs=reward_funcs,
            args=grpo_training_args,
            training_callback=training_callback,
        )

    elif args.train_mode == "sft":
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

    else:
        raise (f"The train mode {args.train_mode} does not exist.")


def evaluate_model(args, model: nn.Module, tokenizer, test_set):
    if args.train_mode == "orpo":
        test_loss, test_rewards, _, test_metrics = evaluate_orpo(
            model=model,
            dataset=test_set,
            batch_size=args.batch_size,
            num_batches=args.test_batches,
            max_seq_length=args.max_seq_length,
            beta=args.beta,
        )
        test_ppl = math.exp(test_loss)
        print(
            f"Test loss {test_loss:.3f}, Test ppl {test_ppl:.3f}, Rewards: {test_rewards[0]:.3f}, {test_rewards[1]:.3f}"
        )

        print("ORPO Test Metrics:")
        for metric_name, metric_value in test_metrics.items():
            print(f"  {metric_name}: {float(metric_value):.3f}")

    elif args.train_mode == "dpo":
        if args.reference_model_path:
            reference_model, _ = load(args.reference_model_path)
        else:
            reference_model = model

        test_loss, _, _, test_metrics = evaluate_dpo(
            model=model,
            ref_model=reference_model.freeze(),
            dataset=test_set,
            batch_size=args.batch_size,
            num_batches=args.test_batches,
            max_seq_length=args.max_seq_length,
            beta=args.beta,
            delta=args.delta,
            loss_type=args.dpo_cpo_loss_type,
        )

        test_ppl = math.exp(test_loss)

        print(f"Test loss {test_loss:.3f}, Test ppl {test_ppl:.3f}")
        print("DPO Test Metrics:")
        for metric_name, metric_value in test_metrics.items():
            print(f"  {metric_name}: {float(metric_value):.3f}")

    elif args.train_mode == "rlhf":
        if args.reference_model_path:
            reference_model, _ = load(args.reference_model_path)
        else:
            reference_model, _ = load(args.model)

        test_loss, _, _, test_metrics = evaluate_rlhf(
            model=model,
            ref_model=reference_model.freeze(),
            dataset=test_set,
            batch_size=args.batch_size,
            num_batches=args.test_batches,
            max_seq_length=args.max_seq_length,
            beta=args.beta,
            loss_type=args.dpo_cpo_loss_type,
            judge=args.judge,
            max_tokens=args.max_completion_length,
        )

    elif args.train_mode == "online_dpo":
        if args.reference_model_path:
            reference_model, _ = load(args.reference_model_path)
        else:
            reference_model, _ = load(args.model)

        test_loss, _, _, test_metrics = evaluate_online_dpo(
            model=model,
            ref_model=reference_model.freeze(),
            dataset=test_set,
            batch_size=args.batch_size,
            num_batches=args.test_batches,
            max_seq_length=args.max_seq_length,
            beta=args.beta,
            delta=args.delta,
            loss_type=args.dpo_cpo_loss_type,
            judge=args.judge,
            max_tokens=args.max_completion_length,
        )

    elif args.train_mode == "xpo":
        if args.reference_model_path:
            reference_model, _ = load(args.reference_model_path)
        else:
            reference_model, _ = load(args.model)

        test_loss, _, _, test_metrics = evaluate_xpo(
            model=model,
            ref_model=reference_model.freeze(),
            dataset=test_set,
            batch_size=args.batch_size,
            num_batches=args.test_batches,
            max_seq_length=args.max_seq_length,
            beta=args.beta,
            delta=args.delta,
            loss_type=args.dpo_cpo_loss_type,
            judge=args.judge,
            max_tokens=args.max_completion_length,
            alpha=args.alpha,
        )

    elif args.train_mode == "cpo":
        test_loss, _, _, test_metrics = evaluate_cpo(
            model=model,
            dataset=test_set,
            batch_size=args.batch_size,
            num_batches=args.test_batches,
            max_seq_length=args.max_seq_length,
            beta=args.beta,
            delta=args.delta,
            loss_type=args.dpo_cpo_loss_type,
        )

        test_ppl = math.exp(test_loss)

        print(f"Test loss {test_loss:.3f}, Test ppl {test_ppl:.3f}")
        print("CPO Test Metrics:")
        for metric_name, metric_value in test_metrics.items():
            print(f"  {metric_name}: {float(metric_value):.3f}")

    elif args.train_mode == "grpo":
        if args.reference_model_path:
            reference_model, _ = load(args.reference_model_path)
        else:
            reference_model = model

        test_loss, _, test_rewards = evaluate_grpo(
            model=model,
            ref_model=reference_model.freeze(),
            dataset=test_set,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            num_batches=args.test_batches,
            max_seq_length=args.max_seq_length,
            beta=args.beta,
            group_size=args.group_size,
            epsilon=args.epsilon,
            epsilon_high=args.epsilon_high,
            temperature=args.temperature,
            max_tokens=args.max_seq_length,
        )

        test_ppl = math.exp(test_loss)

        rewards_str = ", ".join([f"{k}: {v:.3f}" for k, v in test_rewards.items()])
        print(
            f"Test loss {test_loss:.3f}, Test ppl {test_ppl:.3f}, Rewards: {rewards_str}"
        )

    elif args.train_mode == "sft":
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
    train_set, valid_set, test_set = load_dataset(args, tokenizer)

    if args.test and not args.train:
        if args.adapter_path != "":
            load_adapters(model, args.adapter_path)

    elif args.train:
        print("Training")
        train_model(args, model, tokenizer, train_set, valid_set, training_callback)
    else:
        raise ValueError("Must provide at least one of --train or --test")

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
