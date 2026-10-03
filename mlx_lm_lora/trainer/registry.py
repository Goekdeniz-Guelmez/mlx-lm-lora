"""Training mode registration and shared CLI-to-trainer configuration."""

from dataclasses import dataclass, fields
from typing import Any, Callable, Dict, Optional, Tuple, Type

from .cpo_trainer import CPOTrainingArgs, train_cpo
from .dpo_trainer import DPOTrainingArgs, train_dpo
from .dsla_trainer import DSLATrainingArgs, train_dsla
from .ftpo_trainer import FTPOTrainingArgs, train_ftpo
from .grpo_trainer import GRPOTrainingArgs, train_grpo
from .klpo_trainer import KLPOTrainingArgs, train_klpo
from .online_dpo_trainer import OnlineDPOTrainingArgs, train_online_dpo
from .orpo_trainer import ORPOTrainingArgs, train_orpo
from .ppo_trainer import PPOTrainingArgs, train_ppo
from .rlhf_reinforce_trainer import RLHFReinforceTrainingArgs, train_rlhf_reinforce
from .sft_trainer import SFTTrainingArgs, train_sft
from .xpo_trainer import XPOTrainingArgs, train_xpo

_QAT_OPTIONS = (
    "qat_enable",
    "qat_bits",
    "qat_group_size",
    "qat_mode",
    "qat_start_step",
    "qat_interval",
)
_NO_LONG_CONTEXT_WITH_QAT = ("seq_step_size",) + _QAT_OPTIONS
_ONLINE_INHERITED_OPTIONS = _NO_LONG_CONTEXT_WITH_QAT + ("judge_system",)


@dataclass(frozen=True)
class TrainingMode:
    """Connect a public mode name to its trainer and required inputs."""

    args_type: Type
    train: Callable
    train_inputs: Tuple[str, ...] = ()
    arg_aliases: Optional[Dict[str, str]] = None
    # Omitted inherited fields retain their trainer dataclass defaults.
    excluded_args: Tuple[str, ...] = ()
    passes_seq_step_size: bool = False
    uses_reward_functions: bool = False
    requires_reference_model: bool = False
    conditional_reference_model: Optional[Tuple[str, Any]] = None
    requires_judge: bool = False


# This table drives parser choices, trainer dispatch, and model requirements.
TRAINING_MODES = {
    "sft": TrainingMode(
        SFTTrainingArgs,
        train_sft,
        arg_aliases={"loss_type": "sft_loss_type"},
        passes_seq_step_size=True,
    ),
    "dpo": TrainingMode(
        DPOTrainingArgs,
        train_dpo,
        train_inputs=("ref_model",),
        arg_aliases={"loss_type": "dpo_cpo_loss_type"},
        passes_seq_step_size=True,
        requires_reference_model=True,
    ),
    "dsla": TrainingMode(
        DSLATrainingArgs,
        train_dsla,
        train_inputs=("ref_model",),
        arg_aliases={"loss_type": "dsla_loss"},
        passes_seq_step_size=True,
        conditional_reference_model=("dsla_loss", "dpo"),
    ),
    "ftpo": TrainingMode(
        FTPOTrainingArgs,
        train_ftpo,
        train_inputs=("ref_model",),
        excluded_args=_NO_LONG_CONTEXT_WITH_QAT,
        requires_reference_model=True,
    ),
    "cpo": TrainingMode(
        CPOTrainingArgs,
        train_cpo,
        arg_aliases={"loss_type": "dpo_cpo_loss_type"},
        excluded_args=_QAT_OPTIONS,
        passes_seq_step_size=True,
    ),
    "orpo": TrainingMode(ORPOTrainingArgs, train_orpo, passes_seq_step_size=True),
    "grpo": TrainingMode(
        GRPOTrainingArgs,
        train_grpo,
        train_inputs=("ref_model", "tokenizer", "reward_funcs"),
        excluded_args=_NO_LONG_CONTEXT_WITH_QAT + ("top_p", "top_k", "min_p"),
        uses_reward_functions=True,
        requires_reference_model=True,
    ),
    "klpo": TrainingMode(
        KLPOTrainingArgs,
        train_klpo,
        train_inputs=("tokenizer", "reward_funcs"),
        arg_aliases={
            "route": "klpo_route",
            "kl_estimator": "klpo_kl_estimator",
            "mc_samples": "klpo_mc_samples",
            "top_k": "klpo_top_k",
            "tail_floor": "klpo_tail_floor",
        },
        excluded_args=_NO_LONG_CONTEXT_WITH_QAT,
        uses_reward_functions=True,
    ),
    "online_dpo": TrainingMode(
        OnlineDPOTrainingArgs,
        train_online_dpo,
        train_inputs=(
            "tokenizer",
            "ref_model",
            "judge_model",
            "judge_tokenizer",
            "judge_config",
        ),
        arg_aliases={"loss_type": "dpo_cpo_loss_type"},
        excluded_args=_ONLINE_INHERITED_OPTIONS,
        requires_reference_model=True,
        requires_judge=True,
    ),
    "xpo": TrainingMode(
        XPOTrainingArgs,
        train_xpo,
        train_inputs=(
            "tokenizer",
            "ref_model",
            "judge_model",
            "judge_tokenizer",
            "judge_config",
        ),
        arg_aliases={"loss_type": "dpo_cpo_loss_type"},
        excluded_args=_ONLINE_INHERITED_OPTIONS + ("temperature",),
        requires_reference_model=True,
        requires_judge=True,
    ),
    "rlhf_reinforce": TrainingMode(
        RLHFReinforceTrainingArgs,
        train_rlhf_reinforce,
        train_inputs=(
            "tokenizer",
            "ref_model",
            "judge_model",
            "judge_tokenizer",
            "judge_config",
        ),
        excluded_args=_ONLINE_INHERITED_OPTIONS,
        requires_reference_model=True,
        requires_judge=True,
    ),
    "ppo": TrainingMode(
        PPOTrainingArgs,
        train_ppo,
        train_inputs=(
            "tokenizer",
            "ref_model",
            "judge_model",
            "judge_tokenizer",
            "judge_config",
        ),
        arg_aliases={"loss_type": "dpo_cpo_loss_type"},
        excluded_args=_ONLINE_INHERITED_OPTIONS,
        requires_reference_model=True,
        requires_judge=True,
    ),
}


def build_training_args(
    mode: TrainingMode, cli_args: Any, adapter_file: Optional[str] = None
) -> Any:
    """Build trainer options from matching CLI fields and mode-specific aliases."""
    mode_fields = {
        item.name
        for item in fields(mode.args_type)
        if item.name not in mode.excluded_args
    }
    if not mode.passes_seq_step_size:
        mode_fields.discard("seq_step_size")
    values = {
        name: getattr(cli_args, name)
        for name in mode_fields
        if hasattr(cli_args, name)
    }

    if "steps_per_save" in mode_fields:
        values["steps_per_save"] = cli_args.save_every
    if "seq_step_size" in mode_fields and mode.passes_seq_step_size:
        values["seq_step_size"] = (
            512 if getattr(cli_args, "efficient_long_context", False) else None
        )
    if adapter_file is not None and "adapter_file" in mode_fields:
        values["adapter_file"] = adapter_file

    for name, source in (mode.arg_aliases or {}).items():
        values[name] = getattr(cli_args, source)

    raw_weights = getattr(cli_args, "reward_weights", None)
    if "reward_weights" in mode_fields and raw_weights is not None:
        if isinstance(raw_weights, str):
            raw_weights = raw_weights.strip("[]").split(",")
        values["reward_weights"] = [float(weight) for weight in raw_weights]

    return mode.args_type(**values)


def needs_reference_model(mode: TrainingMode, cli_args: Any) -> bool:
    """Return whether this mode needs a separate frozen reference model."""
    if mode.requires_reference_model:
        return True
    condition = mode.conditional_reference_model
    return condition is not None and getattr(cli_args, condition[0]) == condition[1]
