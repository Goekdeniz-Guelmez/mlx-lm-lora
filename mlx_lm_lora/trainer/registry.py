"""Algorithm registration and shared CLI-to-trainer configuration."""

from dataclasses import dataclass, fields
from typing import Any, Callable, Dict, Optional, Tuple, Type

from .cpo_trainer import CPOTrainingArgs, evaluate_cpo, train_cpo
from .dpo_trainer import DPOTrainingArgs, evaluate_dpo, train_dpo
from .dsla_trainer import DSLATrainingArgs, evaluate_dsla, train_dsla
from .ftpo_trainer import FTPOTrainingArgs, evaluate_ftpo, train_ftpo
from .grpo_trainer import GRPOTrainingArgs, evaluate_grpo, train_grpo
from .klpo_trainer import KLPOTrainingArgs, evaluate_klpo, train_klpo
from .online_dpo_trainer import (
    OnlineDPOTrainingArgs,
    evaluate_online_dpo,
    train_online_dpo,
)
from .orpo_trainer import ORPOTrainingArgs, evaluate_orpo, train_orpo
from .ppo_trainer import PPOTrainingArgs, evaluate_ppo, train_ppo
from .rlhf_reinforce_trainer import (
    RLHFReinforceTrainingArgs,
    evaluate_rlhf_reinforce,
    train_rlhf_reinforce,
)
from .sft_trainer import SFTTrainingArgs, evaluate_sft, get_sft_loss, train_sft
from .xpo_trainer import XPOTrainingArgs, evaluate_xpo, train_xpo

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
_EVALUATION_BATCH_OPTIONS = ("batch_size", "max_seq_length")
_PREFERENCE_EVALUATION_OPTIONS = _EVALUATION_BATCH_OPTIONS + (
    "beta",
    "delta",
    "loss_type",
)
_ROLLOUT_EVALUATION_OPTIONS = _EVALUATION_BATCH_OPTIONS + ("beta", "max_tokens")


@dataclass(frozen=True)
class TrainingMode:
    """Connect a public mode to its trainers, CLI options, and result fields."""

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
    evaluate: Optional[Callable] = None
    eval_options: Tuple[str, ...] = _EVALUATION_BATCH_OPTIONS
    eval_defaults: Optional[Dict[str, Any]] = None
    # None omits args; () uses all training options; other tuples select fields.
    eval_arg_fields: Optional[Tuple[str, ...]] = None
    # A single result field denotes a scalar evaluator return value.
    eval_result_fields: Tuple[Optional[str], ...] = ("loss", None, None, "metrics")
    eval_label: Optional[str] = None
    eval_simple_loss: bool = False


# This table drives parser choices, training/evaluation, and model requirements.
TRAINING_MODES = {
    "sft": TrainingMode(
        SFTTrainingArgs,
        train_sft,
        arg_aliases={"loss_type": "sft_loss_type"},
        passes_seq_step_size=True,
        evaluate=evaluate_sft,
        eval_options=_EVALUATION_BATCH_OPTIONS + ("loss", "recurrence_chunk_size"),
        eval_result_fields=("loss",),
    ),
    "dpo": TrainingMode(
        DPOTrainingArgs,
        train_dpo,
        train_inputs=("ref_model",),
        arg_aliases={"loss_type": "dpo_cpo_loss_type"},
        passes_seq_step_size=True,
        requires_reference_model=True,
        evaluate=evaluate_dpo,
        eval_options=_PREFERENCE_EVALUATION_OPTIONS,
    ),
    "dsla": TrainingMode(
        DSLATrainingArgs,
        train_dsla,
        train_inputs=("ref_model",),
        arg_aliases={"loss_type": "dsla_loss"},
        passes_seq_step_size=True,
        conditional_reference_model=("dsla_loss", "dpo"),
        evaluate=evaluate_dsla,
        eval_options=(),
        eval_arg_fields=(),
        eval_result_fields=("loss", "rewards", None, "metrics"),
        eval_simple_loss=True,
    ),
    "ftpo": TrainingMode(
        FTPOTrainingArgs,
        train_ftpo,
        train_inputs=("ref_model",),
        excluded_args=_NO_LONG_CONTEXT_WITH_QAT,
        requires_reference_model=True,
        evaluate=evaluate_ftpo,
        eval_arg_fields=(
            "lambda_mse_target",
            "tau_mse_target",
            "lambda_mse",
            "clip_epsilon_logits",
        ),
        eval_result_fields=("loss", "metrics"),
        eval_simple_loss=True,
    ),
    "cpo": TrainingMode(
        CPOTrainingArgs,
        train_cpo,
        arg_aliases={"loss_type": "dpo_cpo_loss_type"},
        excluded_args=_QAT_OPTIONS,
        passes_seq_step_size=True,
        evaluate=evaluate_cpo,
        eval_options=_PREFERENCE_EVALUATION_OPTIONS,
    ),
    "orpo": TrainingMode(
        ORPOTrainingArgs,
        train_orpo,
        passes_seq_step_size=True,
        evaluate=evaluate_orpo,
        eval_options=_EVALUATION_BATCH_OPTIONS + ("beta",),
        eval_result_fields=("loss", "rewards", None, "metrics"),
    ),
    "grpo": TrainingMode(
        GRPOTrainingArgs,
        train_grpo,
        train_inputs=("ref_model", "tokenizer", "reward_funcs"),
        excluded_args=_NO_LONG_CONTEXT_WITH_QAT + ("top_p", "top_k", "min_p"),
        uses_reward_functions=True,
        requires_reference_model=True,
        evaluate=evaluate_grpo,
        eval_options=_ROLLOUT_EVALUATION_OPTIONS
        + (
            "group_size",
            "epsilon",
            "epsilon_high",
            "grpo_loss_type",
            "end_answer_token",
            "temperature",
            "top_p",
            "top_k",
            "min_p",
        ),
        eval_defaults={
            "end_answer_token": None,
            "top_p": 1.0,
            "top_k": -1,
            "min_p": 0.0,
        },
        eval_result_fields=("loss", "tokens", "metrics"),
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
        evaluate=evaluate_klpo,
        eval_options=_ROLLOUT_EVALUATION_OPTIONS
        + (
            "route",
            "kl_estimator",
            "mc_samples",
            "top_k",
            "tail_floor",
            "temperature",
            "reward_weights",
        ),
        eval_result_fields=("loss", "tokens", "metrics"),
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
        evaluate=evaluate_online_dpo,
        eval_options=_PREFERENCE_EVALUATION_OPTIONS + ("max_tokens", "temperature"),
        eval_label="Online DPO",
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
        evaluate=evaluate_xpo,
        eval_options=_PREFERENCE_EVALUATION_OPTIONS + ("max_tokens", "alpha"),
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
        evaluate=evaluate_rlhf_reinforce,
        eval_options=_ROLLOUT_EVALUATION_OPTIONS,
        eval_label="RLHF Reinforce",
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
        evaluate=evaluate_ppo,
        eval_options=_ROLLOUT_EVALUATION_OPTIONS
        + ("epsilon", "loss_type", "temperature"),
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
        name: getattr(cli_args, name) for name in mode_fields if hasattr(cli_args, name)
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
        values["reward_weights"] = _reward_weights(raw_weights)

    return mode.args_type(**values)


def _reward_weights(raw_weights):
    if raw_weights is None:
        return None
    if isinstance(raw_weights, str):
        raw_weights = raw_weights.strip("[]").split(",")
    return [float(weight) for weight in raw_weights]


def build_evaluation_kwargs(mode: TrainingMode, cli_args: Any) -> Dict[str, Any]:
    """Map evaluation options without overriding omitted evaluator defaults."""
    aliases = {"max_tokens": "max_completion_length", "loss": "sft_loss_type"}
    aliases.update(mode.arg_aliases or {})
    defaults = mode.eval_defaults or {}
    values = {}
    for name in mode.eval_options:
        source = aliases.get(name, name)
        values[name] = (
            getattr(cli_args, source, defaults[name])
            if name in defaults
            else getattr(cli_args, source)
        )

    if "loss" in values:
        values["loss"] = get_sft_loss(values["loss"])
    if "reward_weights" in values:
        values["reward_weights"] = (
            _reward_weights(values["reward_weights"])
            if values["reward_weights"]
            else None
        )
    if mode.eval_arg_fields is not None:
        values["args"] = (
            mode.args_type(
                **{name: getattr(cli_args, name) for name in mode.eval_arg_fields}
            )
            if mode.eval_arg_fields
            else build_training_args(mode, cli_args)
        )
    return values


def needs_reference_model(mode: TrainingMode, cli_args: Any) -> bool:
    """Return whether this mode needs a separate frozen reference model."""
    if mode.requires_reference_model:
        return True
    condition = mode.conditional_reference_model
    return condition is not None and getattr(cli_args, condition[0]) == condition[1]
