import io
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import create_autospec, patch

import mlx.core as mx
import mlx.nn as nn

from mlx_lm_lora import train
from mlx_lm_lora.trainer import registry, sft_trainer


def _config(**overrides):
    return SimpleNamespace(**{**train.CONFIG_DEFAULTS, **overrides})


class RegistryTest(unittest.TestCase):
    def test_training_preserves_memory_and_speed_options(self):
        config = _config(
            iters=3,
            grad_checkpoint=True,
            gradient_accumulation_steps=3,
            micro_batch_size=1,
            recurrence_chunk_size=11,
            efficient_long_context=True,
            qat_enable=True,
            qat_bits=4,
        )
        for name, mode in registry.TRAINING_MODES.items():
            with self.subTest(mode=name):
                args = registry.build_training_args(mode, config, "adapter.safetensors")
                self.assertTrue(args.grad_checkpoint)
                self.assertEqual(args.gradient_accumulation_steps, 3)
                self.assertEqual(args.recurrence_chunk_size, 11)
                self.assertEqual(args.adapter_file, "adapter.safetensors")
                self.assertEqual(
                    args.seq_step_size,
                    512 if name in {"sft", "dpo", "dsla", "cpo", "orpo"} else None,
                )
                self.assertEqual(
                    args.qat_enable, name in {"sft", "dpo", "dsla", "orpo"}
                )
                if hasattr(args, "micro_batch_size"):
                    self.assertEqual(args.micro_batch_size, 1)

    def test_algorithm_defaults_and_aliases_remain_distinct(self):
        config = _config(temperature=0.2, top_p=0.4, top_k=7, min_p=0.3)
        grpo_args = registry.build_training_args(
            registry.TRAINING_MODES["grpo"], config
        )
        self.assertEqual(grpo_args.temperature, 0.2)
        self.assertEqual(
            (grpo_args.top_p, grpo_args.top_k, grpo_args.min_p), (0.95, 20, 0.0)
        )
        xpo_args = registry.build_training_args(registry.TRAINING_MODES["xpo"], config)
        self.assertEqual(
            xpo_args.temperature, registry.TRAINING_MODES["xpo"].args_type().temperature
        )
        for name, source in (
            ("sft", "sft_loss_type"),
            ("dpo", "dpo_cpo_loss_type"),
            ("cpo", "dpo_cpo_loss_type"),
            ("dsla", "dsla_loss"),
        ):
            with self.subTest(mode=name):
                args = registry.build_training_args(
                    registry.TRAINING_MODES[name], config
                )
                self.assertEqual(args.loss_type, getattr(config, source))

    def test_dsla_compatibility_helper_and_reference_requirement(self):
        config = _config(dsla_loss="cpo", latent_weight=0.3)
        mode = registry.TRAINING_MODES["dsla"]
        self.assertEqual(
            train._dsla_training_args(config),
            registry.build_training_args(mode, config),
        )
        self.assertFalse(registry.needs_reference_model(mode, config))
        config.dsla_loss = "dpo"
        self.assertTrue(registry.needs_reference_model(mode, config))

    def test_evaluation_dispatch_contracts(self):
        config = _config(recurrence_chunk_size=11, reward_weights="[1, 2]")
        batch_options = {"batch_size": 1, "max_seq_length": 2048}
        preference_options = {
            **batch_options,
            "beta": 0.1,
            "delta": 50.0,
            "loss_type": "sigmoid",
        }
        rollout_options = {**batch_options, "beta": 0.1, "max_tokens": 512}
        contracts = {
            "sft": {
                **batch_options,
                "loss": sft_trainer.default_loss,
                "recurrence_chunk_size": 11,
            },
            "dpo": preference_options,
            "cpo": preference_options,
            "orpo": {**batch_options, "beta": 0.1},
            "dsla": {"args": train._dsla_training_args(config)},
            "ftpo": {
                **batch_options,
                "args": registry.TRAINING_MODES["ftpo"].args_type(
                    lambda_mse_target=0.05,
                    tau_mse_target=1.0,
                    lambda_mse=0.4,
                    clip_epsilon_logits=2.0,
                ),
            },
            "online_dpo": {**preference_options, "max_tokens": 512, "temperature": 0.8},
            "xpo": {**preference_options, "max_tokens": 512, "alpha": 1e-5},
            "ppo": {
                **rollout_options,
                "epsilon": 1e-4,
                "loss_type": "sigmoid",
                "temperature": 0.8,
            },
            "rlhf_reinforce": rollout_options,
            "grpo": {
                **rollout_options,
                "group_size": 4,
                "epsilon": 1e-4,
                "epsilon_high": None,
                "grpo_loss_type": "grpo",
                "end_answer_token": None,
                "temperature": 0.8,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
            },
            "klpo": {
                **rollout_options,
                "route": "token",
                "kl_estimator": "mc",
                "mc_samples": 128,
                "top_k": 128,
                "tail_floor": 1e-6,
                "temperature": 0.8,
                "reward_weights": [1.0, 2.0],
            },
        }
        model, tokenizer, reference, judge, judge_tokenizer, dataset = [
            object() for _ in range(6)
        ]
        reward_functions = [object()]
        metrics = {"accuracy": 0.5}
        self.assertEqual(set(contracts), set(registry.TRAINING_MODES))
        for name, mode in registry.TRAINING_MODES.items():
            with self.subTest(mode=name):
                config.train_mode = name
                expected = {
                    "model": model,
                    "dataset": dataset,
                    "num_batches": 500,
                    **contracts[name],
                }
                if name in {
                    "dpo",
                    "dsla",
                    "ftpo",
                    "grpo",
                    "online_dpo",
                    "xpo",
                    "ppo",
                    "rlhf_reinforce",
                }:
                    expected["ref_model"] = reference
                if name in {
                    "online_dpo",
                    "xpo",
                    "ppo",
                    "rlhf_reinforce",
                    "grpo",
                    "klpo",
                }:
                    expected["tokenizer"] = tokenizer
                if name in {"online_dpo", "xpo", "ppo", "rlhf_reinforce"}:
                    expected.update(
                        judge_model=judge,
                        judge_tokenizer=judge_tokenizer,
                        judge_config={},
                    )
                if name in {"grpo", "klpo"}:
                    expected["reward_funcs"] = reward_functions

                results = (1.0, [0.2, 0.1], 10, metrics)
                if name == "sft":
                    results = 1.0
                elif name == "ftpo":
                    results = (1.0, metrics)
                elif name in {"grpo", "klpo"}:
                    results = (1.0, 10, metrics)
                elif name == "rlhf_reinforce":
                    results = (1.0, [], 10, metrics)
                evaluator = create_autospec(mode.evaluate, return_value=results)
                output = io.StringIO()
                with (
                    patch.dict(
                        registry.TRAINING_MODES,
                        {name: replace(mode, evaluate=evaluator)},
                    ),
                    patch.object(
                        train,
                        "_configured_reward_functions",
                        return_value=reward_functions,
                    ),
                    redirect_stdout(output),
                ):
                    train.evaluate_model(
                        config,
                        model,
                        tokenizer,
                        reference,
                        judge,
                        judge_tokenizer,
                        dataset,
                    )
                evaluator.assert_called_once_with(**expected)
                self.assertIn("1.000", output.getvalue())
                if name != "sft":
                    self.assertIn("accuracy:", output.getvalue())

    def test_sft_evaluation_reports_real_scalar_result(self):
        class UniformModel(nn.Module):
            def __call__(self, inputs, cache=None):
                del cache
                return mx.zeros((*inputs.shape, 4))

        model = UniformModel()
        dataset = [[0, 1, 2, 3]]
        loss = sft_trainer.evaluate_sft(
            model, dataset, batch_size=1, num_batches=1, max_seq_length=8
        )
        self.assertIsInstance(loss, float)

        output = io.StringIO()
        with redirect_stdout(output):
            train.evaluate_model(
                _config(train_mode="sft", test_batches=1, max_seq_length=8),
                model,
                tokenizer=None,
                test_set=dataset,
            )
        self.assertIn("1.386", output.getvalue())
        self.assertIn("4.000", output.getvalue())

    def test_evaluation_handles_sft_loss_and_reward_weights(self):
        config = _config(sft_loss_type="chunked_nll", reward_weights=[0.5, 2])
        sft_options = registry.build_evaluation_kwargs(
            registry.TRAINING_MODES["sft"], config
        )
        self.assertIs(sft_options["loss"], sft_trainer.chunked_nll_loss)
        klpo_options = registry.build_evaluation_kwargs(
            registry.TRAINING_MODES["klpo"], config
        )
        self.assertEqual(klpo_options["reward_weights"], [0.5, 2.0])

    def test_invalid_reward_configuration_does_not_evaluate(self):
        config = _config(train_mode="grpo")
        mode = registry.TRAINING_MODES["grpo"]
        evaluator = create_autospec(mode.evaluate)
        with (
            patch.dict(
                registry.TRAINING_MODES, {"grpo": replace(mode, evaluate=evaluator)}
            ),
            patch.object(train, "_configured_reward_functions", return_value=None),
            redirect_stdout(io.StringIO()),
        ):
            train.evaluate_model(config, object(), object())
        evaluator.assert_not_called()


if __name__ == "__main__":
    unittest.main()
