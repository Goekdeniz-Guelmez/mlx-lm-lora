import math
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm.models import llama, qwen3, qwen3_5
from mlx_lm.tuner.utils import linear_to_lora_layers

from mlx_lm_lora import train
from mlx_lm_lora.trainer import datasets, dsla_trainer


def _model(tied=False):
    return llama.Model(
        llama.ModelArgs(
            model_type="llama",
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=3,
            num_attention_heads=2,
            num_key_value_heads=2,
            rms_norm_eps=1e-6,
            vocab_size=16,
            tie_word_embeddings=tied,
        )
    )


def _dataset():
    return [
        {
            "chosen": [1, 2, 3, 4],
            "rejected": [1, 2, 5],
            "chosen_prompt_length": 2,
            "rejected_prompt_length": 2,
        },
        {
            "chosen": [6, 7, 8],
            "rejected": [6, 9, 10, 11],
            "chosen_prompt_length": 1,
            "rejected_prompt_length": 1,
        },
    ]


def _batch():
    return next(dsla_trainer.iterate_dsla_batches(_dataset(), 2, 4))


class DSLAMathTest(unittest.TestCase):
    def test_matches_preprint_equations_for_all_components(self):
        chosen = np.array(
            [
                [[1.0, 2.0], [0.3, 0.2], [0.6, -0.5]],
                [[2.0, 1.0], [-0.2, 0.7], [0.1, 0.2]],
            ],
            dtype=np.float32,
        )
        rejected = np.array(
            [
                [[1.0, 2.0], [0.30002, 0.19997], [0.59999, -0.50003]],
                [[2.0, 1.0], [-0.20003, 0.69998], [0.09996, 0.20001]],
            ],
            dtype=np.float32,
        )
        response = mx.array([[0.0, 1.0, 1.0], [0.0, 1.0, 1.0]])
        prompt = mx.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        c, r, p = chosen[:, 1:].mean(1), rejected[:, 1:].mean(1), chosen[:, 0]
        normalize = lambda value: value / np.sqrt(
            (value * value).sum(-1, keepdims=True) + 1e-6
        )
        similarity = (normalize(p) * (normalize(c) - normalize(r))).sum(-1)
        directions = c - r
        mean = directions.mean(0)
        batch_direction = mean / max(np.linalg.norm(mean), 1e-8)
        direction = (normalize(directions) * batch_direction).sum(-1)
        expected_sim = np.logaddexp(0.0, 10.0 * (0.05 - similarity)).mean()
        expected_dir = np.logaddexp(0.0, -10.0 * direction).mean()
        for variant, expected in (
            ("both", (expected_sim + expected_dir) / 2),
            ("similarity", expected_sim),
            ("direction", expected_dir),
        ):
            with self.subTest(variant=variant):
                args = dsla_trainer.DSLATrainingArgs(latent_variant=variant)
                loss, metrics = dsla_trainer.latent_preference_loss(
                    mx.array(chosen),
                    mx.array(rejected),
                    response,
                    response,
                    prompt,
                    prompt,
                    args,
                )
                self.assertAlmostEqual(loss.item(), float(expected), places=5)
                if variant != "direction":
                    self.assertAlmostEqual(
                        metrics["latent_sim_margin"].item(),
                        float(similarity.mean()),
                        places=6,
                    )
                if variant != "similarity":
                    self.assertAlmostEqual(
                        metrics["latent_dir_margin"].item(),
                        float(direction.mean()),
                        places=5,
                    )

    def test_zero_directions_have_finite_loss_and_gradients_in_half_precision(self):
        response, prompt = mx.array([[0.0, 1.0], [0.0, 1.0]]), mx.array(
            [[1.0, 0.0], [1.0, 0.0]]
        )
        for dtype in (mx.float32, mx.float16, mx.bfloat16):
            with self.subTest(dtype=dtype):
                hidden = mx.ones((2, 2, 3), dtype=dtype)
                args = dsla_trainer.DSLATrainingArgs(latent_variant="direction")
                objective = lambda chosen: dsla_trainer.latent_preference_loss(
                    chosen, hidden, response, response, prompt, prompt, args
                )[0]
                loss, gradient = mx.value_and_grad(objective)(hidden)
                self.assertAlmostEqual(loss.item(), math.log(2), places=6)
                self.assertTrue(mx.all(mx.isfinite(gradient)).item())

    def test_latent_gradients_match_finite_differences(self):
        chosen = mx.array([[[1.0, 0.5], [0.8, 0.2]], [[0.3, 1.0], [0.2, 0.7]]])
        rejected = mx.array([[[1.0, 0.5], [-0.5, 0.1]], [[0.3, 1.0], [-0.2, 0.6]]])
        response, prompt = mx.array([[0.0, 1.0], [0.0, 1.0]]), mx.array(
            [[1.0, 0.0], [1.0, 0.0]]
        )
        args = dsla_trainer.DSLATrainingArgs()

        def objective(value):
            return dsla_trainer.latent_preference_loss(
                value, rejected, response, response, prompt, prompt, args
            )[0]

        gradient = mx.grad(objective)(chosen)
        perturbation = mx.zeros_like(chosen)
        perturbation[0, 1, 0] = 1e-3
        numerical = (
            objective(chosen + perturbation) - objective(chosen - perturbation)
        ) / 2e-3
        self.assertAlmostEqual(gradient[0, 1, 0].item(), numerical.item(), places=3)

    def test_pooling_includes_final_response_token_and_excludes_padding(self):
        hidden = mx.array([[[99.0], [1.0], [3.0], [5.0], [999.0]]])
        response, prompt = mx.array([[0.0, 1.0, 1.0, 1.0, 0.0]]), mx.array(
            [[1.0, 0.0, 0.0, 0.0, 0.0]]
        )
        for pooling, expected in (
            ("answer_mean", 3.0),
            ("last_token", 5.0),
            ("last_k_mean", 3.0),
            ("prompt_answer_mean", 27.0),
        ):
            with self.subTest(pooling=pooling):
                self.assertEqual(
                    dsla_trainer._pool(hidden, response, prompt, pooling).item(),
                    expected,
                )

    def test_logps_include_first_response_and_skip_prompt_and_padding(self):
        tokens = mx.array([[1, 2, 3, 0]])
        logits = mx.array(
            [[[0.0, 0.0, 3.0, 0.0], [0.0, 0.0, 0.0, 2.0], [5.0, 0.0, 0.0, 0.0]]]
        )
        response = mx.array([[0.0, 1.0, 1.0, 0.0]])
        sums, means, _ = dsla_trainer._sequence_logps(logits, tokens, response)
        expected = -nn.losses.cross_entropy(
            logits[:, :2], tokens[:, 1:3], reduction="none"
        ).sum()
        self.assertAlmostEqual(sums.item(), expected.item(), places=6)
        self.assertAlmostEqual(means.item(), expected.item() / 2, places=6)

    def test_odds_clamp_matches_preprint(self):
        logps = mx.array([0.0, -1e-8, -0.5, -1000.0])
        expected = np.minimum(np.array(logps), math.log1p(-1e-6))
        expected = expected - np.log(-np.expm1(expected))
        np.testing.assert_allclose(
            np.array(dsla_trainer._log_odds(logps)), expected, rtol=1e-6, atol=3e-6
        )


class DSLAForwardTest(unittest.TestCase):
    def test_native_logits_and_causal_prompt_are_preserved_at_every_layer(self):
        for tied in (False, True):
            model = _model(tied)
            tokens = mx.array([[1, 2, 3, 4], [1, 2, 8, 9]])
            native_logits = model(tokens)
            for layer in ("final", "middle", "late", "0", "2"):
                with self.subTest(tied=tied, layer=layer):
                    logits, hidden = dsla_trainer.forward_logits_and_hidden(
                        model, tokens, layer
                    )
                    self.assertTrue(
                        mx.allclose(logits, native_logits, atol=1e-6).item()
                    )
                    self.assertTrue(
                        mx.allclose(hidden[0, :2], hidden[1, :2], atol=1e-6).item()
                    )
                    self.assertEqual(hidden.shape, (2, 4, 8))

    def test_qwen3_and_nested_hybrid_qwen3_5_use_native_forward(self):
        qwen = qwen3.Model(
            qwen3.ModelArgs(
                model_type="qwen3",
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=2,
                num_attention_heads=2,
                rms_norm_eps=1e-6,
                vocab_size=16,
                num_key_value_heads=2,
                max_position_embeddings=128,
                rope_theta=10000.0,
                head_dim=4,
                tie_word_embeddings=False,
            )
        )
        hybrid_args = dict(
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            vocab_size=16,
            num_key_value_heads=2,
            head_dim=4,
            full_attention_interval=2,
            linear_num_value_heads=2,
            linear_num_key_heads=2,
            linear_key_head_dim=4,
            linear_value_head_dim=4,
            linear_conv_kernel_dim=2,
            rope_parameters={
                "type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 1.0,
            },
        )
        hybrid = qwen3_5.Model(
            qwen3_5.ModelArgs(model_type="qwen3_5", text_config=hybrid_args)
        )
        for model in (qwen, hybrid):
            model.train()
            tokens = mx.array([[1, 2, 3]])
            expected = model(tokens)
            for layer in ("final", "0", "1"):
                logits, hidden = dsla_trainer.forward_logits_and_hidden(
                    model, tokens, layer
                )
                self.assertTrue(mx.allclose(logits, expected, atol=1e-6).item())
                self.assertEqual(hidden.shape, (1, 3, 8))

    def test_unsupported_backbones_and_invalid_layers_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, "logits cannot"):
            dsla_trainer.forward_logits_and_hidden(nn.Linear(2, 2), mx.array([[1, 2]]))
        for layer in ("unknown", "-1", "3"):
            with self.assertRaises(ValueError):
                dsla_trainer.forward_logits_and_hidden(
                    _model(), mx.array([[1, 2]]), layer
                )

    def test_forward_hook_is_restored_after_failure(self):
        model = _model()
        original = type(model.model).__call__
        with mock.patch.object(
            type(model), "__call__", side_effect=RuntimeError("test")
        ):
            with self.assertRaises(RuntimeError):
                dsla_trainer.forward_logits_and_hidden(model, mx.array([[1, 2]]))
        self.assertIs(type(model.model).__call__, original)


class DSLAIntegrationTest(unittest.TestCase):
    def test_quantized_lora_and_dora_have_finite_adapter_gradients(self):
        for dora in (False, True):
            with self.subTest(dora=dora):
                model = llama.Model(
                    llama.ModelArgs(
                        model_type="llama",
                        hidden_size=64,
                        intermediate_size=128,
                        num_hidden_layers=2,
                        num_attention_heads=4,
                        num_key_value_heads=4,
                        rms_norm_eps=1e-6,
                        vocab_size=64,
                        tie_word_embeddings=False,
                    )
                )
                nn.quantize(model, group_size=64, bits=4)
                model.freeze()
                linear_to_lora_layers(
                    model, 1, {"rank": 2, "scale": 1.0, "dropout": 0.0}, use_dora=dora
                )
                batch = _batch()
                args = dsla_trainer.DSLATrainingArgs(
                    loss_type="orpo", latent_layer="middle"
                )
                loss, gradients = nn.value_and_grad(
                    model, lambda: dsla_trainer.dsla_loss(model, *batch, args=args)[0]
                )()
                self.assertTrue(mx.isfinite(loss).item())
                flat = tree_flatten(gradients)
                self.assertTrue(any("lora_b" in key for key, _ in flat))
                self.assertTrue(
                    all(mx.all(mx.isfinite(value)).item() for _, value in flat)
                )
                self.assertGreater(
                    sum(mx.abs(value).sum().item() for _, value in flat), 0.0
                )

    def test_partial_accumulation_matches_two_manual_updates(self):
        policy, expected = _model(), _model()
        mx.eval(policy.parameters())
        expected.update(policy.parameters())
        expected_optimizer = optim.SGD(learning_rate=1e-3)
        args = dsla_trainer.DSLATrainingArgs(
            loss_type="orpo",
            batch_size=2,
            iters=3,
            gradient_accumulation_steps=2,
            max_seq_length=4,
            steps_per_report=3,
        )
        batch = _batch()
        manual_grad = nn.value_and_grad(
            expected, lambda: dsla_trainer.dsla_loss(expected, *batch, args=args)[0]
        )
        # The first two identical microbatches average to one gradient; the final
        # singleton accumulation window must use its full gradient, not half.
        for _ in range(2):
            _, gradient = manual_grad()
            expected_optimizer.update(expected, gradient)
            mx.eval(expected.state, expected_optimizer.state)
        with tempfile.TemporaryDirectory() as directory:
            args.adapter_file = str(Path(directory) / "adapters.safetensors")
            dsla_trainer.train_dsla(
                policy, optim.SGD(learning_rate=1e-3), _dataset(), args=args
            )
        for (_, actual), (_, value) in zip(
            tree_flatten(policy.parameters()), tree_flatten(expected.parameters())
        ):
            self.assertTrue(mx.allclose(actual, value, atol=2e-6, rtol=2e-5).item())

    def test_zero_latent_weight_matches_each_base_preference_loss(self):
        model, reference, batch = _model(), _model(), _batch()
        for objective in ("dpo", "orpo", "cpo"):
            args = dsla_trainer.DSLATrainingArgs(loss_type=objective, latent_weight=0.0)
            result = dsla_trainer.dsla_loss(
                model, *batch, args=args, ref_model=reference
            )
            chosen_logits, _ = dsla_trainer.forward_logits_and_hidden(model, batch[0])
            rejected_logits, _ = dsla_trainer.forward_logits_and_hidden(model, batch[1])
            chosen = dsla_trainer._sequence_logps(
                chosen_logits[:, :-1], batch[0], batch[2]
            )
            rejected = dsla_trainer._sequence_logps(
                rejected_logits[:, :-1], batch[1], batch[3]
            )
            if objective == "orpo":
                odds = dsla_trainer._log_odds(chosen[1]) - dsla_trainer._log_odds(
                    rejected[1]
                )
                expected = (-chosen[1] - args.beta * nn.log_sigmoid(odds)).mean()
            else:
                ref_c = dsla_trainer._sequence_logps(
                    reference(batch[0][:, :-1]), batch[0], batch[2]
                )[0]
                ref_r = dsla_trainer._sequence_logps(
                    reference(batch[1][:, :-1]), batch[1], batch[3]
                )[0]
                margin = chosen[0] - rejected[0]
                if objective == "dpo":
                    margin = margin - (ref_c - ref_r)
                expected = -nn.log_sigmoid(args.beta * margin).mean()
            self.assertAlmostEqual(result[0].item(), expected.item(), places=6)
            self.assertIn("preference_loss", result[3])

    def test_all_preference_variants_produce_finite_nonzero_gradients(self):
        batch, reference = _batch(), _model()
        for objective in ("dpo", "orpo", "cpo"):
            for variant in (
                ("sigmoid",)
                if objective == "orpo"
                else ("sigmoid", "hinge", "ipo", "dpop")
            ):
                model = _model()
                args = dsla_trainer.DSLATrainingArgs(
                    loss_type=objective, dpo_cpo_loss_type=variant
                )
                value_and_grad = nn.value_and_grad(
                    model,
                    lambda: dsla_trainer.dsla_loss(
                        model, *batch, args=args, ref_model=reference
                    ),
                )
                result, gradients = value_and_grad()
                self.assertTrue(mx.isfinite(result[0]).item())
                flat = [value for _, value in tree_flatten(gradients)]
                self.assertTrue(
                    all(mx.all(mx.isfinite(value)).item() for value in flat)
                )
                self.assertGreater(
                    sum(mx.abs(value).sum().item() for value in flat), 0.0
                )

    def test_reference_has_no_gradients_and_is_required_only_for_dpo(self):
        model, reference, batch = _model(), _model(), _batch()
        args = dsla_trainer.DSLATrainingArgs()
        with self.assertRaisesRegex(ValueError, "frozen reference"):
            dsla_trainer.dsla_loss(model, *batch, args=args)
        with self.assertRaisesRegex(ValueError, "separate from the policy"):
            dsla_trainer.dsla_loss(model, *batch, args=args, ref_model=model)
        grad = nn.value_and_grad(
            reference,
            lambda: dsla_trainer.dsla_loss(
                model, *batch, args=args, ref_model=reference
            )[0],
        )()[1]
        self.assertEqual(
            sum(mx.abs(value).sum().item() for _, value in tree_flatten(grad)), 0.0
        )

    def test_compiled_training_flushes_accumulation_and_emits_latent_metrics(self):
        for objective in ("dpo", "orpo", "cpo"):
            with self.subTest(
                objective=objective
            ), tempfile.TemporaryDirectory() as directory:
                model, reference = _model(), _model()
                optimizer, callback = optim.SGD(learning_rate=1e-3), mock.Mock()
                args = dsla_trainer.DSLATrainingArgs(
                    loss_type=objective,
                    batch_size=2,
                    iters=3,
                    max_seq_length=4,
                    gradient_accumulation_steps=2,
                    steps_per_report=3,
                    steps_per_save=2,
                    steps_per_eval=2,
                    val_batches=1,
                    adapter_file=str(Path(directory) / "adapters.safetensors"),
                )
                dsla_trainer.train_dsla(
                    model, optimizer, _dataset(), _dataset(), args, reference, callback
                )
                self.assertEqual(optimizer.step.item(), 2)
                self.assertTrue(Path(args.adapter_file).exists())
                self.assertTrue(
                    (Path(directory) / "0000002_adapters.safetensors").exists()
                )
                self.assertIn(
                    "train_latent_loss", callback.on_train_loss_report.call_args.args[0]
                )
                self.assertIn(
                    "val_latent_loss", callback.on_val_loss_report.call_args.args[0]
                )

    def test_qat_preserves_reference_and_restores_projection_hooks(self):
        model, reference, batch = _model(), _model(), _batch()
        expected = reference(batch[0])
        mx.eval(expected)
        original = nn.Linear.__call__
        with tempfile.TemporaryDirectory() as directory:
            args = dsla_trainer.DSLATrainingArgs(
                batch_size=2,
                iters=2,
                max_seq_length=4,
                steps_per_report=2,
                qat_enable=True,
                qat_bits=4,
                qat_group_size=0,
                qat_start_step=1,
                adapter_file=str(Path(directory) / "adapters.safetensors"),
            )
            dsla_trainer.train_dsla(
                model,
                optim.SGD(learning_rate=1e-3),
                _dataset(),
                args=args,
                ref_model=reference,
            )
        self.assertIs(nn.Linear.__call__, original)
        self.assertTrue(mx.allclose(reference(batch[0]), expected, atol=1e-6).item())

    def test_qat_changes_policy_forward_without_changing_reference_forward(self):
        policy, reference, tokens = _model(), _model(), mx.array([[1, 2, 3]])
        policy_logits, reference_logits = policy(tokens), reference(tokens)
        mx.eval(policy_logits, reference_logits)
        args = dsla_trainer.DSLATrainingArgs(
            qat_enable=True, qat_bits=2, qat_group_size=0
        )
        originals = dsla_trainer._install_policy_qat(policy, args)
        try:
            self.assertFalse(mx.allclose(policy(tokens), policy_logits).item())
            self.assertTrue(mx.allclose(reference(tokens), reference_logits).item())
        finally:
            for cls, original in originals.items():
                cls.__call__ = original

    def test_checkpointing_preserves_intermediate_layer_gradients(self):
        model, batch = _model(), _batch()
        args = dsla_trainer.DSLATrainingArgs(loss_type="orpo", latent_layer="middle")
        value_and_grad = nn.value_and_grad(
            model, lambda: dsla_trainer.dsla_loss(model, *batch, args=args)[0]
        )
        expected_loss, expected_grad = value_and_grad()
        mx.eval(expected_loss, expected_grad)
        from mlx_lm_lora.trainer.sft_trainer import grad_checkpoint

        cls, original = type(model.layers[0]), type(model.layers[0]).__call__
        grad_checkpoint(model.layers[0])
        try:
            loss, gradient = value_and_grad()
            self.assertAlmostEqual(loss.item(), expected_loss.item(), places=5)
            for (_, value), (_, expected) in zip(
                tree_flatten(gradient), tree_flatten(expected_grad)
            ):
                self.assertTrue(
                    mx.allclose(value, expected, atol=2e-5, rtol=2e-5).item()
                )
        finally:
            cls.__call__ = original

    def test_cli_and_config_select_standalone_dsla(self):
        parsed = train.build_parser().parse_args(
            ["--train-mode", "dsla", "--dsla-loss", "cpo"]
        )
        self.assertEqual(parsed.train_mode, "dsla")
        config = types.SimpleNamespace(
            **{**train.CONFIG_DEFAULTS, "train_mode": "dsla", "dsla_loss": "cpo"}
        )
        args = train._dsla_training_args(config)
        self.assertEqual(args.loss_type, "cpo")
        self.assertEqual(args.latent_layer, "final")

    def test_dsla_rejects_invalid_config_and_missing_prompt_metadata(self):
        for changes in (
            {"loss_type": "ppo"},
            {"latent_gamma": 0.0},
            {"latent_weight": -1.0},
            {"latent_margin": float("nan")},
        ):
            with self.assertRaises(ValueError):
                dsla_trainer.DSLATrainingArgs(**changes)
        for example in (
            {"chosen": [1, 2], "rejected": [1, 2]},
            {
                "chosen": [1, 2],
                "rejected": [1, 2],
                "chosen_prompt_length": 2,
                "rejected_prompt_length": 1,
            },
        ):
            with self.assertRaisesRegex(ValueError, "nonempty prompt and response"):
                next(dsla_trainer.iterate_dsla_batches([example], 1, 2))
        with self.assertRaisesRegex(ValueError, "efficient_long_context"):
            dsla_trainer.train_dsla(
                _model(),
                None,
                _dataset(),
                args=dsla_trainer.DSLATrainingArgs(seq_step_size=2),
            )

    def test_distributed_batch_iterator_shards_by_worker_rank(self):
        world = mock.Mock()
        world.size.return_value, world.rank.return_value = 2, 1
        with mock.patch.object(mx.distributed, "init", return_value=world):
            batch = _batch()
        self.assertEqual(batch[0].shape[0], 1)
        self.assertEqual(batch[0][0, 0].item(), 6)

    def test_dsla_dataset_requires_matching_rendered_prompt_prefix(self):
        class Tokenizer:
            def apply_chat_template(self, messages, add_generation_prompt=False):
                text = "".join(
                    f"{message['role']}:{message['content']}|" for message in messages
                )
                if add_generation_prompt:
                    text += "assistant:"
                return [ord(char) for char in text]

        data = [{"prompt": "p", "chosen": "yes", "rejected": "no"}]
        config = types.SimpleNamespace(train_mode="dsla")
        dataset = datasets.create_dataset(data, Tokenizer(), config)
        self.assertGreater(
            len(dataset[0]["chosen"]), dataset[0]["chosen_prompt_length"]
        )
        tokenizer = mock.Mock()
        tokenizer.apply_chat_template.side_effect = [[1, 2, 3], [1, 2, 4], [9, 2]]
        with self.assertRaisesRegex(ValueError, "exact token prefix"):
            datasets.create_dataset(data, tokenizer, config)


if __name__ == "__main__":
    unittest.main()
