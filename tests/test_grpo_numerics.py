"""Numerical and rollout regressions for the GRPO trainer."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from mlx_lm_lora.trainer import grpo_trainer as grpo


class TableModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = mx.array([[0.0, 1.0, -1.0], [1.0, 0.0, -1.0], [0.0, -1.0, 1.0]])

    def __call__(self, inputs):
        return self.weight[inputs]


def objective(logps, refs, mask, advantages=None, **kwargs):
    options = dict(
        beta=0.1,
        epsilon=0.2,
        epsilon_high=0.2,
        max_tokens=4,
        importance_sampling_level="token",
        grpo_loss_type="grpo",
    )
    options.update(kwargs)
    return grpo._grpo_objective(
        logps,
        refs,
        mask,
        mx.ones((logps.shape[0],)) if advantages is None else advantages,
        **options,
    )


class GRPONumericsTest(unittest.TestCase):
    def test_kl_matches_estimator_and_reference_has_no_gradient(self):
        policy = mx.array([[-2.0, -3.0]])
        reference = mx.array([[-1.0, -4.0]])
        mask = mx.ones((1, 2), dtype=mx.bool_)
        loss, _, metrics = objective(policy, reference, mask)
        expected = np.mean(np.expm1([1.0, -1.0]) - [1.0, -1.0])
        self.assertAlmostEqual(metrics["kl"].item(), expected, delta=3e-6)
        self.assertAlmostEqual(loss.item(), -1 + 0.1 * expected, places=6)
        ref_grad = mx.grad(lambda ref: objective(policy, ref, mask)[0])(reference)
        self.assertTrue(mx.array_equal(ref_grad, mx.zeros_like(reference)).item())
        policy_grad = mx.grad(lambda p: objective(p, reference, mask)[0])(policy)
        expected_grad = (-1 + 0.1 * (1 - np.exp([1.0, -1.0]))) / 2
        np.testing.assert_allclose(np.array(policy_grad)[0], expected_grad, rtol=1e-5)

    def test_extreme_ratios_have_finite_loss_and_gradient(self):
        policy = mx.array([[-1000.0, -1.0]])
        reference = mx.array([[-1.0, -1000.0]])
        mask = mx.ones((1, 2), dtype=mx.bool_)
        loss, _, metrics = objective(policy, reference, mask)
        grad = mx.grad(lambda p: objective(p, reference, mask)[0])(policy)
        self.assertTrue(mx.isfinite(loss).item())
        self.assertTrue(mx.all(mx.isfinite(grad)).item())
        self.assertEqual(metrics["kl_clip_ratio"].item(), 0.5)

    def test_near_identical_policies_have_nonnegative_small_kl(self):
        policy = mx.array([[-1.0, -1.0]])
        refs = policy + mx.array([[1e-4, -1e-4]])
        _, _, metrics = objective(policy, refs, mx.ones((1, 2), dtype=mx.bool_))
        self.assertGreaterEqual(metrics["kl"].item(), 0)
        self.assertAlmostEqual(metrics["kl"].item(), 5e-9, delta=1e-11)

    def test_masked_nan_and_empty_rows_do_not_poison_loss(self):
        logps = mx.array([[-1.0, float("nan")], [float("nan"), float("nan")]])
        refs = mx.array([[-2.0, float("inf")], [float("nan"), float("inf")]])
        mask = mx.array([[True, False], [False, False]])
        for variant in ("grpo", "bnpo", "dr_grpo"):
            with self.subTest(variant=variant):
                loss, tokens, metrics = objective(
                    logps, refs, mask, grpo_loss_type=variant
                )
                grad = mx.grad(
                    lambda p: objective(p, refs, mask, grpo_loss_type=variant)[0]
                )(logps)
                self.assertTrue(mx.isfinite(loss).item())
                self.assertEqual(tokens.item(), 1)
                self.assertTrue(all(mx.isfinite(v).item() for v in metrics.values()))
                self.assertTrue(mx.all(mx.isfinite(grad)).item())
                self.assertEqual(grad[0, 1].item(), 0)
                self.assertTrue(mx.all(grad[1] == 0).item())

    def test_loss_normalizations(self):
        logps = mx.zeros((2, 3))
        mask = mx.array([[True, False, False], [True, True, True]])
        for variant, expected in (("grpo", -2.0), ("bnpo", -2.5), ("dr_grpo", -1.25)):
            loss, _, _ = objective(
                logps, logps, mask, mx.array([1.0, 3.0]), beta=0, grpo_loss_type=variant
            )
            self.assertEqual(loss.item(), expected)

    def test_on_policy_gradient_survives_for_both_importance_levels(self):
        logps = mx.zeros((1, 3))
        mask = mx.array([[True, True, False]])
        for level in ("token", "sequence"):
            grad = mx.grad(
                lambda p: objective(
                    p, logps, mask, beta=0, importance_sampling_level=level
                )[0]
            )(logps)
            self.assertTrue(mx.allclose(grad, mx.array([[-0.5, -0.5, 0]])).item())

    def test_large_logits_are_not_downcast_to_float16(self):
        logits = mx.array([[[100000.0, 100001.0, 99999.0]]])
        selected = grpo._select_token_logps(logits, mx.array([[1]]), mx.array([[True]]))
        self.assertAlmostEqual(
            selected.item(), -np.log1p(np.exp(-1) + np.exp(-2)), places=6
        )
        grad = mx.grad(
            lambda x: grpo._select_token_logps(
                x, mx.array([[1]]), mx.array([[True]])
            ).sum()
        )(logits)
        self.assertTrue(mx.all(mx.isfinite(grad)).item())

    def test_float16_and_bfloat16_scoring_produce_float32(self):
        for dtype in (mx.float16, mx.bfloat16):
            logits = mx.array([[[0.0, -100.0, 1.0]]], dtype=dtype)
            result = grpo._select_token_logps(
                logits, mx.array([[1]]), mx.array([[True]])
            )
            self.assertEqual(result.dtype, mx.float32)
            self.assertAlmostEqual(result.item(), -101.31326, places=4)

    def test_prompt_conditioning_includes_first_completion_token(self):
        model = TableModel()
        batch = ([[0, 1], [2]], None, None, None, None)
        completions = [mx.array([0]), mx.array([1, 0]), mx.array([], dtype=mx.int32)]
        inputs, mask, lengths = grpo._prepare_grpo_inputs(batch, completions, [0, 1, 1])
        self.assertEqual(inputs.tolist(), [[0, 1, 0], [2, 1, 0], [2, 0, 0]])
        self.assertEqual(mask.tolist(), [[False, True], [True, True], [False, False]])
        self.assertEqual(lengths.tolist(), [1, 2, 0])
        logps = grpo._get_token_logps(model, inputs, mask)
        expected = nn.log_softmax(model.weight[1])[0]
        self.assertAlmostEqual(logps[0, 1].item(), expected.item(), places=6)

    def test_empty_completions_have_zero_loss_and_gradient(self):
        model = TableModel()

        def loss_fn(m):
            return grpo.grpo_loss(
                m,
                None,
                ([[1]], None, None, None, None),
                [mx.array([], dtype=mx.int32)],
                batch_indices=[0],
                advantages=mx.array([1.0]),
            )

        (loss, tokens, metrics), grads = nn.value_and_grad(model, loss_fn)(model)
        self.assertEqual(loss.item(), 0)
        self.assertEqual(tokens.item(), 0)
        self.assertTrue(mx.all(grads["weight"] == 0).item())
        self.assertTrue(all(mx.isfinite(v).item() for v in metrics.values()))

    def test_microbatches_preserve_full_loss_gradients_and_metrics(self):
        model = TableModel()
        reference = TableModel()
        reference.weight = reference.weight * 0.5
        completions = [mx.array([0, 1, 0]), mx.array([], dtype=mx.int32), mx.array([1])]
        for variant in ("grpo", "bnpo", "dr_grpo"):
            for level in ("token", "sequence"):
                kwargs = dict(
                    ref_model=reference,
                    batch=([[1], [2]], None, None, None, None),
                    completions=completions,
                    completion_texts=["a", "", "b"],
                    batch_indices=[0, 0, 1],
                    advantages=mx.array([1.0, -1.0, 2.0]),
                    grpo_loss_type=variant,
                    importance_sampling_level=level,
                    beta=0.1,
                )
                compute = nn.value_and_grad(model, grpo.grpo_loss)
                (full_loss, full_tokens, full_metrics), full_grad = compute(
                    model, **kwargs
                )
                (chunk_loss, chunk_tokens, chunk_metrics), chunk_grad = (
                    grpo._grpo_value_and_grad(compute, model, 1, **kwargs)
                )
                self.assertAlmostEqual(chunk_loss.item(), full_loss.item(), places=6)
                self.assertEqual(chunk_tokens.item(), full_tokens.item())
                self.assertTrue(
                    mx.allclose(
                        chunk_grad["weight"], full_grad["weight"], atol=1e-6
                    ).item()
                )
                for key, value in full_metrics.items():
                    self.assertAlmostEqual(
                        chunk_metrics[key].item(), value.item(), places=6, msg=key
                    )

    def test_reference_is_not_evaluated_when_beta_is_zero(self):
        def forbidden(_):
            raise AssertionError("reference must not run")

        loss, _, metrics = grpo.grpo_loss(
            TableModel(),
            forbidden,
            ([[1]], None, None, None, None),
            [mx.array([0])],
            batch_indices=[0],
            advantages=mx.array([1.0]),
            beta=0,
        )
        self.assertEqual(loss.item(), -1.0)
        self.assertEqual(metrics["kl"].item(), 0.0)


class GRPORewardsTest(unittest.TestCase):
    def calculate(self, funcs, **kwargs):
        return grpo.calculate_rewards_and_advantages(
            funcs,
            ["p"] * 3,
            ["a", "b", "c"],
            ["x"] * 3,
            ["math"] * 3,
            [4, 2, 4],
            [2, 4],
            **kwargs,
        )

    def test_rewards_are_called_once_with_types_and_keep_group_alignment(self):
        calls = []

        def reward(**kwargs):
            calls.append(kwargs)
            return [1.0, 9.0, 3.0]

        advantages, metrics = self.calculate([reward])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["types"], ["math"] * 3)
        np.testing.assert_allclose(
            np.array(advantages), [-0.9999, 0, 0.9999], rtol=1e-5
        )
        self.assertAlmostEqual(metrics["reward_mean"].item(), 13 / 3, places=6)

    def test_missing_reward_function_has_zero_coverage_without_nan_metrics(self):
        def absent(**kwargs):
            return None

        def reward(**kwargs):
            return [2.0] * 3

        advantages, metrics = self.calculate([absent, reward])
        self.assertTrue(mx.all(advantages == 0).item())
        self.assertEqual(metrics["absent_coverage"].item(), 0)
        self.assertTrue(all(mx.isfinite(v).item() for v in metrics.values()))

    def test_infinite_rewards_and_weights_are_rejected(self):
        def reward(**kwargs):
            return [float("inf"), 1, 2]

        with self.assertRaisesRegex(ValueError, "infinite reward"):
            self.calculate([reward])
        with self.assertRaisesRegex(ValueError, "weights must be finite"):
            self.calculate([reward], reward_weights=[float("nan")])

    def test_wrong_reward_count_is_rejected(self):
        def reward(**kwargs):
            return [1]

        with self.assertRaisesRegex(ValueError, "one reward per completion"):
            self.calculate([reward])


class GRPOLifecycleTest(unittest.TestCase):
    def test_generation_preserves_tokens_order_and_closes_generator(self):
        model = TableModel()

        class Tokenizer:
            eos_token_ids = {2}

            def encode(self, text, add_special_tokens=False):
                self.assert_text = text
                return [0, 1]

            def decode(self, tokens):
                return repr(tokens)

        class Generator:
            closed = False

            def __init__(self, model, **kwargs):
                self.options = kwargs

            def insert(self, prompts, max_tokens):
                self.once = False
                return [10, 20]

            def next_generated(self):
                if self.once:
                    return []
                self.once = True
                return [
                    SimpleNamespace(uid=20, token=1),
                    SimpleNamespace(uid=10, token=2),
                ]

            def close(self):
                Generator.closed = True

        with patch.object(grpo, "BatchGenerator", Generator):
            tokens, texts, indices = grpo.generate_grpo(
                model, Tokenizer(), [[1]], 4, 2, 1, "</answer>", 0.8, 0.95, 2, 0
            )
        self.assertEqual([t.tolist() for t in tokens], [[2], [1]])
        self.assertEqual(texts, ["[]", "[1]"])
        self.assertEqual(indices, [0, 0])
        self.assertTrue(Generator.closed)
        self.assertTrue(model.training)

    def test_real_batched_generation_with_tiny_transformer(self):
        from mlx_lm.models.qwen2 import Model, ModelArgs

        mx.random.seed(7)
        model = Model(
            ModelArgs(
                model_type="qwen2",
                hidden_size=16,
                num_hidden_layers=1,
                intermediate_size=32,
                num_attention_heads=2,
                rms_norm_eps=1e-5,
                vocab_size=8,
                num_key_value_heads=2,
            )
        )
        tokenizer = SimpleNamespace(eos_token_ids=set(), decode=lambda ids: str(ids))
        tokens, texts, indices = grpo.generate_grpo(
            model, tokenizer, [[1, 2]], 3, 2, 1, None, 0.0, 1.0, 0, 0.0
        )
        self.assertEqual(indices, [0, 0])
        self.assertEqual([t.size for t in tokens], [3, 3])
        self.assertEqual(tokens[0].tolist(), tokens[1].tolist())
        self.assertEqual(texts[0], str(tokens[0].tolist()))
        loss, count, metrics = grpo.grpo_loss(
            model,
            None,
            ([[1, 2]], None, None, None, None),
            tokens,
            batch_indices=indices,
            advantages=mx.array([1.0, -1.0]),
        )
        self.assertTrue(mx.isfinite(loss).item())
        self.assertEqual(count.item(), 6)
        self.assertEqual(metrics["kl"].item(), 0)

    def test_evaluation_restores_training_mode_for_empty_completions(self):
        model = TableModel()

        def reward(**kwargs):
            return [1.0]

        with patch.object(
            grpo,
            "generate_grpo",
            return_value=([mx.array([], dtype=mx.int32)], [""], [0]),
        ):
            loss, count, metrics = grpo.evaluate_grpo(
                model,
                None,
                [([1], [], "prompt", "answer")],
                None,
                1,
                1,
                0.0,
                0.2,
                None,
                1,
                8,
                4,
                0.8,
                1.0,
                0,
                0.0,
                reward_funcs=[reward],
            )
        self.assertEqual(loss, 0)
        self.assertEqual(count.item(), 0)
        self.assertTrue(all(np.isfinite(v) for v in metrics.values()))
        self.assertTrue(model.training)

    def test_batch_iterator_rejects_empty_dataset(self):
        with self.assertRaisesRegex(ValueError, "Dataset must"):
            next(grpo.iterate_grpo_batches([], 1, 8))

    def test_workers_receive_distinct_rows(self):
        dataset = [([i], [], str(i), "") for i in range(4)]
        for rank in (0, 1):
            world = SimpleNamespace(size=lambda: 2, rank=lambda: rank)
            with patch.object(grpo.mx.distributed, "init", return_value=world):
                batch = next(grpo.iterate_grpo_batches(dataset, 4, 8))
            self.assertEqual(batch[0], [[rank], [rank + 2]])

    def test_training_reuses_iterator_and_flushes_partial_accumulation(self):
        model = TableModel()
        optimizer = optim.SGD(learning_rate=0.01)
        starts = []

        def batches(**kwargs):
            starts.append(1)
            for token in (0, 1, 2):
                yield ([[token]], [[]], [str(token)], ["answer"], None)

        seen = []

        def generate(**kwargs):
            seen.append(kwargs["prompt_tokens"][0][0])
            return [mx.array([0]), mx.array([1])], ["a", "b"], [0, 0]

        def reward(**kwargs):
            return [1.0, 3.0]

        with tempfile.TemporaryDirectory() as directory, patch.object(
            grpo, "generate_grpo", generate
        ):
            args = grpo.GRPOTrainingArgs(
                iters=3,
                batch_size=1,
                group_size=2,
                gradient_accumulation_steps=2,
                steps_per_report=3,
                steps_per_save=10,
                beta=0,
                adapter_file=Path(directory) / "adapter.safetensors",
            )
            grpo.train_grpo(
                model,
                None,
                None,
                optimizer,
                [],
                reward_funcs=[reward],
                args=args,
                iterate_batches=batches,
            )
        self.assertEqual(starts, [1])
        self.assertEqual(seen, [0, 1, 2])
        self.assertEqual(optimizer.step.item(), 2)
        self.assertTrue(mx.all(mx.isfinite(model.weight)).item())


if __name__ == "__main__":
    unittest.main()
