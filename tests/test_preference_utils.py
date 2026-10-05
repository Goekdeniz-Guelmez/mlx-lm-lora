"""Regressions for shared offline preference scoring and evaluation."""

import unittest

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_lm_lora.trainer import cpo_trainer, dpo_trainer, orpo_trainer
from mlx_lm_lora.trainer.long_context import iter_cached_sft_chunks
from mlx_lm_lora.trainer.preference_utils import (
    accumulate_score_gradients,
    compute_scores_chunked,
    evaluate_preference_batches,
    get_token_scores,
)


class TrackingCache:
    def __init__(self):
        self.offset = 99
        self.resets = 0
        self.seen_offsets = []

    def reset(self):
        self.offset = 0
        self.resets += 1


class TableModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = mx.array([[0.0, 1.0, -1.0], [1.0, 0.0, -1.0], [0.0, -1.0, 1.0]])

    def __call__(self, inputs, cache=None):
        if cache is not None:
            cache.seen_offsets.append(cache.offset)
            cache.offset += inputs.shape[1]
        return self.weight[inputs]


class PreferenceUtilsTest(unittest.TestCase):
    def test_float32_scores_keep_existing_input_mask_semantics(self):
        class HalfModel:
            def __call__(self, inputs, cache=None):
                return mx.broadcast_to(
                    mx.array([60000.0, -60000.0, 0.0], dtype=mx.float16),
                    (*inputs.shape, 3),
                )

        tokens = mx.array([[0, 1, 2]])
        mask = mx.array([[1.0, 1.0, 0.0]])
        scores = get_token_scores(HalfModel(), tokens, mask)
        self.assertEqual(scores.dtype, mx.float32)
        self.assertTrue(mx.all(mx.isfinite(scores)).item())
        self.assertLess(scores[0, 1].item(), -1000.0)

    def test_chunk_scores_cover_boundaries_and_reset_each_sequence(self):
        model = TableModel()
        for length in (8, 9, 10):
            with self.subTest(length=length):
                tokens = mx.array([[i % 3 for i in range(length)]] * 2)
                masks = mx.array([[1.0] * length, [1.0] * (length - 2) + [0.0] * 2])
                expected = get_token_scores(model, tokens, masks).sum(-1)
                cache = TrackingCache()
                for _ in range(2):
                    actual = compute_scores_chunked(model, cache, tokens, masks, 4)
                    np.testing.assert_allclose(
                        np.array(actual), np.array(expected), rtol=1e-6
                    )
                self.assertEqual(cache.resets, 2)
                starts = [start for start, _ in iter_cached_sft_chunks(length, 4)]
                self.assertEqual(cache.seen_offsets, starts * 2)

    def test_chunk_gradients_match_full_pair_and_keep_accumulated_gradient(self):
        model = TableModel()
        chosen = mx.array([[0, 1, 2, 0, 2, 1, 0, 1, 2], [1, 0, 2, 1, 0, 2, 1, 0, 0]])
        rejected = mx.array([[0, 2, 1, 2, 0, 1, 2, 0, 1], [1, 2, 0, 2, 1, 0, 1, 2, 0]])
        masks = mx.array([[1.0] * 9, [1.0] * 7 + [0.0] * 2])
        chosen_weights = mx.array([-0.25, 0.5])
        rejected_weights = mx.array([0.1, -0.3])

        def full_loss(model):
            chosen_sum = get_token_scores(model, chosen, masks).sum(-1)
            rejected_sum = get_token_scores(model, rejected, masks).sum(-1)
            return (chosen_sum * chosen_weights + rejected_sum * rejected_weights).sum()

        _, expected = nn.value_and_grad(model, full_loss)(model)
        cache = TrackingCache()
        actual = accumulate_score_gradients(
            model, cache, chosen, masks, chosen_weights, 4
        )
        actual = accumulate_score_gradients(
            model, cache, rejected, masks, rejected_weights, 4, actual
        )
        np.testing.assert_allclose(
            np.array(actual["weight"]), np.array(expected["weight"]), atol=1e-6
        )
        self.assertEqual(cache.resets, 2)

    def test_dpo_model_scores_keep_reference_gradients_detached(self):
        model, reference = TableModel(), TableModel()
        chosen, rejected = mx.array([[0, 1, 2]]), mx.array([[0, 2, 1]])
        masks = mx.ones((1, 3))

        def loss(reference):
            return dpo_trainer.dpo_loss_from_model(
                model, reference, chosen, rejected, masks, masks, beta=0.1, delta=2.0
            )[0]

        _, gradient = nn.value_and_grad(reference, loss)(reference)
        self.assertTrue(
            mx.array_equal(gradient["weight"], mx.zeros_like(reference.weight)).item()
        )

    def test_evaluation_preserves_algorithm_reward_weighting_and_batch_limit(self):
        results = [
            (
                mx.array(2.0),
                mx.array([1.0, -1.0]),
                mx.array(2.0),
                {"margin": mx.array(3.0)},
            ),
            (
                mx.array(4.0),
                mx.array([3.0, -3.0]),
                mx.array(6.0),
                {"margin": mx.array(5.0)},
            ),
        ]
        for weight_rewards, expected_rewards in (
            (False, [0.5, -0.5]),
            (True, [2.5, -2.5]),
        ):
            loss, rewards, tokens, metrics = evaluate_preference_batches(
                [(result,) for result in results],
                -1,
                lambda result: result,
                weight_rewards=weight_rewards,
            )
            self.assertEqual(loss, 3.5)
            self.assertEqual(rewards, expected_rewards)
            self.assertEqual(tokens.item(), 8)
            self.assertEqual(metrics["margin"], 4.5)
        loss, rewards, tokens, _ = evaluate_preference_batches(
            [(result,) for result in results],
            1,
            lambda result: result,
            weight_rewards=True,
        )
        self.assertEqual(loss, 2)
        self.assertEqual(rewards, [1, -1])
        self.assertEqual(tokens.item(), 2)

    def test_algorithm_evaluation_matches_individual_batch_results(self):
        model = TableModel()
        dataset = [
            {"chosen": [0, 1], "rejected": [0, 2, 1]},
            {"chosen": [0, 1, 2, 1], "rejected": [0, 2]},
        ]
        options = dict(beta=0.1, delta=2.0, loss_type="sigmoid")
        algorithms = (
            (
                dpo_trainer.iterate_dpo_batches,
                lambda batch: dpo_trainer.dpo_loss_from_model(
                    model, None, *batch, **options
                ),
                lambda: dpo_trainer.evaluate_dpo(
                    model, None, dataset, 1, -1, 0.1, 2.0, 8, "sigmoid"
                ),
                False,
            ),
            (
                cpo_trainer.iterate_cpo_batches,
                lambda batch: cpo_trainer.cpo_loss_from_model(model, *batch, **options),
                lambda: cpo_trainer.evaluate_cpo(
                    model, dataset, 1, -1, 0.1, 2.0, 8, "sigmoid"
                ),
                False,
            ),
            (
                orpo_trainer.iterate_orpo_batches,
                lambda batch: orpo_trainer.orpo_loss_from_model(
                    model, *batch, beta=0.1
                ),
                lambda: orpo_trainer.evaluate_orpo(model, dataset, 1, -1, 0.1, 8),
                True,
            ),
        )
        for batches, loss_fn, evaluate, weight_rewards in algorithms:
            with self.subTest(algorithm=batches.__name__):
                results = [loss_fn(batch) for batch in batches(dataset, 1, 8)]
                count = sum(result[2].item() for result in results)
                expected_loss = (
                    sum(result[0].item() * result[2].item() for result in results)
                    / count
                )
                expected_rewards = (
                    sum(
                        np.array(result[1])
                        * (result[2].item() if weight_rewards else 1)
                        for result in results
                    )
                    / count
                )
                loss, rewards, tokens, metrics = evaluate()
                self.assertAlmostEqual(loss, expected_loss, places=6)
                np.testing.assert_allclose(rewards, expected_rewards, rtol=1e-6)
                self.assertEqual(tokens.item(), count)
                for key in metrics:
                    expected = (
                        sum(
                            result[3][key].item() * result[2].item()
                            for result in results
                        )
                        / count
                    )
                    self.assertAlmostEqual(metrics[key], expected, places=6)


if __name__ == "__main__":
    unittest.main()
