"""Shared reward validation and KLPO microbatch equivalence regressions."""

import unittest

import mlx.core as mx
import numpy as np
from mlx import nn

from mlx_lm_lora.trainer import klpo_trainer as klpo
from mlx_lm_lora.trainer.rollout_utils import evaluate_rewards


class RewardEvaluationTest(unittest.TestCase):
    def test_weighted_partial_rewards_preserve_order_and_callback_metadata(self):
        calls = []

        def first(**kwargs):
            calls.append(kwargs)
            return [None, 2.0, float("nan")]

        def second(**kwargs):
            calls.append(kwargs)
            return [3.0, None, 5.0]

        rewards, metrics = evaluate_rewards(
            [first, second],
            ["p0", "p1", "p2"],
            ["a", "b", "c"],
            ["x", "y", "z"],
            ["math", None, "code"],
            [2.0, -1.0],
        )
        np.testing.assert_array_equal(rewards, [-3.0, 4.0, -5.0])
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0], calls[1])
        self.assertEqual(calls[0]["types"], ["math", None, "code"])
        self.assertEqual(metrics["first_coverage"], 1 / 3)
        self.assertEqual(metrics["second_mean"], 4.0)
        self.assertEqual(metrics["second_std"], 1.0)

    def test_completion_without_any_applicable_reward_is_rejected(self):
        def absent(**kwargs):
            return None

        def partial(**kwargs):
            return [1.0, None]

        with self.assertRaisesRegex(RuntimeError, "completion 1"):
            evaluate_rewards([absent, partial], ["p"] * 2, ["a"] * 2, [], None)


class TableModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = mx.array([[0.0, 1.0, -1.0], [1.0, 0.0, -1.0], [0.0, -1.0, 1.0]])

    def __call__(self, inputs):
        return self.weight[inputs]


class KLPOMicrobatchTest(unittest.TestCase):
    def test_token_and_binary_sequence_routes_match_full_batch(self):
        model = TableModel()
        completions = [mx.array([0, 1, 0]), mx.array([1]), mx.array([2, 0])]
        q_logps = mx.log(mx.array([0.2, 0.5, 0.3]))
        records = []
        for completion in completions:
            length = completion.size
            records.append(
                {
                    "action_logps": q_logps[completion],
                    "mc_ids": mx.broadcast_to(mx.array([0, 1]), (length, 2)),
                    "mc_logps": mx.broadcast_to(q_logps[:2], (length, 2)),
                    "head_ids": mx.broadcast_to(mx.array([1, 2]), (length, 2)),
                    "head_logps": mx.broadcast_to(q_logps[1:], (length, 2)),
                    "full_logps": mx.broadcast_to(q_logps, (length, 3)),
                }
            )
        rewards = mx.array([1.0, 3.0, -1.0])
        kwargs = {
            "batch": ([[1, 2], [2]], None, None, None, None),
            "completions": completions,
            "completion_texts": ["a", "b", "c"],
            "batch_indices": [0, 0, 1],
            "rollout_records": records,
            "rewards": rewards,
            # Terminal reward statistics are computed once over the rollout.
            "reward_metrics": {
                "reward_mean": rewards.mean(),
                "reward_std": rewards.std(),
            },
            "beta": 0.1,
            "mc_samples": 2,
            "top_k": 2,
            "max_tokens": 3,
        }
        compute = nn.value_and_grad(model, klpo.klpo_loss)
        for route in ("token", "sequence"):
            for estimator in ("binary", "mc", "topk", "full"):
                if route == "sequence" and estimator != "binary":
                    continue
                with self.subTest(route=route, estimator=estimator):
                    options = dict(kwargs, route=route, kl_estimator=estimator)
                    (loss, tokens, metrics), grads = compute(model, **options)
                    (chunk_loss, chunk_tokens, chunk_metrics), chunk_grads = (
                        klpo._klpo_value_and_grad(compute, model, 2, **options)
                    )
                    self.assertAlmostEqual(chunk_loss.item(), loss.item(), places=5)
                    self.assertEqual(chunk_tokens.item(), tokens.item())
                    self.assertTrue(
                        mx.allclose(
                            chunk_grads["weight"], grads["weight"], atol=1e-5
                        ).item()
                    )
                    for key, value in metrics.items():
                        self.assertAlmostEqual(
                            chunk_metrics[key].item(), value.item(), places=5, msg=key
                        )
                    eval_loss, eval_tokens, eval_metrics = klpo._klpo_microbatches(
                        klpo.klpo_loss, model, 1, **options
                    )
                    self.assertAlmostEqual(eval_loss.item(), loss.item(), places=5)
                    self.assertEqual(eval_tokens.item(), tokens.item())
                    for key, value in metrics.items():
                        self.assertAlmostEqual(
                            eval_metrics[key].item(), value.item(), places=5, msg=key
                        )


if __name__ == "__main__":
    unittest.main()
