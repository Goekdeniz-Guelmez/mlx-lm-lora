import unittest

import mlx.core as mx

from mlx_lm_lora.trainer.grpo_trainer import (
    GRPOTrainingArgs,
    compute_log_importance_weights,
)


class ImportanceSamplingTest(unittest.TestCase):

    def test_grpo_defaults_to_token_level(self):
        self.assertEqual(GRPOTrainingArgs().importance_sampling_level, "token")

    def test_token_level_preserves_policy_gradient_at_zero_log_ratio(self):
        length_mask = mx.ones((2, 3))

        def objective(log_ratio):
            log_weights = compute_log_importance_weights(
                log_ratio, length_mask, "token"
            )
            return mx.exp(log_weights).sum()

        gradient = mx.grad(objective)(mx.zeros((2, 3)))

        self.assertTrue(mx.allclose(gradient, mx.ones((2, 3))).item())

    def test_none_level_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "token.*sequence"):
            compute_log_importance_weights(mx.zeros((1, 1)), mx.ones((1, 1)), None)


if __name__ == "__main__":
    unittest.main()
