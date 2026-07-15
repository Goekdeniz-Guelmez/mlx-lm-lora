import unittest

import mlx.core as mx

from mlx_lm_lora.trainer import (
    cpo_trainer,
    dpo_trainer,
    dlpo,
    ftpo_trainer,
    grpo_trainer,
    online_dpo_trainer,
    orpo_trainer,
    ppo_trainer,
    rlhf_reinforce_trainer,
    sft_trainer,
    xpo_trainer,
)


def _scalar(value):
    return float(value.item())


def _preference_inputs():
    return {
        "policy_chosen_score": mx.array([-1.0, -2.0]),
        "policy_rejected_score": mx.array([-2.0, -3.0]),
        "reference_chosen_score": mx.array([-1.5, -2.5]),
        "reference_rejected_score": mx.array([-2.0, -3.0]),
        "chosen_masks": mx.ones((2, 3)),
        "rejected_masks": mx.ones((2, 2)),
        "beta": 0.1,
        "delta": 2.0,
    }


class SFTTrainerTest(unittest.TestCase):
    def test_training_argument_defaults(self):
        args = sft_trainer.SFTTrainingArgs()
        self.assertEqual(args.loss_type, "nll")
        self.assertEqual(args.gradient_accumulation_steps, 1)
        self.assertFalse(args.qat_enable)

    def test_get_sft_loss_selects_supported_losses(self):
        self.assertIs(sft_trainer.get_sft_loss("nll"), sft_trainer.default_loss)
        self.assertIs(
            sft_trainer.get_sft_loss("chunked_nll"),
            sft_trainer.chunked_nll_loss,
        )
        self.assertIs(sft_trainer.get_sft_loss("dft"), sft_trainer.dft_loss)

    def test_get_sft_loss_rejects_unknown_loss(self):
        with self.assertRaisesRegex(ValueError, "Unknown SFT loss type"):
            sft_trainer.get_sft_loss("unknown")

    def test_chunked_nll_matches_nll_across_multiple_chunks(self):
        class UniformModel:
            def __call__(self, inputs, cache=None):
                del cache
                return mx.zeros((*inputs.shape, 4))

        batch = mx.array([[index % 4 for index in range(301)]])
        lengths = mx.array([[1, 300]])
        nll, nll_tokens = sft_trainer.default_loss(UniformModel(), batch, lengths)
        chunked_nll, chunked_tokens = sft_trainer.chunked_nll_loss(
            UniformModel(), batch, lengths
        )

        self.assertAlmostEqual(_scalar(chunked_nll), _scalar(nll), places=6)
        self.assertEqual(_scalar(chunked_tokens), _scalar(nll_tokens))

    def test_fake_quantization_preserves_shape_and_bounds_error(self):
        values = mx.array([[-2.0, -0.2, 0.3, 2.0, 0.7]])
        quantized = sft_trainer._symmetric_fake_quantize_tensor(values, 4, 2)
        self.assertEqual(quantized.shape, values.shape)
        self.assertTrue(mx.all(mx.isfinite(quantized)).item())
        self.assertLessEqual(_scalar(mx.max(mx.abs(quantized))), 2.01)

    def test_fake_quantization_supports_per_tensor_mode(self):
        values = mx.array([-1.0, -0.25, 0.5, 1.0])
        quantized = sft_trainer._symmetric_fake_quantize_tensor(values, 8, 0)
        self.assertEqual(quantized.shape, values.shape)
        self.assertTrue(mx.all(mx.isfinite(quantized)).item())

    def test_find_cache_offset_searches_nested_caches(self):
        class Cache:
            offset = 7

        self.assertEqual(sft_trainer._find_cache_offset([None, [Cache()]]), 7)
        self.assertIsNone(sft_trainer._find_cache_offset(None))

    def test_reset_prompt_cache_calls_reset_protocol(self):
        class Cache:
            def __init__(self):
                self.was_reset = False

            def reset(self):
                self.was_reset = True

        cache = Cache()
        self.assertIs(sft_trainer.reset_prompt_cache(cache), cache)
        self.assertTrue(cache.was_reset)

    def test_iterate_batches_rejects_too_small_dataset(self):
        iterator = sft_trainer.iterate_batches([[1, 2]], 2, 16)
        with self.assertRaisesRegex(ValueError, "at least batch_size=2"):
            next(iterator)


class DPOTrainerTest(unittest.TestCase):
    def test_dpo_defaults(self):
        args = dpo_trainer.DPOTrainingArgs()
        self.assertEqual(args.loss_type, "sigmoid")
        self.assertEqual(args.beta, 0.1)
        self.assertEqual(args.latent_weight, 0.1)
        self.assertEqual(args.latent_pooling, "answer_mean")

    def test_dpo_all_loss_variants_are_finite(self):
        inputs = _preference_inputs()
        for loss_type in ("sigmoid", "hinge", "ipo", "dpop"):
            with self.subTest(loss_type=loss_type):
                loss, reward, tokens, metrics = dpo_trainer.dpo_loss(
                    **inputs, loss_type=loss_type
                )
                self.assertTrue(mx.isfinite(loss).item())
                self.assertEqual(reward.shape, (2,))
                self.assertEqual(_scalar(tokens), 10.0)
                self.assertIn("margins", metrics)

    def test_dpo_rejects_unknown_loss(self):
        with self.assertRaisesRegex(ValueError, "Unknown loss type"):
            dpo_trainer.dpo_loss(**_preference_inputs(), loss_type="bad")

    def test_dpo_compute_score_uses_mean_only_for_ipo(self):
        scores = mx.array([[2.0, 4.0]])
        mask = mx.ones((1, 2))
        self.assertEqual(_scalar(dpo_trainer.compute_score(scores, mask, "ipo")), 3.0)
        self.assertEqual(
            _scalar(dpo_trainer.compute_score(scores, mask, "sigmoid")), 6.0
        )

    def test_dpo_batch_iterator_pads_and_tracks_masks(self):
        data = [
            {"chosen": [1, 2, 3], "rejected": [4, 5]},
            {"chosen": [6, 7], "rejected": [8, 9, 10, 11]},
        ]
        chosen, rejected, chosen_mask, rejected_mask = next(
            dpo_trainer.iterate_dpo_batches(data, 2, 3)
        )
        self.assertEqual(chosen.shape, (2, 3))
        self.assertEqual(rejected.shape, (2, 3))
        self.assertEqual(_scalar(chosen_mask.sum()), 5.0)
        self.assertEqual(_scalar(rejected_mask.sum()), 5.0)

    def test_dlpo_batch_iterator_splits_prompt_and_response_masks(self):
        data = [
            {"chosen": [1, 2, 3, 4], "rejected": [5, 6, 7],
             "chosen_prompt_length": 2, "rejected_prompt_length": 2},
            {"chosen": [8, 9, 10], "rejected": [11, 12, 13, 14],
             "chosen_prompt_length": 1, "rejected_prompt_length": 1},
        ]
        batch = next(dpo_trainer.iterate_dpo_batches(
            data, 2, 4, include_prompt_masks=True
        ))
        self.assertEqual(len(batch), 6)
        self.assertEqual(_scalar(batch[2].sum()), 4.0)
        self.assertEqual(_scalar(batch[4].sum()), 3.0)


class DLPOTest(unittest.TestCase):
    def test_latent_reporting_includes_only_available_metrics(self):
        report = dlpo.format_latent_metrics(
            {"latent_loss": 0.5, "latent_sim_margin": 0.25}
        )
        self.assertEqual(report, ", latent_loss 0.500, sim_margin 0.250")

    def test_latent_loss_is_finite_and_reports_both_components(self):
        args = dpo_trainer.DPOTrainingArgs(loss_type="dlpo-dpo")
        chosen = mx.array([[[1.0, 0.0], [0.8, 0.2], [1.0, 1.0]],
                           [[0.0, 1.0], [0.2, 0.8], [1.0, 1.0]]])
        rejected = mx.array([[[1.0, 0.0], [-0.8, 0.2], [1.0, 1.0]],
                             [[0.0, 1.0], [-0.2, 0.8], [1.0, 1.0]]])
        response = mx.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
        prompt = mx.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        loss, metrics = dlpo.latent_preference_loss(
            chosen, rejected, response, response, prompt, prompt, args
        )
        self.assertTrue(mx.isfinite(loss).item())
        self.assertIn("latent_sim_loss", metrics)
        self.assertIn("latent_dir_loss", metrics)


class FTPOTrainerTest(unittest.TestCase):
    def test_ftpo_defaults_match_antidoom(self):
        args = ftpo_trainer.FTPOTrainingArgs()
        self.assertEqual(args.lambda_mse_target, 0.05)
        self.assertEqual(args.tau_mse_target, 1.0)
        self.assertEqual(args.lambda_mse, 0.4)
        self.assertEqual(args.clip_epsilon_logits, 2.0)

    def test_ftpo_loss_is_finite_and_reports_wins(self):
        policy = mx.array([[0.0, 3.0, -1.0, 0.5]])
        reference = mx.zeros_like(policy)
        loss, samples, metrics = ftpo_trainer.ftpo_loss(
            policy, reference, mx.array([[1, 3]]), mx.array([[1.0, 1.0]]), mx.array([2])
        )
        self.assertTrue(mx.isfinite(loss).item())
        self.assertEqual(_scalar(samples), 1.0)
        self.assertEqual(_scalar(metrics["chosen_win"]), 1.0)
        self.assertGreaterEqual(_scalar(metrics["margin_win"]), 0.5)

    def test_ftpo_clipped_pair_has_no_preference_loss(self):
        policy = mx.array([[0.0, 4.0, 0.0]])
        loss, _, metrics = ftpo_trainer.ftpo_loss(
            policy,
            policy,
            mx.array([[1]]),
            mx.array([[1.0]]),
            mx.array([2]),
            lambda_mse=0.0,
            lambda_mse_target=0.0,
            clip_epsilon_logits=2.0,
        )
        self.assertAlmostEqual(_scalar(loss), 0.0, places=7)
        self.assertAlmostEqual(_scalar(metrics["active_weight"]), 0.0, places=7)


class CPOTrainerTest(unittest.TestCase):
    def test_cpo_all_loss_variants_are_finite(self):
        inputs = _preference_inputs()
        inputs.pop("reference_chosen_score")
        inputs.pop("reference_rejected_score")
        for loss_type in ("sigmoid", "hinge", "ipo", "dpop"):
            with self.subTest(loss_type=loss_type):
                loss, reward, tokens, metrics = cpo_trainer.cpo_loss(
                    **inputs, loss_type=loss_type
                )
                self.assertTrue(mx.isfinite(loss).item())
                self.assertEqual(reward.shape, (2,))
                self.assertEqual(_scalar(tokens), 10.0)
                self.assertIn("accuracies", metrics)

    def test_cpo_rejects_unknown_loss(self):
        inputs = _preference_inputs()
        inputs.pop("reference_chosen_score")
        inputs.pop("reference_rejected_score")
        with self.assertRaisesRegex(ValueError, "Unknown loss type"):
            cpo_trainer.cpo_loss(**inputs, loss_type="bad")


class OnlineDPOTrainerTest(unittest.TestCase):
    def test_online_dpo_defaults(self):
        args = online_dpo_trainer.OnlineDPOTrainingArgs()
        self.assertEqual(args.judge, "human")
        self.assertEqual(args.max_completion_length, 512)

    def test_online_dpo_loss_returns_per_sample_reward_summary(self):
        loss, reward, tokens, metrics = online_dpo_trainer.online_dpo_loss(
            **_preference_inputs(), loss_type="sigmoid"
        )
        self.assertTrue(mx.isfinite(loss).item())
        self.assertEqual(reward.shape, (2,))
        self.assertEqual(_scalar(tokens), 10.0)
        self.assertGreaterEqual(_scalar(metrics["accuracies"]), 0.0)

    def test_online_dpo_rejects_unknown_loss(self):
        with self.assertRaisesRegex(ValueError, "Unknown loss type"):
            online_dpo_trainer.online_dpo_loss(**_preference_inputs(), loss_type="bad")

    def test_online_dpo_batch_iterator_preserves_prompt_pairs(self):
        data = [
            {"prompt": [1], "prompt_text": "one"},
            {"prompt": [2, 3], "prompt_text": "two"},
        ]
        prompts, texts = next(online_dpo_trainer.iterate_online_dpo_batches(data, 2, 8))
        self.assertEqual(prompts, [[1], [2, 3]])
        self.assertEqual(texts, ["one", "two"])


class ORPOTrainerTest(unittest.TestCase):
    def test_orpo_defaults(self):
        args = orpo_trainer.ORPOTrainingArgs()
        self.assertEqual(args.beta, 0.1)
        self.assertEqual(args.reward_scaling, 1.0)

    def test_orpo_loss_is_finite_with_nonfinite_inputs(self):
        loss, reward, tokens, metrics = orpo_trainer.orpo_loss(
            chosen_logps=mx.array([float("nan"), -1.0]),
            chosen_logits_mean=mx.array(0.5),
            rejected_logps=mx.array([float("-inf"), -2.0]),
            rejected_logits_mean=mx.array(-0.5),
            chosen_masks=mx.ones((2, 2)),
            rejected_masks=mx.ones((2, 3)),
            preference_scores=mx.array([1.0, 0.5]),
            beta=0.2,
        )
        self.assertTrue(mx.isfinite(loss).item())
        self.assertTrue(mx.all(mx.isfinite(reward)).item())
        self.assertEqual(_scalar(tokens), 10.0)
        self.assertIn("rejected_logits_mean", metrics)

    def test_orpo_batch_iterator_includes_preference_scores(self):
        data = [
            {"chosen": [1, 2], "rejected": [3], "preference_score": 0.25},
            {"chosen": [4], "rejected": [5, 6], "preference_score": 0.75},
        ]
        batch = next(orpo_trainer.iterate_orpo_batches(data, 2, 8))
        self.assertEqual(batch[0].shape, (2, 8))
        self.assertTrue(mx.allclose(batch[4], mx.array([0.75, 0.25])).item())


class PPOTrainerTest(unittest.TestCase):
    def test_ppo_defaults(self):
        self.assertEqual(ppo_trainer.PPOTrainingArgs().epsilon, 0.2)

    def test_ppo_loss_reports_clipping_and_kl_metrics(self):
        inputs = _preference_inputs()
        inputs.pop("delta")
        loss, reward, tokens, metrics = ppo_trainer.ppo_loss(**inputs, epsilon=0.2)
        self.assertTrue(mx.isfinite(loss).item())
        self.assertEqual(reward.shape, (2,))
        self.assertEqual(_scalar(tokens), 10.0)
        self.assertIn("clip_fraction", metrics)
        self.assertIn("kl_penalty", metrics)
        self.assertAlmostEqual(_scalar(metrics["advantages_mean"]), 0.0, places=5)


class XPOTrainerTest(unittest.TestCase):
    def test_alpha_schedule_selects_current_segment(self):
        schedule = [0.1, 0.2, 0.3]
        self.assertEqual(xpo_trainer.get_current_alpha(0, 9, schedule), 0.1)
        self.assertEqual(xpo_trainer.get_current_alpha(4, 9, schedule), 0.2)
        self.assertEqual(xpo_trainer.get_current_alpha(99, 9, schedule), 0.3)

    def test_single_alpha_is_constant(self):
        self.assertEqual(xpo_trainer.get_current_alpha(50, 100, [0.4]), 0.4)

    def test_xpo_exploration_bonus_changes_loss_and_metrics(self):
        inputs = _preference_inputs()
        base_loss, _, _, base_metrics = xpo_trainer.xpo_loss(
            **inputs, loss_type="sigmoid", alpha=0.0
        )
        explored_loss, _, _, explored_metrics = xpo_trainer.xpo_loss(
            **inputs, loss_type="sigmoid", alpha=0.5
        )
        self.assertNotEqual(_scalar(base_loss), _scalar(explored_loss))
        self.assertEqual(base_metrics["exploration_bonus"], 0)
        self.assertNotEqual(_scalar(explored_metrics["exploration_bonus"]), 0.0)

    def test_xpo_rejects_unknown_loss(self):
        with self.assertRaisesRegex(ValueError, "Unknown loss type"):
            xpo_trainer.xpo_loss(**_preference_inputs(), loss_type="bad")


class RLHFReinforceTrainerTest(unittest.TestCase):
    def test_compute_kl_penalty_is_zero_for_identical_logits(self):
        logits = mx.array([[[1.0, 0.0], [0.5, -0.5]]])
        penalty = rlhf_reinforce_trainer.compute_kl_penalty(
            logits, logits, mx.ones((1, 2))
        )
        self.assertTrue(mx.allclose(penalty, mx.zeros((1,))).item())

    def test_reinforce_loss_counts_tokens_and_reports_metrics(self):
        policy = mx.array([[[2.0, 0.0], [0.5, 1.5]]])
        reference = mx.array([[[1.5, 0.5], [1.0, 1.0]]])
        loss, tokens, metrics = rlhf_reinforce_trainer.rlhf_reinforce_loss(
            policy, reference, mx.array([1.0]), mx.ones((1, 2)), beta=0.1
        )
        self.assertTrue(mx.isfinite(loss).item())
        self.assertEqual(_scalar(tokens), 2.0)
        self.assertEqual(
            set(metrics),
            {"rewards", "kl_penalty", "advantages", "policy_logps", "ref_logps"},
        )

    def test_get_model_logits_shifts_inputs_and_masks(self):
        class Model:
            def __call__(self, inputs):
                return inputs + 10

        tokens = mx.array([[1, 2, 3]])
        logits, targets, masks = rlhf_reinforce_trainer.get_model_logits(
            Model(), tokens, mx.array([[1, 1, 0]])
        )
        self.assertTrue(mx.array_equal(logits, mx.array([[11, 12]])).item())
        self.assertTrue(mx.array_equal(targets, mx.array([[2, 3]])).item())
        self.assertTrue(mx.array_equal(masks, mx.array([[1, 0]])).item())


class GRPOTrainerTest(unittest.TestCase):
    def test_sequence_importance_weight_averages_only_valid_tokens(self):
        ratios = mx.array([[1.0, 3.0, 100.0], [2.0, 4.0, 6.0]])
        mask = mx.array([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
        result = grpo_trainer.compute_log_importance_weights(ratios, mask, "sequence")
        self.assertTrue(mx.allclose(result, mx.array([[2.0], [4.0]])).item())

    def test_reward_weights_must_match_reward_functions(self):
        def reward(prompts, completions, answer, types=None):
            return [1.0] * len(completions)

        with self.assertRaisesRegex(ValueError, "reward weights"):
            grpo_trainer.calculate_rewards_and_advantages(
                [reward], ["p"], ["c"], ["a"], [None], [0], [0], [1.0, 2.0]
            )

    def test_all_missing_rewards_are_rejected(self):
        def missing(prompts, completions, answer, types=None):
            return [None] * len(completions)

        with self.assertRaisesRegex(RuntimeError, "All reward functions returned None"):
            grpo_trainer.calculate_rewards_and_advantages(
                [missing], ["p"], ["c"], ["a"], [None], [0], [0]
            )

    def test_grouped_rewards_produce_centered_advantages(self):
        def reward(prompts, completions, answer, types=None):
            return [1.0, 3.0]

        advantages, metrics = grpo_trainer.calculate_rewards_and_advantages(
            [reward],
            ["p", "p"],
            ["low", "high"],
            ["a", "a"],
            [None, None],
            [0, 0],
            [0],
        )
        self.assertAlmostEqual(_scalar(mx.mean(advantages)), 0.0, places=5)
        self.assertGreater(_scalar(advantages[1]), _scalar(advantages[0]))
        self.assertEqual(_scalar(metrics["total_rewards_mean"]), 2.0)

    def test_single_completion_has_zero_advantage(self):
        def reward(prompts, completions, answer, types=None):
            return [2.0]

        advantages, _ = grpo_trainer.calculate_rewards_and_advantages(
            [reward], ["p"], ["c"], ["a"], [None], [0], [0]
        )
        self.assertEqual(_scalar(advantages[0]), 0.0)


if __name__ == "__main__":
    unittest.main()
