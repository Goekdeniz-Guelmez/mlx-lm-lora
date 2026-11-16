import math
import unittest

import mlx.core as mx
import mlx.nn as nn

from mlx_lm_lora.trainer import cpo_trainer, dpo_trainer, orpo_trainer
from mlx_lm_lora.trainer.long_context import iter_cached_sft_chunks


class UniformModel:
    def eval(self):
        pass

    def __call__(self, inputs, cache=None):
        return mx.zeros((*inputs.shape, 4))


class PreferenceMaskingTest(unittest.TestCase):
    def test_response_scores_include_first_and_last_tokens_only(self):
        tokens = mx.array([[0, 1, 2, 3, 0, 0]])
        mask = mx.array([[0, 0, 1, 1, 0, 0]])
        logits = mx.array([[[0.0, 1.0, 2.0, 3.0]] * 5])

        class FixedModel:
            def __call__(self, inputs, cache=None):
                return logits

        expected = -nn.losses.cross_entropy(logits, tokens[:, 1:])
        for trainer in (dpo_trainer, cpo_trainer):
            with self.subTest(trainer=trainer.__name__):
                scores = trainer.get_token_scores(FixedModel(), tokens, mask)
                self.assertTrue(
                    mx.allclose(scores, expected * mx.array([[0, 1, 1, 0, 0]])).item()
                )
                self.assertAlmostEqual(
                    trainer.compute_score(scores, mask, "ipo").item(),
                    expected[:, 1:3].mean().item(),
                )
        logps, _ = orpo_trainer.get_logps(FixedModel(), tokens, mask)
        self.assertAlmostEqual(logps.item(), expected[:, 1:3].mean().item())

    def test_iterators_mask_prompt_padding_and_truncation(self):
        data = [
            {
                "chosen": [0, 1, 2, 3],
                "rejected": [0, 1, 2],
                "chosen_prompt_length": 2,
                "rejected_prompt_length": 2,
            },
            {
                "chosen": [0, 1, 2, 3],
                "rejected": [0, 1, 2, 3],
                "chosen_prompt_length": 5,
                "rejected_prompt_length": 5,
            },
        ]
        for iterator in (
            dpo_trainer.iterate_dpo_batches,
            cpo_trainer.iterate_cpo_batches,
            orpo_trainer.iterate_orpo_batches,
        ):
            with self.subTest(iterator=iterator.__name__):
                batch = next(iterator(data, 2, 4))
                self.assertEqual(batch[2][:, 4:].sum().item(), 0)
                self.assertEqual(batch[3][:, 4:].sum().item(), 0)
                self.assertEqual(batch[2][:, :4].tolist(), [[0, 0, 1, 1], [0, 0, 0, 0]])
                self.assertEqual(batch[3][:, :4].tolist(), [[0, 0, 1, 0], [0, 0, 0, 0]])

    def test_chunked_scores_match_full_scores_at_response_boundary(self):
        tokens = mx.array([[0, 1, 2, 3, 0, 1, 2]])
        mask = mx.array([[0, 0, 0, 1, 1, 1, 1]])
        for trainer in (dpo_trainer, cpo_trainer):
            with self.subTest(trainer=trainer.__name__):
                full = trainer.get_token_scores(UniformModel(), tokens, mask).sum()
                chunks = sum(
                    trainer.get_token_scores(
                        UniformModel(), tokens[:, start:end], mask[:, start:end]
                    ).sum()
                    for start, end in iter_cached_sft_chunks(tokens.shape[1], 3)
                )
                self.assertAlmostEqual(full.item(), chunks.item(), places=6)
                self.assertAlmostEqual(full.item(), -4 * math.log(4), places=6)

    def test_validation_rewards_use_same_weighting_as_margin(self):
        data = [
            {
                "chosen": [0, 1, 2, 3],
                "rejected": [0, 1, 2],
                "chosen_prompt_length": 2,
                "rejected_prompt_length": 2,
            }
        ]
        kwargs = dict(
            model=UniformModel(),
            dataset=data,
            batch_size=1,
            num_batches=1,
            beta=0.1,
            delta=2.0,
            max_seq_length=4,
            loss_type="sigmoid",
        )
        for evaluate in (dpo_trainer.evaluate_dpo, cpo_trainer.evaluate_cpo):
            with self.subTest(evaluate=evaluate.__name__):
                extra = (
                    {"ref_model": None} if evaluate == dpo_trainer.evaluate_dpo else {}
                )
                _, rewards, tokens, metrics = evaluate(**kwargs, **extra)
                self.assertEqual(tokens.item(), 3)
                self.assertAlmostEqual(rewards[0], -0.2 * math.log(4), places=6)
                self.assertAlmostEqual(rewards[1], -0.1 * math.log(4), places=6)
                self.assertAlmostEqual(
                    rewards[0] - rewards[1], metrics["margins"], places=6
                )
