"""Numerical checks for the MLX KLPO objective."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import numpy as np

from mlx_lm_lora.trainer import klpo_trainer as klpo


class KLPONumericsTest(unittest.TestCase):
    def test_generation_discards_sampler_lookahead_record(self):
        class Model:
            training = True

            def eval(self):
                self.training = False

            def train(self, value=True):
                self.training = value

        class Tokenizer:
            eos_token_ids = frozenset()

            def encode(self, text, add_special_tokens=False):
                return [1]

            def decode(self, tokens):
                return str(tokens)

        class Generator:
            def __init__(self, model, **kwargs):
                self.emitted = False

            def insert(self, prompts, max_tokens, samplers=None):
                self.sampler = samplers[0]
                self.sampler(mx.zeros((1, 4)))
                self.sampler(mx.zeros((1, 4)))
                return [7]

            def next_generated(self):
                if self.emitted:
                    return []
                self.emitted = True
                return [SimpleNamespace(uid=7, token=2)]

            def close(self):
                pass

        def fake_recording_sampler(record, temperature):
            def sampler(logprobs):
                record.action_logps.append(mx.array(-1.0))
                return mx.array([2], dtype=mx.int32)

            return sampler

        with patch.object(klpo, "BatchGenerator", Generator), patch.object(
            klpo, "_recording_sampler", fake_recording_sampler
        ):
            completions, _, records, _ = klpo.generate_klpo(
                Model(), Tokenizer(), [[1]], 2, 1, None, 1.0, "binary", 1, 1
            )

        self.assertEqual(completions[0].shape, (1,))
        self.assertEqual(records[0]["action_logps"].shape, (1,))

    def test_binary_identical_sampler_has_zero_kl(self):
        current = mx.array([[-1.0, -2.0]])
        behavior = current
        rewards = mx.array([1.0])
        mask = mx.array([[True, True]])
        loss, local_kl, _, _ = klpo._sequence_loss(
            current,
            behavior,
            rewards,
            mask,
            estimator="binary",
            beta=0.1,
        )
        self.assertTrue(mx.allclose(local_kl, mx.zeros_like(local_kl)).item())
        self.assertTrue(mx.isfinite(loss).item())

    def test_binary_rounded_one_logps_remain_finite(self):
        current = mx.array([[0.0]])
        behavior = mx.array([[-0.1]])
        rewards = mx.array([1.0])
        mask = mx.array([[True]])

        for route, loss_fn in (("token", klpo._token_loss), ("sequence", klpo._sequence_loss)):
            with self.subTest(route=route):
                loss, local_kl, _, _ = loss_fn(
                    current,
                    behavior,
                    rewards,
                    mask,
                    estimator="binary",
                    beta=0.1,
                )
                self.assertTrue(mx.isfinite(loss).item())
                self.assertTrue(mx.all(mx.isfinite(local_kl)).item())

    def test_token_mc_keeps_auxiliary_policy_gradient(self):
        current = mx.array([[-1.0, -2.0]])
        behavior = mx.array([[-1.2, -1.8]])
        auxiliary = mx.array([[[-1.5], [-2.5]]])
        aux_behavior = mx.array([[[-1.4], [-2.4]]])
        rewards = mx.array([2.0])
        mask = mx.array([[True, True]])

        gradient = mx.grad(
            lambda value: klpo._token_loss(
                value,
                behavior,
                rewards,
                mask,
                estimator="mc",
                beta=0.1,
                aux=(auxiliary, aux_behavior),
            )[0].sum()
        )(current)
        self.assertTrue(mx.all(mx.isfinite(gradient)).item())
        self.assertGreater(float(np.abs(np.asarray(gradient)).sum()), 0.0)

    def test_sequence_mc_requires_leave_one_out_samples(self):
        with self.assertRaisesRegex(ValueError, "at least two"):
            klpo._sequence_loss(
                mx.array([[-1.0]]),
                mx.array([[-1.0]]),
                mx.array([1.0]),
                mx.array([[True]]),
                estimator="mc",
                beta=0.1,
                aux=(mx.array([[[-1.0]]]), mx.array([[[-1.0]]])),
            )

    def test_klpo_argument_validation(self):
        with self.assertRaisesRegex(ValueError, "Sequence MC-KL"):
            klpo._validate_klpo_args("sequence", "mc", 1, 2, 0.1, 1e-6)
        with self.assertRaisesRegex(ValueError, "positive"):
            klpo._validate_klpo_args("token", "mc", 1, 2, 0.0, 1e-6)

    def test_all_route_estimator_surrogates_are_finite(self):
        current = mx.array([[-1.0, -1.5], [-2.0, -1.2]])
        behavior = mx.array([[-1.1, -1.4], [-1.8, -1.3]])
        rewards = mx.array([1.0, 0.5])
        mask = mx.array([[True, True], [True, True]])
        aux = (
            mx.array([[[-1.2, -1.3], [-1.4, -1.6]], [[-1.9, -2.1], [-1.1, -1.4]]]),
            mx.array([[[-1.1, -1.2], [-1.5, -1.7]], [[-1.8, -2.0], [-1.2, -1.5]]]),
        )
        head = (
            mx.array(
                [
                    [[-1.0, -1.4], [-1.5, -1.8]],
                    [[-2.0, -2.3], [-1.2, -1.6]],
                ]
            ),
            mx.array(
                [
                    [[-1.1, -1.3], [-1.6, -1.7]],
                    [[-1.9, -2.2], [-1.3, -1.5]],
                ]
            ),
        )
        full = (
            mx.concatenate([head[0], mx.array([[[-2.0], [-2.2]], [[-2.5], [-1.9]]])], axis=-1),
            mx.concatenate([head[1], mx.array([[[-2.1], [-2.0]], [[-2.4], [-2.1]]])], axis=-1),
        )
        for route in ("token", "sequence"):
            for estimator in ("binary", "mc", "topk", "full"):
                with self.subTest(route=route, estimator=estimator):
                    kwargs = {"estimator": estimator, "beta": 0.1}
                    if estimator == "mc":
                        kwargs["aux"] = aux
                    elif estimator == "topk":
                        kwargs["head"] = head
                    elif estimator == "full":
                        kwargs["full"] = full
                    fn = klpo._token_loss if route == "token" else klpo._sequence_loss
                    loss, _, _, _ = fn(current, behavior, rewards, mask, **kwargs)
                    self.assertTrue(mx.isfinite(loss).item())


if __name__ == "__main__":
    unittest.main()
