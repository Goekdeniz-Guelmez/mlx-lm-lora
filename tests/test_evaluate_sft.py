import unittest

import mlx.core as mx

from mlx_lm_lora.trainer.sft_trainer import evaluate_sft


class _FakeModel:
    """Minimal stand-in for a loaded model.

    ``evaluate_sft`` only calls ``eval()`` on the model and asks
    ``model_uses_recurrence`` whether it exposes named modules, so a plain
    object without ``named_modules`` is enough to exercise the helper.
    """

    def eval(self):
        return self


class EvaluateSftTest(unittest.TestCase):
    def test_returns_loss_and_token_count(self):
        """evaluate_sft must return ``(loss, n_tokens)``.

        ``mlx_lm_lora.train`` unpacks both values when it reports the test
        split, so returning a bare float raises
        ``TypeError: cannot unpack non-iterable float object``.
        """

        def fake_iterate_batches(dataset, batch_size, max_seq_length):
            yield (mx.zeros((1, 4), dtype=mx.int32), mx.zeros((1, 4), dtype=mx.int32))
            yield (mx.zeros((1, 2), dtype=mx.int32), mx.zeros((1, 2), dtype=mx.int32))

        def fake_loss(model, x, y):
            if x.shape[1] == 4:
                return mx.array(2.5), mx.array(4)
            return mx.array(1.5), mx.array(2)

        loss, ntok = evaluate_sft(
            model=_FakeModel(),
            dataset=[],
            batch_size=1,
            num_batches=2,
            loss=fake_loss,
            iterate_batches=fake_iterate_batches,
        )

        # Token weighted mean of the two fake batches: (2.5 * 4 + 1.5 * 2) / 6.
        self.assertAlmostEqual(loss, 13.0 / 6.0, places=5)
        self.assertEqual(ntok, 6)

    def test_token_count_is_an_int(self):
        def fake_iterate_batches(dataset, batch_size, max_seq_length):
            yield (mx.zeros((1, 3), dtype=mx.int32), mx.zeros((1, 3), dtype=mx.int32))

        def fake_loss(model, x, y):
            return mx.array(1.0), mx.array(3)

        _, ntok = evaluate_sft(
            model=_FakeModel(),
            dataset=[],
            batch_size=1,
            num_batches=1,
            loss=fake_loss,
            iterate_batches=fake_iterate_batches,
        )

        self.assertIsInstance(ntok, int)
        self.assertEqual(ntok, 3)


if __name__ == "__main__":
    unittest.main()
