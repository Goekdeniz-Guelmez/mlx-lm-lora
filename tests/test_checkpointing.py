import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from mlx_lm_lora.trainer.checkpointing import (
    load_training_checkpoint,
    save_training_checkpoint,
)


class CheckpointingTest(unittest.TestCase):
    def test_full_training_state_round_trip(self):
        model = nn.Linear(2, 1)
        optimizer = optim.Adam(learning_rate=1e-3)
        gradients = {
            "weight": mx.ones_like(model.weight),
            "bias": mx.ones_like(model.bias),
        }
        optimizer.update(model, gradients)
        mx.eval(model.parameters(), optimizer.state)
        saved_weights = dict(tree_flatten(model.trainable_parameters()))

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint"
            save_training_checkpoint(
                checkpoint,
                model=model,
                optimizer=optimizer,
                iteration=7,
                optimizer_step=1,
                grad_accum=gradients,
                trained_tokens=123,
            )

            model.update({"weight": mx.zeros_like(model.weight)})
            restored = load_training_checkpoint(
                checkpoint, model=model, optimizer=optimizer
            )

        self.assertEqual(restored["iteration"], 7)
        self.assertEqual(restored["optimizer_step"], 1)
        self.assertEqual(restored["trained_tokens"], 123)
        self.assertEqual(optimizer.step.item(), 1)
        self.assertIsNotNone(restored["grad_accum"])
        for name, value in tree_flatten(model.trainable_parameters()):
            self.assertTrue(mx.array_equal(value, saved_weights[name]).item())
