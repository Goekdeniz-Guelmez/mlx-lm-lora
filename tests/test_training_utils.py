import tempfile
import unittest
from pathlib import Path
from unittest import mock

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_lm_lora.trainer.training_utils import save_adapters


class SaveAdaptersTest(unittest.TestCase):
    def test_periodic_save_contains_only_trainable_weights_in_both_files(self):
        model = nn.Sequential(nn.Linear(3, 2), nn.Linear(2, 1))
        model.freeze()
        model.layers[1].unfreeze()
        expected = dict(tree_flatten(model.trainable_parameters()))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter.safetensors"
            with mock.patch.object(
                model, "trainable_parameters", wraps=model.trainable_parameters
            ) as parameters:
                save_adapters(model, path, 42, report=False)
            parameters.assert_called_once_with()
            for saved in (path, path.parent / "0000042_adapters.safetensors"):
                weights = mx.load(str(saved))
                self.assertEqual(weights.keys(), expected.keys())
                for key, value in weights.items():
                    self.assertTrue(mx.array_equal(value, expected[key]).item())

    def test_final_save_does_not_create_numbered_checkpoint(self):
        model = nn.Linear(2, 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter.safetensors"
            save_adapters(model, str(path), report=False)
            self.assertEqual(list(path.parent.iterdir()), [path])


if __name__ == "__main__":
    unittest.main()
