import tempfile
import unittest
from pathlib import Path
from unittest import mock

from mlx_lm_lora import utils


class UtilsTest(unittest.TestCase):
    def test_full_finetuning_uses_model_artifacts_not_adapter_artifacts(self):
        class Model:
            class Args:
                pass

            args = Args()

        model = Model()
        tokenizer = mock.Mock()

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with (
                mock.patch.object(utils, "load", return_value=(model, tokenizer)),
                mock.patch.object(utils, "save_config") as save_config,
            ):
                _, _, weights_file = utils.from_pretrained(
                    "base-model", new_adapter_path=directory, lora_config=None
                )

            self.assertEqual(weights_file, output / "model.safetensors")
            save_config.assert_called_once_with({}, output / "config.json")
            tokenizer.save_pretrained.assert_called_once_with(output)
            self.assertFalse((output / "adapter_config.json").exists())

    def test_calculate_iters_rounds_partial_batches_up(self):
        with mock.patch("builtins.print"):
            self.assertEqual(utils.calculate_iters(range(10), 4, 3), 9)

    def test_calculate_iters_handles_empty_dataset(self):
        with mock.patch("builtins.print"):
            self.assertEqual(utils.calculate_iters([], 4, 3), 0)

    def test_find_lmstudio_models_path_returns_existing_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            expected = home / ".lmstudio" / "models"
            expected.mkdir(parents=True)
            with mock.patch.object(Path, "home", return_value=home):
                self.assertEqual(utils.find_lmstudio_models_path(), expected)

    def test_find_lmstudio_models_path_raises_for_missing_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            with mock.patch.object(Path, "home", return_value=home):
                with self.assertRaisesRegex(FileNotFoundError, str(home)):
                    utils.find_lmstudio_models_path()


if __name__ == "__main__":
    unittest.main()
