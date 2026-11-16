import json
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm.models import llama

from mlx_lm_lora.trainer import dsla_trainer, sft_trainer
from mlx_lm_lora.trainer.checkpointing import (
    FINAL_CHECKPOINT_NAME,
    TrainingProgress,
    checkpoint_dir,
    load_training_state,
    resume_training,
    save_training_state,
)


def _tiny_model():
    return llama.Model(
        llama.ModelArgs(
            model_type="llama",
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            rms_norm_eps=1e-6,
            vocab_size=16,
        )
    )


def _seed(value=0):
    mx.random.seed(value)
    np.random.seed(value)


def _weights(model):
    return dict(tree_flatten(model.trainable_parameters()))


class CheckpointStateTest(unittest.TestCase):
    def test_round_trip_restores_weights_optimizer_rng_and_counters(self):
        model = nn.Linear(2, 1)
        optimizer = optim.Adam(learning_rate=1e-3)
        gradients = {
            "weight": mx.ones_like(model.weight),
            "bias": mx.ones_like(model.bias),
        }
        optimizer.update(model, gradients)
        mx.eval(model.parameters(), optimizer.state)
        saved_weights = _weights(model)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint"
            _seed(3)
            save_training_state(
                path,
                model=model,
                optimizer=optimizer,
                iteration=7,
                optimizer_step=1,
                grad_accum=gradients,
                trained_tokens=123,
            )
            expected_mx, expected_np = mx.random.uniform(), np.random.rand()

            model.update({"weight": mx.zeros_like(model.weight)})
            fresh_optimizer = optim.Adam(learning_rate=1e-3)
            _seed(99)
            progress = load_training_state(path, model=model, optimizer=fresh_optimizer)

            self.assertEqual(mx.random.uniform().item(), expected_mx.item())
            self.assertEqual(np.random.rand(), expected_np)

        self.assertEqual(progress.iteration, 7)
        self.assertEqual(progress.start_iteration, 8)
        self.assertEqual(progress.optimizer_step, 1)
        self.assertEqual(progress.trained_tokens, 123)
        self.assertEqual(fresh_optimizer.step.item(), 1)
        self.assertTrue(mx.array_equal(progress.grad_accum["bias"], gradients["bias"]))
        for name, value in _weights(model).items():
            self.assertTrue(mx.array_equal(value, saved_weights[name]).item())

    def test_incomplete_or_foreign_checkpoints_are_rejected(self):
        model, optimizer = nn.Linear(2, 1), optim.SGD(learning_rate=0.1)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                load_training_state(directory, model=model, optimizer=optimizer)
            save_training_state(
                directory, model=model, optimizer=optimizer, iteration=1
            )
            state_file = Path(directory) / "state.json"
            state = json.loads(state_file.read_text())
            state_file.write_text(json.dumps({**state, "version": 999}))
            with self.assertRaises(ValueError):
                load_training_state(directory, model=model, optimizer=optimizer)

    def test_resume_is_a_no_op_without_a_checkpoint(self):
        args = sft_trainer.SFTTrainingArgs()
        progress = resume_training(args, nn.Linear(2, 1), optim.SGD(0.1))
        self.assertEqual(progress, TrainingProgress())
        self.assertEqual(progress.start_iteration, 1)


class ResumeMatchesUninterruptedTrainingTest(unittest.TestCase):
    """Stopping at a checkpoint and resuming must give bit-identical weights."""

    def _assert_resume_matches(self, make_args, run):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("full", "resumed"):
                (root / name).mkdir()

            _seed()
            full = _tiny_model()
            run(full, make_args(root / "full", iters=6))
            self.assertTrue((root / "full" / FINAL_CHECKPOINT_NAME).exists())
            # Iteration 3 is mid-accumulation, as if the run had crashed there.
            checkpoint = checkpoint_dir(root / "full" / "adapters.safetensors", 3)
            self.assertTrue((checkpoint / "grad_accum.safetensors").exists())

            _seed(42)  # Must not matter: RNG state comes from the checkpoint.
            resumed = _tiny_model()
            run(
                resumed,
                make_args(root / "resumed", iters=6, resume_checkpoint=checkpoint),
            )

        expected = _weights(full)
        for name, value in _weights(resumed).items():
            self.assertTrue(mx.array_equal(value, expected[name]).item(), name)

    def test_sft_with_adam_and_partial_gradient_accumulation(self):
        dataset = [[1, 2, 3, 4, 5], [6, 7, 8], [9, 10, 11, 12], [2, 4, 6, 8, 10]]

        def make_args(directory, **overrides):
            return sft_trainer.SFTTrainingArgs(
                batch_size=2,
                max_seq_length=8,
                gradient_accumulation_steps=2,
                steps_per_report=100,
                steps_per_eval=100,
                steps_per_save=3,
                adapter_file=str(directory / "adapters.safetensors"),
                **overrides,
            )

        def run(model, args):
            sft_trainer.train_sft(model, optim.AdamW(1e-2), dataset, None, args)

        self._assert_resume_matches(make_args, run)

    def test_dsla_restores_its_accumulation_counter(self):
        dataset = [
            {
                "chosen": [1, 2, 3, 4],
                "rejected": [1, 2, 5],
                "chosen_prompt_length": 2,
                "rejected_prompt_length": 2,
            },
            {
                "chosen": [6, 7, 8],
                "rejected": [6, 9, 10, 11],
                "chosen_prompt_length": 1,
                "rejected_prompt_length": 1,
            },
        ]

        def make_args(directory, **overrides):
            return dsla_trainer.DSLATrainingArgs(
                loss_type="orpo",
                batch_size=2,
                max_seq_length=4,
                gradient_accumulation_steps=2,
                steps_per_report=100,
                steps_per_eval=None,
                steps_per_save=3,
                adapter_file=str(directory / "adapters.safetensors"),
                **overrides,
            )

        def run(model, args):
            dsla_trainer.train_dsla(model, optim.Adam(1e-2), dataset, None, args)

        self._assert_resume_matches(make_args, run)


if __name__ == "__main__":
    unittest.main()
