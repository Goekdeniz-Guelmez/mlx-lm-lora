import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

import mlx.core as mx
import numpy as np

from mlx_lm_lora.trainer import (
    cpo_trainer,
    distributed,
    dpo_trainer,
    dsla_trainer,
    ftpo_trainer,
    grpo_trainer,
    online_dpo_trainer,
    orpo_trainer,
    sft_trainer,
    xpo_trainer,
)

REPO = Path(__file__).resolve().parents[1]


class FakeGroup:
    def __init__(self, rank, size):
        self._rank, self._size = rank, size

    def rank(self):
        return self._rank

    def size(self):
        return self._size


def as_rank(rank, size):
    return mock.patch.object(
        mx.distributed, "init", lambda *a, **k: FakeGroup(rank, size)
    )


def first_batch(iterate, dataset, batch_size, rank, size):
    np.random.seed(0)
    with as_rank(rank, size):
        return next(iterate(dataset, batch_size, 64, train=True))


def preference_rows(n):
    return [
        {
            "chosen": [1] + [i + 2] * (i + 2),
            "rejected": [1] + [i + 3] * (i + 1),
            "prompt": [1] * (i + 1),
            "prompt_text": f"p{i}",
            "chosen_prompt_length": 1,
            "rejected_prompt_length": 1,
            "prompt_length": 1,
        }
        for i in range(n)
    ]


def flat(x):
    return sorted(np.array(x).reshape(-1).tolist())


class ShardTest(unittest.TestCase):
    def test_shard_partitions_global_batch(self):
        batch = list(range(8))
        shards = [distributed.shard(batch, rank=r, size=4) for r in range(4)]
        self.assertEqual(sorted(sum(shards, [])), batch)
        self.assertEqual(shards[1], [1, 5])

    def test_shard_rejects_uneven_batch(self):
        with self.assertRaises(ValueError):
            distributed.shard(list(range(6)), rank=0, size=4)
        with self.assertRaises(ValueError):
            distributed.local_batch_size(3, 2)
        self.assertEqual(distributed.local_batch_size(8, 2), 4)

    def test_single_process_reductions(self):
        self.assertAlmostEqual(distributed.reduce_mean(mx.array(6.0), 3).item(), 2.0)
        self.assertEqual(
            distributed.reduce_metrics({"a": mx.array(4.0), "b": 2}, 2),
            {"a": 2.0, "b": 1.0},
        )
        x = mx.array([1.0, 2.0, 3.0, 6.0])
        mean, std = distributed.global_mean_std(x)
        self.assertAlmostEqual(mean.item(), 3.0)
        self.assertAlmostEqual(std.item(), float(np.sqrt(np.var([1, 2, 3, 6]))), 5)


class IteratorShardingTest(unittest.TestCase):
    """Two ranks must split the single-process batch, never duplicate it."""

    def check(self, iterate, dataset, batch_size, key):
        whole = key(first_batch(iterate, dataset, batch_size, 0, 1))
        parts = [key(first_batch(iterate, dataset, batch_size, r, 2)) for r in (0, 1)]
        self.assertNotEqual(parts[0], parts[1])
        self.assertEqual(sorted(parts[0] + parts[1]), sorted(whole))

    def test_sft(self):
        data = [[i + 1] * (i + 2) for i in range(8)]
        self.check(
            sft_trainer.iterate_batches, data, 4, lambda b: [int(r[0]) for r in b[0]]
        )

    def test_preference_iterators(self):
        rows = preference_rows(8)
        first = lambda b: [int(r[1]) for r in b[0]]
        for module, name in [
            (dpo_trainer, "iterate_dpo_batches"),
            (cpo_trainer, "iterate_cpo_batches"),
            (orpo_trainer, "iterate_orpo_batches"),
            (dsla_trainer, "iterate_dsla_batches"),
        ]:
            with self.subTest(name):
                self.check(getattr(module, name), rows, 4, first)

    def test_online_iterators(self):
        rows = preference_rows(8)
        for module in (online_dpo_trainer, xpo_trainer):
            with self.subTest(module.__name__):
                self.check(
                    module.iterate_online_dpo_batches, rows, 4, lambda b: list(b[1])
                )

    def test_grpo(self):
        rows = [([1] * (i + 1), [2], f"p{i}", "a") for i in range(8)]
        self.check(grpo_trainer.iterate_grpo_batches, rows, 4, lambda b: list(b[2]))

    def test_ftpo(self):
        rows = [
            {"prompt_ids": [i + 1] * (i + 1), "chosen_ids": [3], "rejected_token_id": 4}
            for i in range(8)
        ]
        self.check(
            ftpo_trainer.iterate_ftpo_batches,
            rows,
            4,
            lambda b: [int(r[0]) for r in b[0]],
        )


WORKER = textwrap.dedent(
    """
    import mlx.core as mx
    from mlx_lm_lora.trainer import distributed

    group = distributed.init_distributed("ring")
    rank, size = group.rank(), group.size()
    assert size == 2

    # reduce_mean: rank r summed (r + 1) over 2 steps.
    assert distributed.reduce_mean(mx.array(float(rank + 1)), 2).item() == 0.75

    # global_mean_std and its gradient match one process with the full batch.
    full = mx.array([1.0, 2.0, 4.0, 8.0])
    local = full[rank * 2 : rank * 2 + 2]

    def loss(x):
        mean, std = distributed.global_mean_std(x)
        return (((x - mean) / std) * x).sum()

    def single(x):
        mean, std = x.mean(), mx.sqrt(x.var() + 1e-8)
        return (((x - mean) / std) * x).sum()

    mean, std = distributed.global_mean_std(local)
    assert abs(mean.item() - 3.75) < 1e-5, mean
    grad = mx.grad(loss)(local)
    expected = mx.grad(single)(full)[rank * 2 : rank * 2 + 2]
    assert mx.allclose(grad, expected, atol=1e-5).item(), (grad, expected)
    print("ok", rank)
    """
)


@unittest.skipUnless(shutil.which("mlx.launch"), "mlx.launch is not installed")
class RingLaunchTest(unittest.TestCase):
    def test_two_process_reductions(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "worker.py"
            script.write_text(WORKER)
            env = dict(os.environ, PYTHONPATH=str(REPO))
            result = subprocess.run(
                ["mlx.launch", "-n", "2", "--backend", "ring", "--"]
                + [sys.executable, str(script)],
                capture_output=True,
                text=True,
                env=env,
                timeout=120,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("ok 0", result.stdout)
        self.assertIn("ok 1", result.stdout)

    def test_two_processes_match_one_process(self):
        """Same global batch: SGD weights after 3 steps agree for offline modes."""
        script = str(REPO / "tests" / "distributed_equivalence.py")
        env = dict(os.environ, PYTHONPATH=str(REPO))
        with tempfile.TemporaryDirectory() as tmp:
            single = subprocess.run(
                [sys.executable, script, f"{tmp}/one"],
                capture_output=True,
                text=True,
                env=env,
                timeout=600,
            )
            self.assertEqual(single.returncode, 0, single.stderr[-2000:])
            launched = subprocess.run(
                ["mlx.launch", "-n", "2", "--backend", "ring", "--"]
                + [sys.executable, script, f"{tmp}/two"],
                capture_output=True,
                text=True,
                env=env,
                timeout=600,
            )
            # mlx.launch exits 0 even when a rank fails, so check both finished.
            self.assertIn("done 0", launched.stdout, launched.stderr[-2000:])
            self.assertIn("done 1", launched.stdout, launched.stderr[-2000:])
            for mode in ("sft", "dpo", "cpo", "orpo", "dsla", "ftpo"):
                one = mx.load(f"{tmp}/one/{mode}/weights.safetensors")
                two = mx.load(f"{tmp}/two/{mode}/weights.safetensors")
                diff = max(mx.abs(one[k] - two[k]).max().item() for k in one)
                self.assertLess(diff, 1e-5, mode)


if __name__ == "__main__":
    unittest.main()
