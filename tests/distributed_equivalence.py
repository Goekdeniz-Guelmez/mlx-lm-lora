"""Train offline modes for a few SGD steps and save the weights.

Run once with one process and once under ``mlx.launch -n 2``; with the same
global batch the saved weights must agree. Used by tests/test_distributed.py.

    python tests/distributed_equivalence.py OUT_DIR
    mlx.launch -n 2 --backend ring -- python tests/distributed_equivalence.py OUT_DIR
"""

import sys
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm.models import llama

from mlx_lm_lora.trainer import (
    cpo_trainer,
    dpo_trainer,
    dsla_trainer,
    ftpo_trainer,
    orpo_trainer,
    sft_trainer,
)

MODES = ("sft", "dpo", "cpo", "orpo", "dsla", "ftpo")


def tiny_model():
    mx.random.seed(0)
    return llama.Model(
        llama.ModelArgs(
            model_type="llama",
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            rms_norm_eps=1e-6,
            vocab_size=32,
        )
    )


def preference_rows():
    # Equal lengths, so per-rank token means equal the global token mean.
    return [
        {
            "chosen": [1, 2, i + 3, i + 4, 5, 6],
            "rejected": [1, 2, i + 5, 7, i + 8, 9],
            "chosen_prompt_length": 2,
            "rejected_prompt_length": 2,
            "prompt_length": 2,
        }
        for i in range(8)
    ]


def run(mode, out):
    model = tiny_model()
    ref = tiny_model().freeze()
    mx.eval(model.parameters(), ref.parameters())
    np.random.seed(0)
    mx.random.seed(0)
    common = dict(
        batch_size=4,
        iters=3,
        val_batches=1,
        steps_per_report=1,
        steps_per_eval=1000,
        steps_per_save=1000,
        max_seq_length=64,
        adapter_file=str(out / mode / "adapters.safetensors"),
    )
    (out / mode).mkdir(parents=True, exist_ok=True)
    opt = optim.SGD(learning_rate=0.5)
    rows = preference_rows()
    if mode == "sft":
        data = [[1, 2, i + 3, i + 4, 5, 6, 7] for i in range(8)]
        sft_trainer.train_sft(
            model, opt, data, args=sft_trainer.SFTTrainingArgs(**common)
        )
    elif mode == "dpo":
        args = dpo_trainer.DPOTrainingArgs(**common)
        dpo_trainer.train_dpo(model, ref, opt, rows, args=args)
    elif mode == "cpo":
        cpo_trainer.train_cpo(
            model, opt, rows, args=cpo_trainer.CPOTrainingArgs(**common)
        )
    elif mode == "orpo":
        args = orpo_trainer.ORPOTrainingArgs(**common)
        orpo_trainer.train_orpo(model, opt, rows, args=args)
    elif mode == "dsla":
        args = dsla_trainer.DSLATrainingArgs(**common)
        dsla_trainer.train_dsla(model, opt, rows, args=args, ref_model=ref)
    elif mode == "ftpo":
        data = [
            {"prompt_ids": [1, i + 2, 3], "chosen_ids": [4, 5], "rejected_token_id": 6}
            for i in range(8)
        ]
        args = ftpo_trainer.FTPOTrainingArgs(**common)
        ftpo_trainer.train_ftpo(model, ref, opt, data, args=args)
    weights = dict(tree_flatten(model.trainable_parameters()))
    if mx.distributed.init().rank() == 0:
        mx.save_safetensors(str(out / mode / "weights.safetensors"), weights)


if __name__ == "__main__":
    root = Path(sys.argv[1])
    for mode in sys.argv[2:] or MODES:
        run(mode, root)
    print("done", mx.distributed.init().rank(), flush=True)
