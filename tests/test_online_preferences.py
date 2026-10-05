import unittest
from unittest import mock

from mlx_lm_lora.trainer import online_dpo_trainer as online_dpo


class JudgedPairTextsTest(unittest.TestCase):
    def test_orders_completion_pairs_and_preserves_prompt_text(self):
        prompts = ["prompt 0: ", "prompt 1: ", "prompt 2: "]
        completions = [["a", "b"], ["c", "d"], ["e", "f"]]
        with mock.patch.object(online_dpo, "LLMPairwiseJudge") as judge:
            judge.return_value.judge.return_value = [0, 1, -1]
            chosen, rejected = online_dpo._judged_pair_texts(
                prompts, completions, "model", "tokenizer", {"system_prompt": "rank"}
            )
            judge.assert_called_once_with(
                model="model", tokenizer="tokenizer", system_prompt="rank"
            )
            judge.return_value.judge.assert_called_once_with(
                prompts, completions=completions
            )
        self.assertEqual(chosen, ["prompt 0: a", "prompt 1: d", "prompt 2: f"])
        self.assertEqual(rejected, ["prompt 0: b", "prompt 1: c", "prompt 2: e"])

    def test_human_judge_uses_same_pair_ordering(self):
        with mock.patch.object(online_dpo, "HumanPairwiseJudge") as judge:
            judge.return_value.judge.return_value = [1]
            self.assertEqual(
                online_dpo._judged_pair_texts(["p"], [["a", "b"]], "human", None, None),
                (["pb"], ["pa"]),
            )
            judge.assert_called_once_with()

    def test_optional_judge_config_uses_default_system_prompt(self):
        with mock.patch.object(online_dpo, "LLMPairwiseJudge") as judge:
            judge.return_value.judge.return_value = []
            self.assertEqual(
                online_dpo._judged_pair_texts([], [], "model", "tokenizer", None),
                ([], []),
            )
            judge.assert_called_once_with(
                model="model", tokenizer="tokenizer", system_prompt=None
            )


if __name__ == "__main__":
    unittest.main()
