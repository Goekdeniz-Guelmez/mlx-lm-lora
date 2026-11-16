import unittest

from mlx_lm_lora.trainer import grpo_reward_functions as rewards


class RewardRegistryTest(unittest.TestCase):
    def test_built_in_rewards_are_registered(self):
        names = rewards.list_available_reward_functions()
        self.assertIn("r1_accuracy_reward_func", names)
        self.assertIn("r1_count_xml", names)

    def test_get_reward_function_returns_registered_callable(self):
        self.assertIs(
            rewards.get_reward_function("r1_int_reward_func"),
            rewards.r1_int_reward_func,
        )

    def test_get_unknown_reward_describes_available_functions(self):
        with self.assertRaisesRegex(KeyError, "missing.*Available functions"):
            rewards.get_reward_function("missing")

    def test_custom_name_registration(self):
        name = "test_custom_reward"

        @rewards.register_reward_function(name)
        def custom(prompts, completions, answer, types=None):
            return [1.0] * len(completions)

        self.assertIs(rewards.get_reward_function(name), custom)
        rewards.REWARD_REGISTRY.pop(name)

    def test_registered_rewards_accept_the_trainer_keyword_contract(self):
        """calculate_rewards_and_advantages passes every argument by keyword.

        A reward function whose third parameter is named anything other than
        `answer` therefore raises TypeError on the first training step rather
        than at registration, which is a long way from where the mistake was
        made. Pin the call shape here so a rename cannot pass the suite.
        """
        batch = {
            "prompts": ["<answer>1</answer>"],
            "completions": ["<reasoning>r</reasoning>\n<answer>1</answer>\n"],
            "answer": ["1"],
            "types": [None],
        }
        for name in rewards.list_available_reward_functions():
            with self.subTest(reward=name):
                scores = rewards.get_reward_function(name)(**batch)
                self.assertEqual(len(scores), len(batch["completions"]))

    def test_default_rewards_are_in_expected_order(self):
        self.assertEqual(
            rewards.get_default_reward_functions(),
            [
                rewards.r1_accuracy_reward_func,
                rewards.r1_int_reward_func,
                rewards.r1_strict_format_reward_func,
                rewards.r1_soft_format_reward_func,
                rewards.r1_count_xml,
            ],
        )


class R1RewardFunctionsTest(unittest.TestCase):
    def test_extract_xml_answer_strips_whitespace(self):
        self.assertEqual(rewards.r1_extract_xml_answer("x<answer> 42 </answer>y"), "42")

    def test_extract_xml_answer_without_tags_returns_text(self):
        self.assertEqual(rewards.r1_extract_xml_answer(" plain "), "plain")

    def test_integer_reward_accepts_only_nonempty_digits(self):
        completions = ["<answer>42</answer>", "<answer>-1</answer>", ""]
        self.assertEqual(
            rewards.r1_int_reward_func([], completions, []), [0.5, 0.0, 0.0]
        )

    def test_empty_completions_return_one_zero_per_prompt(self):
        self.assertEqual(rewards.r1_int_reward_func(["a", "b"], [], []), [0.0, 0.0])

    def test_accuracy_reward_matches_extracted_answers(self):
        completions = ["<answer>yes</answer>", "<answer>no</answer>", ""]
        self.assertEqual(
            rewards.r1_accuracy_reward_func([], completions, ["yes", "yes", ""]),
            [2.0, 0.0, 0.0],
        )

    def test_accuracy_reward_handles_missing_answers(self):
        self.assertEqual(
            rewards.r1_accuracy_reward_func(["a", "b"], ["x"], []), [0.0, 0.0]
        )

    def test_soft_format_accepts_ordered_nonempty_sections(self):
        completion = "<think>reason</think>\n<answer>42</answer>"
        self.assertEqual(
            rewards.r1_soft_format_reward_func([], [completion], []), [0.5]
        )

    def test_soft_format_rejects_empty_or_reordered_sections(self):
        completions = [
            "<think></think><answer>42</answer>",
            "<answer>42</answer><think>reason</think>",
            "",
        ]
        self.assertEqual(
            rewards.r1_soft_format_reward_func([], completions, []),
            [0.0, 0.0, 0.0],
        )

    def test_strict_format_requires_spaced_single_line_shape(self):
        completions = [
            "<think> reason </think> <answer> 42 </answer>",
            "<think>reason</think><answer>42</answer>",
        ]
        self.assertEqual(
            rewards.r1_strict_format_reward_func([], completions, []), [0.5, 0.0]
        )

    def test_count_xml_scores_all_four_markers(self):
        text = "<think>\nreason</think><answer>42</answer>"
        self.assertEqual(rewards.r1_count_xml([], [text], []), [0.5])

    def test_count_xml_penalizes_trailing_text_and_clamps_at_zero(self):
        complete = "<think>\nr</think><answer>a</answer>extra"
        malformed = "</answer>" + ("x" * 500)
        scores = rewards.r1_count_xml([], [complete, malformed], [])
        self.assertAlmostEqual(scores[0], 0.495)
        self.assertEqual(scores[1], 0.0)


if __name__ == "__main__":
    unittest.main()
