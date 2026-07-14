import types
import unittest
from unittest import mock

from mlx_lm_lora.trainer import datasets


class FakeTokenizer:
    eos_token_id = 99

    def __init__(self):
        self.calls = []

    def encode(self, text):
        self.calls.append(("encode", text))
        return [ord(char) for char in text]

    def apply_chat_template(
        self, messages, add_generation_prompt=False, tokenize=True, tools=None
    ):
        self.calls.append(
            ("template", messages, add_generation_prompt, tokenize, tools)
        )
        rendered = "|".join(
            f"{message['role']}:{message['content']}" for message in messages
        )
        if add_generation_prompt:
            rendered += "|assistant:"
        return [len(rendered), len(messages)] if tokenize else rendered


class DatasetAdaptersTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeTokenizer()

    def test_grpo_dataset_uses_chat_template_and_preserves_metadata(self):
        dataset = datasets.GRPODataset(
            [{"prompt": 12, "answer": 4, "system": "sys", "type": "math"}],
            self.tokenizer,
        )
        prompt_tokens, answer_tokens, prompt, answer, kind = dataset[0]
        self.assertEqual(prompt_tokens, [29, 2])
        self.assertEqual(answer_tokens, [52])
        self.assertEqual((prompt, answer, kind), ("12", "4", "math"))

    def test_grpo_text_completion_bypasses_chat_template(self):
        dataset = datasets.GRPODataset(
            [{"prompt": "p", "answer": "a", "text": "raw"}],
            self.tokenizer,
            text_completion_key="text",
        )
        self.assertEqual(dataset[0][0], [114, 97, 119])
        self.assertFalse(any(call[0] == "template" for call in self.tokenizer.calls))

    def test_preference_dataset_encodes_rejected_value(self):
        dataset = datasets.PreferenceDataset(
            [{"chosen": "good", "rejected": "bad"}], self.tokenizer
        )
        self.assertEqual(dataset[0]["chosen"], [103, 111, 111, 100])
        self.assertEqual(dataset[0]["rejected"], [98, 97, 100])

    def test_prompt_dataset_accepts_message_list(self):
        item = {"prompt": [{"role": "user", "content": "hello"}]}
        result = datasets.PromptDataset([item], self.tokenizer).process(item)
        self.assertEqual(result["prompt"], [21, 1])
        self.assertEqual(result["prompt_text"], "user:hello|assistant:")

    def test_prompt_dataset_wraps_scalar_as_user_message(self):
        item = {"prompt": 123}
        datasets.PromptDataset([item], self.tokenizer).process(item)
        template_call = self.tokenizer.calls[0]
        self.assertEqual(template_call[1], [{"role": "user", "content": "123"}])

    def test_dpo_dataset_builds_distinct_assistant_responses(self):
        dataset = datasets.DPODataset(
            [{"system": "s", "prompt": "p", "chosen": "yes", "rejected": "no"}],
            self.tokenizer,
        )
        self.assertNotEqual(dataset[0]["chosen"], dataset[0]["rejected"])
        calls = [call for call in self.tokenizer.calls if call[0] == "template"]
        self.assertEqual(calls[0][1][-1]["content"], "yes")
        self.assertEqual(calls[1][1][-1]["content"], "no")

    def test_ftpo_dataset_keeps_only_single_token_alternatives(self):
        dataset = datasets.FTPODataset(
            [
                {
                    "context_with_chat_template": "ctx",
                    "rejected_decoded": "x",
                    "multi_chosen_decoded": ["y", "zz", "x"],
                }
            ],
            self.tokenizer,
        )
        self.assertEqual(
            dataset[0],
            {
                "prompt_ids": [99, 116, 120],
                "chosen_ids": [121],
                "rejected_token_id": 120,
            },
        )

    def test_orpo_extracts_supported_content_shapes(self):
        dataset = datasets.ORPODataset([], self.tokenizer)
        self.assertEqual(dataset._extract_content("text"), "text")
        self.assertEqual(dataset._extract_content({"content": "dict"}), "dict")
        self.assertEqual(
            dataset._extract_content({"messages": [{"content": "nested"}]}),
            "nested",
        )
        self.assertEqual(dataset._extract_content([{"content": "list"}]), "list")
        self.assertEqual(dataset._extract_content(7), "")

    def test_orpo_defaults_preference_score(self):
        dataset = datasets.ORPODataset(
            [{"prompt": "p", "chosen": "yes", "rejected": "no"}], self.tokenizer
        )
        self.assertEqual(dataset[0]["preference_score"], 1.0)

    def test_orpo_converts_explicit_preference_score_to_float(self):
        dataset = datasets.ORPODataset(
            [
                {
                    "prompt": "p",
                    "chosen": "yes",
                    "rejected": "no",
                    "preference_score": "0.75",
                }
            ],
            self.tokenizer,
        )
        self.assertEqual(dataset[0]["preference_score"], 0.75)

    def test_text_dataset_appends_eos(self):
        dataset = datasets.TextDataset([{"text": "a"}], self.tokenizer)
        self.assertEqual(dataset.process(dataset[0]), [97, 99])

    def test_text_dataset_handles_empty_text(self):
        dataset = datasets.TextDataset([{"text": ""}], self.tokenizer)
        self.assertEqual(dataset.process(dataset[0]), [99])

    def test_text_dataset_does_not_duplicate_eos(self):
        self.tokenizer.encode = mock.Mock(return_value=[1, 99])
        dataset = datasets.TextDataset([{"text": "x"}], self.tokenizer)
        self.assertEqual(dataset.process(dataset[0]), [1, 99])

    def test_chat_dataset_reports_prompt_offset_when_masking(self):
        item = {
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"},
            ]
        }
        tokens, offset = datasets.ChatDataset(
            [item], self.tokenizer, mask_prompt=True
        ).process(item)
        self.assertGreater(len(tokens), offset)

    def test_completions_dataset_reports_prompt_offset_when_masking(self):
        item = {"question": "q", "response": "a"}
        tokens, offset = datasets.CompletionsDataset(
            [item], self.tokenizer, "question", "response", True
        ).process(item)
        self.assertGreater(len(tokens), offset)

    def test_cache_dataset_processes_each_item_once(self):
        wrapped = mock.MagicMock()
        wrapped.__len__.return_value = 1
        wrapped.__getitem__.return_value = {"value": 1}
        wrapped.process.return_value = [1]
        cached = datasets.CacheDataset(wrapped)
        self.assertEqual(cached[0], [1])
        self.assertEqual(cached[0], [1])
        wrapped.process.assert_called_once_with({"value": 1})


class DatasetFactoryTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeTokenizer()

    def test_create_dataset_selects_sft_formats(self):
        cases = [
            ({"prompt": "p", "completion": "c"}, datasets.CompletionsDataset),
            ({"messages": []}, datasets.ChatDataset),
            ({"text": "t"}, datasets.TextDataset),
        ]
        for sample, expected_type in cases:
            with self.subTest(expected_type=expected_type.__name__):
                result = datasets.create_dataset(
                    [sample], self.tokenizer, types.SimpleNamespace(train_mode="sft")
                )
                self.assertIsInstance(result, expected_type)

    def test_create_dataset_rejects_prompt_masking_for_plain_text(self):
        config = types.SimpleNamespace(train_mode="sft", mask_prompt=True)
        with self.assertRaisesRegex(ValueError, "Prompt masking"):
            datasets.create_dataset([{"text": "x"}], self.tokenizer, config)

    def test_create_dataset_rejects_unsupported_dpo_shape(self):
        config = types.SimpleNamespace(train_mode="dpo")
        with self.assertRaisesRegex(ValueError, "Unsupported data format"):
            datasets.create_dataset([{"prompt": "x"}], self.tokenizer, config)


if __name__ == "__main__":
    unittest.main()
