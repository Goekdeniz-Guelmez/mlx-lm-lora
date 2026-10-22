import unittest

from mlx_lm.tokenizer_utils import TokenizerWrapper
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from mlx_lm_lora.chat_template import DEFAULT_CHAT_TEMPLATE, ensure_chat_template
from mlx_lm_lora.trainer.datasets import ChatDataset, CompletionsDataset


def make_tokenizer():
    backend = Tokenizer(
        models.WordLevel(
            {
                "[UNK]": 0,
                "</s>": 1,
                "user": 2,
                "assistant": 3,
                ":": 4,
                "hello": 5,
                "hi": 6,
            },
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", eos_token="</s>"
    )


class ChatTemplateTest(unittest.TestCase):
    def test_installs_renderable_template_once(self):
        for wrap in (lambda tokenizer: tokenizer, TokenizerWrapper):
            with self.subTest(wrapper=wrap):
                tokenizer = wrap(make_tokenizer())
                self.assertTrue(ensure_chat_template(tokenizer))
                self.assertEqual(tokenizer.chat_template, DEFAULT_CHAT_TEMPLATE)
                self.assertTrue(tokenizer.has_chat_template)
                self.assertFalse(ensure_chat_template(tokenizer))
                self.assertEqual(
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": "hello"}],
                        tokenize=False,
                        add_generation_prompt=True,
                    ),
                    "user: hello\nassistant: ",
                )

    def test_preserves_existing_template(self):
        for wrap in (lambda tokenizer: tokenizer, TokenizerWrapper):
            with self.subTest(wrapper=wrap):
                tokenizer = make_tokenizer()
                tokenizer.chat_template = "{{ messages[0]['content'] }}"
                tokenizer = wrap(tokenizer)
                self.assertFalse(ensure_chat_template(tokenizer))
                self.assertEqual(
                    tokenizer.chat_template, "{{ messages[0]['content'] }}"
                )

    def test_preserves_mlx_callable_template(self):
        def template(messages, **kwargs):
            return "custom template"

        tokenizer = TokenizerWrapper(make_tokenizer(), chat_template=template)
        self.assertFalse(ensure_chat_template(tokenizer))
        self.assertIsNone(tokenizer.chat_template)
        self.assertEqual(
            tokenizer.apply_chat_template([], tokenize=False), "custom template"
        )

    def test_sft_datasets_include_assistant_eos(self):
        tokenizer = TokenizerWrapper(make_tokenizer())
        ensure_chat_template(tokenizer)
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        self.assertEqual(
            tokenizer.apply_chat_template(messages, tokenize=False),
            "user: hello\nassistant: hi</s>\n",
        )
        for dataset in (
            ChatDataset([{"messages": messages}], tokenizer),
            CompletionsDataset(
                [{"prompt": "hello", "completion": "hi"}],
                tokenizer,
                "prompt",
                "completion",
                False,
            ),
        ):
            with self.subTest(dataset=type(dataset).__name__):
                self.assertEqual(
                    dataset.process(dataset[0])[-1], tokenizer.eos_token_id
                )


if __name__ == "__main__":
    unittest.main()
