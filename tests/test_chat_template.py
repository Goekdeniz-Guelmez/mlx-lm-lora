import unittest

from mlx_lm_lora.chat_template import (
    DEFAULT_CHAT_TEMPLATE,
    ensure_chat_template,
    has_chat_template,
)


class BareTokenizer:
    """A tokenizer that behaves like a base checkpoint: no chat template."""

    def __init__(self):
        self.chat_template = None

    def apply_chat_template(self, messages, **kwargs):
        raise ValueError(
            "Cannot use chat template functions because tokenizer.chat_template "
            "is not set and no template argument was passed!"
        )


class TemplatedTokenizer(BareTokenizer):
    def __init__(self):
        super().__init__()
        self.chat_template = "{{ messages[0]['content'] }}"


class Wrapper:
    """Stands in for mlx_lm's TokenizerWrapper."""

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self._chat_template = None
        self.has_chat_template = tokenizer.chat_template is not None


class ChatTemplateTest(unittest.TestCase):
    def test_default_template_renders_roles(self):
        from jinja2 import Template

        rendered = Template(DEFAULT_CHAT_TEMPLATE).render(
            messages=[
                {"role": "system", "content": "be nice"},
                {"role": "user", "content": "hello"},
            ],
            add_generation_prompt=True,
        )
        self.assertIn("system: be nice", rendered)
        self.assertIn("user: hello", rendered)
        self.assertTrue(rendered.endswith("assistant: "))

    def test_has_chat_template_detects_missing_template(self):
        self.assertFalse(has_chat_template(BareTokenizer()))
        self.assertTrue(has_chat_template(TemplatedTokenizer()))

    def test_ensure_installs_template_when_missing(self):
        tokenizer = BareTokenizer()
        self.assertTrue(ensure_chat_template(tokenizer))
        self.assertIsNotNone(tokenizer.chat_template)
        self.assertTrue(has_chat_template(tokenizer))

    def test_ensure_keeps_existing_template(self):
        tokenizer = TemplatedTokenizer()
        original = tokenizer.chat_template
        self.assertFalse(ensure_chat_template(tokenizer))
        self.assertEqual(tokenizer.chat_template, original)

    def test_ensure_patches_wrapped_tokenizer(self):
        inner = BareTokenizer()
        wrapper = Wrapper(inner)
        self.assertFalse(has_chat_template(wrapper))
        self.assertTrue(ensure_chat_template(wrapper))
        self.assertIsNotNone(inner.chat_template)
        self.assertTrue(wrapper.has_chat_template)
        self.assertTrue(has_chat_template(wrapper))

    def test_ensure_is_noop_without_template(self):
        tokenizer = BareTokenizer()
        self.assertFalse(ensure_chat_template(tokenizer, template=None))
        self.assertIsNone(tokenizer.chat_template)


if __name__ == "__main__":
    unittest.main()
