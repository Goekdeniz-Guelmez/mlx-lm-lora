"""Chat-template helpers.

Base (non instruct) checkpoints are frequently used for LoRA fine-tuning, but
they ship without a chat template.  Every dataset builder in
:mod:`mlx_lm_lora.trainer.datasets` calls ``tokenizer.apply_chat_template(...)``
without a fallback, so training such a model aborts with::

    ValueError: Cannot use chat template functions because tokenizer.chat_template
    is not set and no template argument was passed!

The helpers below install a small, tokenizer agnostic template so those
checkpoints stay trainable.  A checkpoint that already provides a template is
never modified.
"""

from typing import Any, Optional

#: A minimal, tokenizer agnostic chat template.
#:
#: It intentionally avoids special tokens such as ``<|im_start|>`` because they
#: are not present in every vocabulary.  Role names are rendered as plain text
#: so the template works for any tokenizer.
DEFAULT_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ message['role'] }}: {{ message['content'] }}\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}assistant: {% endif %}"
)


def _inner_tokenizer(tokenizer: Any) -> Any:
    """Return the Hugging Face tokenizer wrapped by ``tokenizer``, if any."""
    return getattr(tokenizer, "_tokenizer", None)


def has_chat_template(tokenizer: Any) -> bool:
    """Return ``True`` when ``tokenizer`` can render a chat template.

    Both the raw Hugging Face tokenizers and the ``mlx_lm`` ``TokenizerWrapper``
    are supported.  A wrapper is considered capable when it carries its own
    template callable or when the wrapped tokenizer exposes ``chat_template``.
    """
    if getattr(tokenizer, "_chat_template", None) is not None:
        return True

    inner = _inner_tokenizer(tokenizer)
    target = inner if inner is not None else tokenizer
    return getattr(target, "chat_template", None) is not None


def ensure_chat_template(
    tokenizer: Any, template: Optional[str] = DEFAULT_CHAT_TEMPLATE
) -> bool:
    """Install ``template`` on ``tokenizer`` when it has no chat template.

    Args:
        tokenizer: The tokenizer to patch.  Both raw Hugging Face tokenizers and
            ``mlx_lm`` ``TokenizerWrapper`` instances are supported.
        template: The Jinja template to install.  ``None`` disables the
            fallback and is a no-op.

    Returns:
        ``True`` when a fallback template was installed, ``False`` when the
        tokenizer already had a usable template (or could not be patched).
    """
    if template is None or has_chat_template(tokenizer):
        return False

    inner = _inner_tokenizer(tokenizer)
    target = inner if inner is not None else tokenizer

    try:
        target.chat_template = template
    except Exception:  # pragma: no cover - defensive, tokenizer specific
        return False

    # ``TokenizerWrapper`` caches the capability flag at construction time.
    try:
        tokenizer.has_chat_template = True
    except Exception:  # pragma: no cover - defensive, tokenizer specific
        pass

    return True
