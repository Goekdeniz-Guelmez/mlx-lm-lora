"""Fallback chat formatting for base checkpoints without a template."""

DEFAULT_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ message['role'] }}: {{ message['content'] }}"
    "{{ (eos_token or '') if message['role'] == 'assistant' else '' }}\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}assistant: {% endif %}"
)


def ensure_chat_template(tokenizer) -> bool:
    """Install the fallback if neither an HF nor an MLX template is available."""
    if (
        getattr(tokenizer, "has_chat_template", False)
        or getattr(tokenizer, "chat_template", None) is not None
    ):
        return False

    tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE
    tokenizer.has_chat_template = True
    return True
