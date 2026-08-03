"""Shared prompt formatting and device helpers."""

import torch


def format_prompt(tokenizer, prompt: str) -> str:
    """Format a user prompt exactly as it will be formatted for generation."""
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def get_input_device(model) -> torch.device:
    """Return the embedding device, which is where input IDs must be placed."""
    try:
        return model.get_input_embeddings().weight.device
    except (AttributeError, StopIteration):
        return next(model.parameters()).device
