"""
RUS Activation Collector — runs prompts through the model and captures per-layer
residual stream activations at the last token position.
"""

import torch
from typing import List, Dict
from tqdm import tqdm

from .config import MAX_MODEL_LENGTH, DEFAULT_BATCH_SIZE


def collect_activations(
    model,
    tokenizer,
    prompts: List[str],
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_length: int = MAX_MODEL_LENGTH,
) -> Dict[int, List[torch.Tensor]]:
    """
    Run all prompts through the model and collect per-layer hidden states
    at the LAST token position (representing the model's 'decision' state).

    Returns: dict mapping layer_index -> list of activation tensors (one per prompt)
             Each activation tensor has shape (hidden_dim,)
    """
    device = next(model.parameters()).device
    model.config.output_hidden_states = True

    all_hidden = {}  # layer_idx -> [tensor(batch, hidden_dim), ...]

    for i in tqdm(range(0, len(prompts), batch_size), desc="Collecting activations", unit="batch"):
        batch_prompts = prompts[i : i + batch_size]

        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            outputs = model(**enc)

        hidden_states = outputs.hidden_states  # tuple of (batch, seq_len, hidden_dim)
        last_positions = enc["attention_mask"].sum(dim=1) - 1

        for layer_idx, hs in enumerate(hidden_states):
            last_hidden = hs[range(len(batch_prompts)), last_positions, :]
            if layer_idx not in all_hidden:
                all_hidden[layer_idx] = []
            all_hidden[layer_idx].append(last_hidden.cpu())

    model.config.output_hidden_states = False
    return all_hidden


def collect_pairwise_activations(
    model,
    tokenizer,
    harmful_prompts: List[str],
    harmless_prompts: List[str],
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_length: int = MAX_MODEL_LENGTH,
) -> tuple:
    """
    Collect activations for both harmful and harmless prompt sets.
    Returns (harmful_acts, harmless_acts) where each is {layer_idx: [tensors]}.
    """
    harmful_acts = collect_activations(model, tokenizer, harmful_prompts, batch_size, max_length)
    harmless_acts = collect_activations(model, tokenizer, harmless_prompts, batch_size, max_length)
    return harmful_acts, harmless_acts
