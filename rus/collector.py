"""
RUS Activation Collector — runs prompts through the model and captures per-layer
residual stream activations at the last token position.
"""

import torch
from typing import List, Dict
from tqdm import tqdm

from .config import MAX_MODEL_LENGTH, DEFAULT_BATCH_SIZE
from .formatting import format_prompt, get_input_device


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
    if not prompts:
        return {}
    device = get_input_device(model)
    model.config.output_hidden_states = True

    all_hidden = {}  # layer_idx -> [tensor(batch, hidden_dim), ...]

    try:
        for i in tqdm(range(0, len(prompts), batch_size), desc="Collecting activations", unit="batch"):
            batch_prompts = [format_prompt(tokenizer, p) for p in prompts[i : i + batch_size]]

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

            hidden_states = outputs.hidden_states  # tuple (num_layers+1, batch, seq, hidden)
            token_positions = torch.arange(enc["attention_mask"].shape[1], device=device)
            last_positions = (enc["attention_mask"] * token_positions).argmax(dim=1)

            # hidden_states[i] is the input to layer i (i=0: embeddings);
            # hidden_states[i+1] is layer i's OUTPUT. Store under key `i` so the
            # refusal direction for layer i matches the weights we ablate there.
            for layer_idx, hs in enumerate(hidden_states):
                if layer_idx == 0:
                    continue
                batch_indices = torch.arange(len(batch_prompts), device=hs.device)
                last_hidden = hs[batch_indices, last_positions.to(hs.device), :]
                if (layer_idx - 1) not in all_hidden:
                    all_hidden[layer_idx - 1] = []
                all_hidden[layer_idx - 1].append(last_hidden.cpu())
            del outputs, hidden_states, enc, last_hidden
    finally:
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
