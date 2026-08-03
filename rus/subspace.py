"""
RUS Subspace Analysis — extract refusal directions from paired differences.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple


def compute_refusal_directions(
    harmful_acts: Dict[int, List[torch.Tensor]],
    harmless_acts: Dict[int, List[torch.Tensor]],
    num_layers: int,
    min_refusal_score: float = 0.0,
) -> Dict[int, Dict]:
    """
    Compute the mean harmful-minus-harmless direction and an effect score.

    Returns:
        dict mapping layer_idx -> {
            "direction": torch.Tensor (hidden_dim,), unit vector
            "score": float (bounded standardized effect, 0-1)
            "effect_size": float
        }
    """
    results = {}

    for layer_idx in range(num_layers):
        if layer_idx not in harmful_acts or layer_idx not in harmless_acts:
            continue

        h_harmful = torch.cat(harmful_acts[layer_idx], dim=0).numpy()
        h_harmless = torch.cat(harmless_acts[layer_idx], dim=0).numpy()

        if min(len(h_harmful), len(h_harmless)) < 3:
            continue

        # Abliteration needs the harmful-vs-harmless mean displacement. PCA on
        # centered differentials instead captures variation *among* prompt pairs
        # and can be orthogonal to the actual class separation.
        mean_diff = h_harmful.mean(axis=0) - h_harmless.mean(axis=0)
        direction = torch.from_numpy(mean_diff.astype(np.float32))
        if not torch.isfinite(direction).all() or direction.norm() < 1e-8:
            continue
        direction = direction / direction.norm()

        harmful_projection = h_harmful @ direction.numpy()
        harmless_projection = h_harmless @ direction.numpy()
        pooled_std = np.sqrt(
            (harmful_projection.var(ddof=1) + harmless_projection.var(ddof=1)) / 2.0
        )
        effect = abs(float(harmful_projection.mean() - harmless_projection.mean())) / (
            float(pooled_std) + 1e-8
        )
        score = effect / (1.0 + effect)

        results[layer_idx] = {
            "direction": direction,
            "score": score,
            "effect_size": effect,
        }

    return results


def rank_layers(
    directions: Dict[int, Dict],
    skip_first: int = 2,
    skip_last: int = 3,
) -> List[Tuple[int, float, torch.Tensor]]:
    """
    Rank layers by refusal signal strength, skipping boundary layers.
    Returns list of (layer_idx, score, direction) sorted by score descending.
    """
    if not directions:
        return []

    total_layers = max(directions.keys()) + 1

    ranked = []
    for layer_idx, info in directions.items():
        if layer_idx < skip_first:
            continue
        if layer_idx >= total_layers - skip_last:
            continue
        ranked.append((layer_idx, info["score"], info["direction"]))

    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


def select_best_layers(
    ranked: List[Tuple[int, float, torch.Tensor]],
    top_k: int = 5,
    min_score: float = 0.0,
) -> List[Tuple[int, float, torch.Tensor]]:
    """
    Pick the top-k layers by refusal score.

    min_score is a filter when there are qualifying layers; ranking remains the
    fallback so models with differently scaled activations remain usable.
    """
    if not ranked:
        return []
    selected = [(l, s, d) for l, s, d in ranked if s >= min_score]
    if not selected:
        return ranked[:top_k]
    return selected[:top_k]
