"""
RUS Subspace Analysis — extract refusal directions from activation differences
using PCA on differential pairs (RepE standard method).
"""

import torch
import numpy as np
from typing import List, Dict, Tuple
from sklearn.decomposition import PCA


def compute_refusal_directions(
    harmful_acts: Dict[int, List[torch.Tensor]],
    harmless_acts: Dict[int, List[torch.Tensor]],
    num_layers: int,
    min_refusal_score: float = 0.3,
) -> Dict[int, Dict]:
    """
    For each layer, compute the refusal direction via PCA on activation differentials.

    Returns:
        dict mapping layer_idx -> {
            "direction": torch.Tensor (hidden_dim,), unit vector
            "score": float (explained variance ratio, 0-1)
            "explained_variances": list[float]
        }
    """
    results = {}

    for layer_idx in range(num_layers):
        if layer_idx not in harmful_acts or layer_idx not in harmless_acts:
            continue

        h_harmful = torch.cat(harmful_acts[layer_idx], dim=0).numpy()
        h_harmless = torch.cat(harmless_acts[layer_idx], dim=0).numpy()

        min_n = min(len(h_harmful), len(h_harmless))
        h_harmful = h_harmful[:min_n]
        h_harmless = h_harmless[:min_n]

        differentials = h_harmful - h_harmless

        if differentials.shape[0] < 3:
            continue

        pca = PCA(n_components=1)
        pca.fit(differentials)

        direction = torch.from_numpy(pca.components_[0].astype(np.float32))
        direction = direction / direction.norm()

        score = float(pca.explained_variance_ratio_[0])

        results[layer_idx] = {
            "direction": direction,
            "score": score,
            "explained_variances": [float(v) for v in pca.explained_variance_ratio_],
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

    min_score is only a filter when there ARE qualifying layers. With many
    prompts the PCA explained-variance ratios are naturally low (0.1-0.3),
    so an absolute threshold must never empty the selection — ranking decides.
    """
    if not ranked:
        return []
    selected = [(l, s, d) for l, s, d in ranked if s >= min_score]
    if not selected:
        return ranked[:top_k]
    return selected[:top_k]
