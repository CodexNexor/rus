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
    protect_harmless: bool = True,
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
        harmful_mean = h_harmful.mean(axis=0)
        harmless_mean = h_harmless.mean(axis=0)
        mean_diff = harmful_mean - harmless_mean
        direction = torch.from_numpy(mean_diff.astype(np.float32))
        harmless_direction = torch.from_numpy(harmless_mean.astype(np.float32))
        if harmless_direction.norm() > 1e-8:
            harmless_direction = harmless_direction / harmless_direction.norm()
        if protect_harmless and harmless_direction.norm() > 1e-8:
            direction = direction - torch.dot(direction, harmless_direction) * harmless_direction
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
            "harmless_overlap": float(
                torch.dot(direction, harmless_direction).abs()
                if harmless_direction.norm() > 1e-8 else 0.0
            ),
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


def build_consensus_direction(
    ranked: List[Tuple[int, float, torch.Tensor]],
    top_n: int = 5,
) -> Tuple[torch.Tensor, List[int]]:
    """Build a stable global direction from sign-aligned top layer estimates.

    Residual-stream bases are shared across transformer blocks, but empirical
    directions can flip sign and contain layer-specific noise. Aligning each
    candidate to the best-scoring reference and taking a score-weighted mean
    reduces that noise without increasing the erased subspace rank.
    """
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    candidates = ranked[:top_n]
    if not candidates:
        raise ValueError("Cannot build a consensus from no directions")
    reference = candidates[0][2].float()
    reference = reference / reference.norm().clamp_min(1e-12)
    aligned = []
    weights = []
    layers = []
    for layer_idx, score, direction in candidates:
        unit = direction.float() / direction.float().norm().clamp_min(1e-12)
        if torch.dot(unit, reference) < 0:
            unit = -unit
        aligned.append(unit)
        weights.append(max(float(score), 1e-6))
        layers.append(layer_idx)
    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    consensus = (torch.stack(aligned) * weight_tensor[:, None]).sum(dim=0)
    consensus = consensus / consensus.norm().clamp_min(1e-12)
    return consensus, layers
