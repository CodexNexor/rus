"""
RUS Ablation Engine — project refusal directions out of model weights.
Permanently modifies the model so it can no longer represent refusal.
"""

import torch
from typing import Dict, List, Tuple
from tqdm import tqdm

from .loader import get_weight_targets, discover_layers


def project_direction_from_weight(
    weight: torch.Tensor,
    direction: torch.Tensor,
    coefficient: float = 0.8,
) -> torch.Tensor:
    """
    Remove the component of `direction` from each column of `weight`.

    For output projections (o_proj, down_proj), the refusal direction v_hat lives
    in the OUTPUT space (d_out). We project v_hat out of each column c_j so that
    for any input x, the output y = W @ x has zero component along v_hat.

    W' = W - coeff * v_hat @ outer(v_hat, W) = W - coeff * outer(v_hat, v_hat^T W)

    Args:
        weight: (d_out, d_in) weight matrix — maps from d_in → d_out
        direction: (d_out,) refusal direction in OUTPUT space
        coefficient: steering strength (0 = no change, 1 = full projection)

    Returns:
        Modified weight tensor.
    """
    v_hat = direction / (direction.norm() + 1e-12)
    v_hat = v_hat.to(weight.device, weight.dtype)

    col_projections = v_hat @ weight  # (d_in,) — v_hat · c_j for each column
    weight_modified = weight - coefficient * torch.outer(v_hat, col_projections)

    return weight_modified


def apply_ablation_to_layer(
    model,
    layer_path: str,
    direction: torch.Tensor,
    coefficient: float = 0.8,
) -> Dict[str, float]:
    """
    Apply refusal-direction projection to all modifiable weight targets in a layer.
    Returns stats dict with before/after metrics.
    """
    targets = get_weight_targets(model, layer_path)
    stats = {}

    for tag, weight in targets.items():
        direction_dev = direction.to(weight.device, weight.dtype)

        proj_before = (direction_dev @ weight).abs().mean().item()

        weight_modified = project_direction_from_weight(weight, direction_dev, coefficient)

        with torch.no_grad():
            weight.copy_(weight_modified)

        proj_after = (direction_dev @ weight).abs().mean().item()

        stats[tag] = {
            "projection_before": proj_before,
            "projection_after": proj_after,
            "reduction": (proj_before - proj_after) / max(proj_before, 1e-12),
        }

    return stats


def apply_ablation(
    model,
    selected_layers: List[Tuple[int, float, torch.Tensor]],
    layer_paths: List[str],
    coefficient: float = 0.8,
    coefficient_decay: float = 0.75,
) -> Dict:
    """
    Apply weight-projection ablation to all selected layers.

    Each subsequent layer gets a slightly reduced coefficient
    to avoid compounding effects that degrade quality.

    Returns:
        Dict with per-layer ablation statistics.
    """
    all_stats = {}

    for rank, (layer_idx, refusal_score, direction) in enumerate(
        tqdm(selected_layers, desc="Ablating layers", unit="layer")
    ):
        coeff = coefficient * (coefficient_decay ** rank)
        layer_path = layer_paths[layer_idx]

        try:
            stats = apply_ablation_to_layer(model, layer_path, direction, coeff)
            all_stats[layer_idx] = {
                "layer_path": layer_path,
                "coefficient": coeff,
                "refusal_score": refusal_score,
                "targets": stats,
            }
        except Exception as e:
            all_stats[layer_idx] = {
                "layer_path": layer_path,
                "coefficient": coeff,
                "refusal_score": refusal_score,
                "error": str(e),
            }

    return all_stats
