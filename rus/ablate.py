"""
RUS Ablation Engine — project refusal directions out of model weights.
Permanently modifies the model so it can no longer represent refusal.
"""

import torch
from typing import Dict, List, Tuple, Optional
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

    W' = W - coeff * outer(v_hat, v_hat^T W)

    Args:
        weight: (d_out, d_in) weight matrix in fp16 — maps from d_in → d_out
        direction: (d_out,) refusal direction in OUTPUT space
        coefficient: steering strength (0 = no change, 1 = full projection)

    Returns:
        Modified weight tensor (same dtype/device as input).
    """
    v_hat = direction / (direction.norm() + 1e-12)
    v_hat = v_hat.to(weight.device, weight.dtype)

    col_projections = v_hat @ weight  # (d_in,) — v_hat · c_j for each column
    weight_modified = weight - coefficient * torch.outer(v_hat, col_projections)

    return weight_modified


# ── 8-bit (bitsandbytes) weight support ─────────────────────────────────
# Linear8bitLt stores weights as int8 blocks + fp16 scales (SCB). We cannot
# project int8 tensors, and writing fp16 into them would be a silent no-op.
# Instead: dequantize → project in fp16 → re-quantize with the same absmax
# block scheme → write back. Matches bnb's dynamic absmax 8-bit layout.


def _get_scb(weight, module=None) -> Optional[torch.Tensor]:
    """
    Locate the fp16 scale tensor. Depending on the bnb/transformers version it
    lives on the Int8Params (weight.SCB) or on the module's MatmulLtState.
    """
    scb = getattr(weight, "SCB", None)
    if scb is None and module is not None:
        state = getattr(module, "state", None)
        if state is not None:
            scb = getattr(state, "SCB", None)
    return scb


def _quant_block_size(weight, scb: torch.Tensor) -> int:
    """Infer the bnb quantization block size from the stored scale tensor."""
    cb = weight.data if hasattr(weight, "data") else weight
    numel = cb.numel()
    n_scales = scb.numel()
    if n_scales == 0 or numel % n_scales != 0:
        raise RuntimeError(f"Cannot infer 8-bit block size ({numel} elems, {n_scales} scales).")
    return numel // n_scales


def dequantize_8bit(weight, scb: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Convert an 8-bit quantized weight (int8 data + SCB scales) to fp16.
    Scale layout: every `block_size` flattened elements share one scale.
    """
    cb = weight.data if hasattr(weight, "data") else weight
    if scb is None:
        scb = getattr(weight, "SCB", None)
    if scb is None:
        raise RuntimeError("8-bit weight has no SCB scale tensor.")
    block = _quant_block_size(weight, scb)

    fp = cb.float().reshape(-1, block) * (scb.float().reshape(-1, 1) / 127.0)
    return fp.reshape(cb.shape).to(torch.float16)


def requantize_8bit(fp16_weight: torch.Tensor, block: int) -> tuple:
    """
    Quantize an fp16 matrix back into bnb's dynamic absmax 8-bit format.
    Returns (int8_tensor, fp16_scales) with one scale per `block` elements.
    """
    flat = fp16_weight.float().reshape(-1, block)
    absmax = flat.abs().amax(dim=1)
    scaled = torch.clamp(torch.round(flat * (127.0 / absmax.clamp_min(1e-12).unsqueeze(1))), -127, 127)
    return scaled.reshape(fp16_weight.shape).to(torch.int8), absmax.half().reshape(-1)


def _write_quantized_weight(module, cb: torch.Tensor, scb: torch.Tensor) -> None:
    """Write re-quantized CB/SCB back into the bnb module (all storage slots)."""
    weight = module.weight
    weight.data = cb
    if hasattr(weight, "SCB"):
        weight.SCB = scb
    state = getattr(module, "state", None)
    if state is not None:
        if hasattr(state, "CB"):
            state.CB = cb
        if hasattr(state, "SCB"):
            state.SCB = scb
        if hasattr(state, "has_fp16_weights"):
            state.has_fp16_weights = False


def _ablate_quantized_target(module, direction: torch.Tensor, coefficient: float) -> Dict[str, float]:
    """Dequantize → project → requantize for one 8-bit weight matrix."""
    weight = module.weight
    scb = _get_scb(weight, module)
    fp16 = dequantize_8bit(weight, scb)
    device = getattr(weight, "device", None) or fp16.device
    fp16 = fp16.to(device)
    dir_fp16 = direction.to(device, torch.float16)

    proj_before = (dir_fp16 @ fp16).abs().mean().item()
    fp16_modified = project_direction_from_weight(fp16, dir_fp16, coefficient)
    proj_after = (dir_fp16 @ fp16_modified).abs().mean().item()

    block = _quant_block_size(weight, scb)
    new_cb, new_scb = requantize_8bit(fp16_modified, block)
    _write_quantized_weight(module, new_cb, new_scb)

    # VERIFY the write-back took effect in the module's live storage.
    # If bnb keeps the weights elsewhere (wrong SCB slot etc.) this catches it
    # instead of silently saving an unmodified model.
    check_fp16 = dequantize_8bit(module.weight, _get_scb(module.weight, module))
    check_proj = (dir_fp16 @ check_fp16).abs().mean().item()
    if check_proj > proj_before * 0.5:
        raise RuntimeError(
            f"8-bit write-back did not take effect "
            f"(projection {proj_before:.5f} -> {check_proj:.5f}). "
            f"Unsupported bitsandbytes layout for {module.__class__.__name__}."
        )

    return {
        "projection_before": proj_before,
        "projection_after": proj_after,
        "reduction": (proj_before - proj_after) / max(proj_before, 1e-12),
        "quantized": True,
    }


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
        module = _find_module(model, layer_path, tag)
        scb = _get_scb(weight, module)
        is_quant = weight.dtype in (torch.int8, torch.uint8) or hasattr(weight, "SCB") or scb is not None

        if is_quant:
            if module is None:
                raise RuntimeError(f"8-bit weight found but module not located: {layer_path}.{tag}")
            stats[tag] = _ablate_quantized_target(module, direction, coefficient)
            continue

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


def _find_module(model, layer_path: str, tag: str):
    """Walk dotted paths to the nn.Module holding a weight target."""
    candidates = [
        f"{layer_path}.self_attn.o_proj",
        f"{layer_path}.self_attn.c_proj",
        f"{layer_path}.attention.wo",
        f"{layer_path}.mlp.down_proj",
        f"{layer_path}.mlp.c_proj",
        f"{layer_path}.mlp.fc2",
    ]
    for path in candidates:
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            if hasattr(obj, "weight"):
                return obj
        except (AttributeError, TypeError):
            continue
    return None


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

        # NOTE: errors propagate on purpose. A silently-failed ablation would
        # save an unmodified model and fake the whole run.
        stats = apply_ablation_to_layer(model, layer_path, direction, coeff)
        all_stats[layer_idx] = {
            "layer_path": layer_path,
            "coefficient": coeff,
            "refusal_score": refusal_score,
            "targets": stats,
        }

    return all_stats
