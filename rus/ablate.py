"""
RUS Ablation Engine — project refusal directions out of model weights.
Permanently modifies the model so it can no longer represent refusal.
"""

import torch
from typing import Dict, List, Tuple
from tqdm import tqdm

from .loader import get_weight_targets


TARGET_SUFFIXES = {
    "o_proj": "self_attn.o_proj",
    "c_proj": "attn.c_proj",
    "wo": "attention.wo",
    "down_proj": "mlp.down_proj",
    "c_proj_mlp": "mlp.c_proj",
    "fc2": "mlp.fc2",
}


def project_direction_from_weight(
    weight: torch.Tensor,
    direction: torch.Tensor,
    coefficient: float = 0.8,
    output_axis: int = 0,
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

    if output_axis == 0:
        if weight.shape[0] != v_hat.numel():
            raise ValueError(f"direction has {v_hat.numel()} values for weight {tuple(weight.shape)}")
        col_projections = v_hat @ weight
        weight_modified = weight - coefficient * torch.outer(v_hat, col_projections)
    elif output_axis == 1:
        if weight.shape[1] != v_hat.numel():
            raise ValueError(f"direction has {v_hat.numel()} values for weight {tuple(weight.shape)}")
        row_projections = weight @ v_hat
        weight_modified = weight - coefficient * torch.outer(row_projections, v_hat)
    else:
        raise ValueError("output_axis must be 0 or 1")

    return weight_modified


# ── 8-bit (bitsandbytes) weight support ─────────────────────────────────
# bnb's Linear8bitLt keeps int8 blocks (CB) + fp16 scales (SCB), and the
# storage layout differs across bnb/transformers versions (attributes on the
# weight param vs. on the module state). Writing back into those slots proved
# unreliable, so we never mutate them. Instead we:
#   1. read the authoritative CB/SCB pair via module.state_dict() — bnb's own
#      method, always correct,
#   2. dequantize → project in fp16,
#   3. REPLACE the module with a plain fp16 nn.Linear holding the projected
#      weights. Only the ablated targets become fp16 (~MBs), the rest of the
#      model stays 8-bit. The exporter saves state dicts, so the checkpoint
#      (mixed fp16 + bnb 8-bit) is a valid transformers checkpoint.


def _block_size(cb: torch.Tensor, scb: torch.Tensor) -> int:
    """Infer the bnb quantization block size from CB/SCB shapes."""
    numel = cb.numel()
    n_scales = scb.numel()
    if n_scales == 0 or numel % n_scales != 0:
        raise RuntimeError(f"Cannot infer 8-bit block size ({numel} elems, {n_scales} scales).")
    return numel // n_scales


def dequantize_8bit(cb: torch.Tensor, scb: torch.Tensor) -> torch.Tensor:
    """
    Convert 8-bit blocks (int8 CB + fp16 absmax scales) to fp16.
    Every `block_size` flattened elements share one scale (bnb absmax scheme).
    """
    block = _block_size(cb, scb)
    fp = cb.float().reshape(-1, block) * (scb.float().reshape(-1, 1) / 127.0)
    return fp.reshape(cb.shape).to(torch.float16)


def _replace_with_fp16_linear(model, layer_path: str, tag: str, fp16_weights: torch.Tensor, bias=None) -> None:
    """
    Swap the bnb 8-bit module for a plain fp16 nn.Linear with the projected
    weights. The rest of the model keeps its 8-bit layout.
    """
    import torch.nn as nn

    suffix = TARGET_SUFFIXES.get(tag)
    if suffix is None:
        raise RuntimeError(f"No module path mapping for target tag: {tag}")

    parent = model
    parts = f"{layer_path}.{suffix}".split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    attr = parts[-1]

    in_features, out_features = fp16_weights.shape[1], fp16_weights.shape[0]
    new_layer = nn.Linear(in_features, out_features, bias=bias is not None)
    new_layer = new_layer.to(device=fp16_weights.device, dtype=fp16_weights.dtype)
    with torch.no_grad():
        new_layer.weight.copy_(fp16_weights.contiguous())
        if bias is not None:
            new_layer.bias.copy_(bias)
        new_layer.weight.requires_grad = False
        if new_layer.bias is not None:
            new_layer.bias.requires_grad = False

    setattr(parent, attr, new_layer)
    return new_layer


def _extract_cb_scb(module, sd: Dict) -> tuple:
    """
    Resolve the CB/SCB pair from a bnb 8-bit module, regardless of where bnb
    stashes the scales. Variants seen in the wild:
      a) state_dict has an "SCB" or "weight.SCB" key,
      b) SCB lives as an attribute on the weight param itself (Int8Params.SCB),
      c) CB/SCB have moved to Linear8bitLt.state after the first forward pass.
    Returns (cb, scb) or (None, None) for a non-8-bit module.
    """
    weight = getattr(module, "weight", None)
    state = getattr(module, "state", None)
    cb = sd.get("weight")
    if not isinstance(cb, torch.Tensor):
        cb = getattr(weight, "CB", None)
    if not isinstance(cb, torch.Tensor):
        cb = getattr(state, "CB", None)
    if not isinstance(cb, torch.Tensor):
        return None, None
    scb = sd.get("SCB")
    if scb is None:
        scb = sd.get("weight.SCB")
    if scb is None:
        scb = getattr(weight, "SCB", None) if weight is not None else None
    if scb is None:
        scb = getattr(state, "SCB", None) if state is not None else None
    if not isinstance(scb, torch.Tensor):
        return None, None
    return cb, scb


def _ablate_quantized_target(model, layer_path: str, tag: str, module, direction: torch.Tensor,
                             coefficient: float) -> Dict[str, float]:
    """Dequantize (via module.state_dict) → project → swap in fp16 Linear."""
    sd = module.state_dict()
    cb, scb = _extract_cb_scb(module, sd)
    if cb is None or scb is None:
        raise RuntimeError(
            f"Unsupported bitsandbytes layout for {module.__class__.__name__}: "
            f"state_dict keys = {list(sd.keys())}"
        )

    fp16 = dequantize_8bit(cb, scb).to(cb.device)
    dir_fp16 = direction.to(fp16.device, torch.float16)

    proj_before = (dir_fp16 @ fp16).abs().mean().item()
    fp16_modified = project_direction_from_weight(fp16, dir_fp16, coefficient)
    proj_after = (dir_fp16 @ fp16_modified).abs().mean().item()

    bias = None
    if hasattr(module, "bias") and module.bias is not None:
        bias = module.bias.detach().to(torch.float16)

    new_module = _replace_with_fp16_linear(model, layer_path, tag, fp16_modified, bias)

    # VERIFY: the live module now holds weights with the direction removed.
    # Plain fp16 weights — measurement is deterministic, no bnb layout tricks.
    check_proj = (dir_fp16 @ new_module.weight).abs().mean().item()  # module now nn.Linear
    expected = proj_before * abs(1.0 - coefficient)
    tolerance = max(proj_before * 0.08, 2e-5)
    if abs(check_proj - expected) > tolerance:
        raise RuntimeError(
            f"Ablation did not take effect in the module "
            f"(projection {proj_before:.5f} -> {check_proj:.5f})."
        )

    return {
        "projection_before": proj_before,
        "projection_after": proj_after,
        "reduction": (proj_before - proj_after) / max(proj_before, 1e-12),
        "quantized": True,
    }


def _ablate_4bit_target(model, layer_path: str, tag: str, module,
                        direction: torch.Tensor, coefficient: float) -> Dict[str, float]:
    """Dequantize a bitsandbytes Params4bit weight and replace it with fp16."""
    weight = module.weight
    quant_state = getattr(weight, "quant_state", None)
    if quant_state is None:
        raise RuntimeError(f"Missing 4-bit quantization state for {layer_path}.{tag}")
    try:
        import bitsandbytes.functional as bnbf
    except ImportError as exc:
        raise RuntimeError("4-bit ablation requires bitsandbytes") from exc

    fp16 = bnbf.dequantize_4bit(weight.data, quant_state=quant_state).to(torch.float16)
    dir_fp16 = direction.to(fp16.device, torch.float16)
    if fp16.shape[0] != dir_fp16.numel():
        raise RuntimeError(
            f"Direction/weight mismatch for {layer_path}.{tag}: "
            f"{dir_fp16.numel()} vs {tuple(fp16.shape)}"
        )
    before = (dir_fp16 @ fp16).abs().mean().item()
    modified = project_direction_from_weight(fp16, dir_fp16, coefficient)
    after = (dir_fp16 @ modified).abs().mean().item()
    bias = getattr(module, "bias", None)
    if bias is not None:
        bias = bias.detach().to(device=modified.device, dtype=torch.float16)
    _replace_with_fp16_linear(model, layer_path, tag, modified, bias)
    return {
        "projection_before": before,
        "projection_after": after,
        "reduction": (before - after) / max(before, 1e-12),
        "quantized": True,
        "quantization_bits": 4,
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
    if not targets:
        raise RuntimeError(f"No compatible output-projection weights in {layer_path}")
    stats = {}

    for tag, weight in targets.items():
        module = _find_module(model, layer_path, tag)
        if module is None:
            raise RuntimeError(f"Cannot locate module for {layer_path}.{tag}")

        if getattr(getattr(module, "weight", None), "quant_state", None) is not None:
            stats[tag] = _ablate_4bit_target(
                model, layer_path, tag, module, direction, coefficient
            )
            continue

        sd = module.state_dict()
        cb, scb = _extract_cb_scb(module, sd)
        if cb is not None and scb is not None:  # bnb 8-bit module (either layout)
            print(f"    [8-bit] {layer_path}.{tag}: cb={tuple(cb.shape)} scb={tuple(scb.shape)}")
            stats[tag] = _ablate_quantized_target(model, layer_path, tag, module, direction, coefficient)
            continue
        if weight.dtype in (torch.int8, torch.uint8):
            raise RuntimeError(
                f"Packed quantized weight at {layer_path}.{tag} has no readable "
                f"quantization state (state_dict keys={list(sd.keys())}). "
                "Refusing to treat it as a plain matrix."
            )
        print(f"    [plain] {layer_path}.{tag}: dtype={weight.dtype}")

        # Plain fp16 path
        direction_dev = direction.to(weight.device, weight.dtype)

        output_axis = 1 if module.__class__.__name__ == "Conv1D" else 0
        proj_before = (
            (direction_dev @ weight) if output_axis == 0 else (weight @ direction_dev)
        ).abs().mean().item()

        weight_modified = project_direction_from_weight(
            weight, direction_dev, coefficient, output_axis=output_axis
        )

        with torch.no_grad():
            weight.copy_(weight_modified)

        proj_after = (
            (direction_dev @ weight) if output_axis == 0 else (weight @ direction_dev)
        ).abs().mean().item()

        stats[tag] = {
            "projection_before": proj_before,
            "projection_after": proj_after,
            "reduction": (proj_before - proj_after) / max(proj_before, 1e-12),
        }

    return stats


def _find_module(model, layer_path: str, tag: str):
    """Walk dotted paths to the nn.Module holding a weight target."""
    suffix = TARGET_SUFFIXES.get(tag)
    if suffix is None:
        return None
    obj = model
    try:
        for part in f"{layer_path}.{suffix}".split("."):
            obj = getattr(obj, part)
        return obj if hasattr(obj, "weight") else None
    except (AttributeError, TypeError):
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
    if not 0.0 < coefficient <= 1.0:
        raise ValueError("coefficient must be in (0, 1]")
    if not 0.0 < coefficient_decay <= 1.0:
        raise ValueError("coefficient_decay must be in (0, 1]")
    all_stats = {}

    for rank, (layer_idx, refusal_score, direction) in enumerate(
        tqdm(selected_layers, desc="Ablating layers", unit="layer")
    ):
        coeff = coefficient * (coefficient_decay ** rank)
        if layer_idx < 0 or layer_idx >= len(layer_paths):
            raise IndexError(f"Layer index {layer_idx} is outside the model")
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
