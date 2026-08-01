"""
Unit tests for RUS core math: 8-bit quant round-trip, direction projection,
and the sharded exporter. Run: python tests/test_core.py
"""

import os
import sys
import shutil
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from rus.ablate import (
    dequantize_8bit,
    requantize_8bit,
    project_direction_from_weight,
    _quant_block_size,
    _get_scb,
)
from rus.exporter import _shard_state_dict, _save_shards


class FakeQuantWeight:
    """Mimics the parts of bnb Int8Params the ablation code touches."""

    def __init__(self, cb, scb):
        self.data = cb
        self.SCB = scb
        self.dtype = cb.dtype


class FakeBnbModule:
    """Mimics Linear8bitLt's weight + state slots."""

    def __init__(self, cb, scb):
        self.weight = FakeQuantWeight(cb, scb)
        self.state = type(
            "FakeState",
            (),
            {"CB": cb, "SCB": scb, "has_fp16_weights": False},
        )()


def quantize_absmax(fp16: torch.Tensor, block: int) -> tuple:
    """Mirror bnb's absmax 8-bit scheme: CB in [-127,127], SCB = raw absmax."""
    flat = fp16.float().reshape(-1, block)
    absmax = flat.abs().amax(dim=1)
    cb = torch.clamp(torch.round(flat * (127.0 / absmax.clamp_min(1e-12).unsqueeze(1))), -127, 127)
    return cb.reshape(fp16.shape).to(torch.int8), absmax.half()


def test_round_trip():
    torch.manual_seed(0)
    w = (torch.randn(3584, 3584) * 0.02).half()  # like a Qwen o_proj
    for block in (64, 128, 256):
        cb, scb = quantize_absmax(w, block)
        fake = FakeQuantWeight(cb, scb)
        assert _quant_block_size(fake, scb) == block, f"block detect {block}"
        deq = dequantize_8bit(fake, scb)
        rel_err = (deq.float() - w.float()).abs().mean() / w.float().abs().mean()
        assert rel_err < 0.02, f"dequant rel err {rel_err:.4f}"
        cb2, scb2 = requantize_8bit(deq, block)
        assert cb2.dtype == torch.int8 and scb2.dtype == torch.float16
        deq2 = FakeQuantWeight(cb2, scb2)
        round_trip = dequantize_8bit(deq2)
        err2 = (round_trip.float() - w.float()).abs().mean() / w.float().abs().mean()
        assert err2 < 0.02, f"requant rel err {err2:.4f}"
    print("PASS test_round_trip")


def test_projection():
    torch.manual_seed(1)
    w = torch.randn(256, 256).half()
    v = torch.randn(256).half()
    wm = project_direction_from_weight(w, v, coefficient=0.8)
    vn = v / v.norm()
    before = (vn @ w).abs().mean().item()
    after = (vn @ wm).abs().mean().item()
    assert after < before * 0.3, f"projection not removed: {before} -> {after}"
    print(f"PASS test_projection ({before:.4f} -> {after:.4f})")


def test_quantized_ablation_path():
    """Full 8-bit branch: dequantize → project → requantize → write back."""
    from rus.ablate import _ablate_quantized_target

    torch.manual_seed(2)
    w = (torch.randn(256, 256) * 0.02).half()
    cb, scb = quantize_absmax(w, 64)
    mod = FakeBnbModule(cb, scb)
    v = torch.randn(256).half()

    stats = _ablate_quantized_target(mod, v, coefficient=0.8)
    assert stats["reduction"] > 0.7, stats

    # written-back weights must have the direction projected away
    deq = dequantize_8bit(FakeQuantWeight(mod.weight.data, mod.weight.SCB))
    vn = v / v.norm()
    before = (vn @ w).abs().mean().item()
    after = (vn @ deq).abs().mean().item()
    assert after < before * 0.3, f"{before} -> {after}"
    # state slots synced too
    assert mod.state.CB is mod.weight.data and mod.state.SCB is mod.weight.SCB
    print(f"PASS test_quantized_ablation_path ({before:.4f} -> {after:.4f})")


def test_exporter_shards(tmp="/tmp/rus_export_test"):
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    t = torch.randn(100, 100)
    state = {
        "a": t,
        "b": t[:],  # shared storage — must be deduplicated
        "c.weight": torch.randn(10, 10),
    }
    from rus.exporter import _dedupe_tied
    deduped = _dedupe_tied(state)
    assert "b" not in deduped or "a" not in deduped, deduped.keys()
    with torch.no_grad():
        weight_map = _save_shards(deduped, tmp)
    files = sorted(os.listdir(tmp))
    assert any(f.endswith(".safetensors") for f in files), files
    if len(files) > 1:
        assert "model.safetensors.index.json" in files
    # reload and compare
    import safetensors.torch as st
    keys = []
    for f in files:
        if f.endswith(".safetensors"):
            keys += list(st.load_file(os.path.join(tmp, f)).keys())
    assert sorted(keys) == sorted(deduped.keys()), (sorted(keys), sorted(deduped))
    shutil.rmtree(tmp, ignore_errors=True)
    print("PASS test_exporter_shards")


def test_select_best_layers():
    """Low absolute scores (normal with many prompts) must still yield top-k."""
    from rus.subspace import select_best_layers

    ranked = [(2, 0.28, torch.randn(8)), (3, 0.23, torch.randn(8)), (4, 0.19, torch.randn(8))]
    selected = select_best_layers(ranked, top_k=2, min_score=0.3)
    assert [l for l, _, _ in selected] == [2, 3], "rank-based fallback failed"
    assert select_best_layers([], top_k=2) == []
    print("PASS test_select_best_layers")


def test_quantized_scb_on_state_only():
    """SCB lives on module.state.SCB, not weight.SCB (bnb layout variant)."""
    from rus.ablate import _ablate_quantized_target

    torch.manual_seed(3)
    w = (torch.randn(256, 256) * 0.02).half()
    cb, scb = quantize_absmax(w, 64)
    mod = FakeBnbModule(cb, scb)
    del mod.weight.SCB  # simulate the variant where weight has no SCB
    mod.state.SCB = scb

    v = torch.randn(256).half()
    stats = _ablate_quantized_target(mod, v, coefficient=0.8)
    assert stats["reduction"] > 0.7, stats

    deq = dequantize_8bit(mod.weight, _get_scb(mod.weight, mod))
    vn = v / v.norm()
    before = (vn @ w).abs().mean().item()
    after = (vn @ deq).abs().mean().item()
    assert after < before * 0.3, f"{before} -> {after}"
    print(f"PASS test_quantized_scb_on_state_only ({before:.4f} -> {after:.4f})")


def test_apply_ablation_raises_on_failure():
    """Ablation errors must propagate, never be swallowed."""
    from rus.ablate import apply_ablation

    good = FakeBnbModule(*quantize_absmax((torch.randn(256, 256) * 0.02).half(), 64))
    bad = FakeBnbModule(*quantize_absmax((torch.randn(256, 256) * 0.02).half(), 64))
    del bad.weight.data  # corrupt storage -> must raise

    model = type("M", (), {})()
    model.layers = [good, bad]
    good.self_attn = type("S", (), {"o_proj": good})()
    bad.self_attn = type("S", (), {"o_proj": bad})()
    layer_paths = ["model.layers.0", "model.layers.1"]
    v = torch.randn(256)
    selected = [(0, 0.5, v), (1, 0.4, v)]

    try:
        apply_ablation(model, selected, layer_paths)
        raise AssertionError("expected an exception to propagate")
    except (RuntimeError, AttributeError, TypeError):
        print("PASS test_apply_ablation_raises_on_failure")


if __name__ == "__main__":
    test_round_trip()
    test_projection()
    test_quantized_ablation_path()
    test_quantized_scb_on_state_only()
    test_apply_ablation_raises_on_failure()
    test_exporter_shards()
    test_select_best_layers()
    print("\nAll tests passed ✓")
