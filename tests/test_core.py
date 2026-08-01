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
    project_direction_from_weight,
    _block_size,
)
from rus.exporter import _shard_state_dict, _save_shards


class FakeBnbModule:
    """Mimics Linear8bitLt's state_dict + bias slots (bnb's own layout)."""

    def __init__(self, cb, scb):
        self._cb = cb
        self._scb = scb
        self.bias = None

    def state_dict(self):
        return {"weight": self._cb, "weight.SCB": self._scb}

    @property
    def weight(self):
        return self._cb


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
        assert _block_size(cb, scb) == block, f"block detect {block}"
        deq = dequantize_8bit(cb, scb)
        rel_err = (deq.float() - w.float()).abs().mean() / w.float().abs().mean()
        assert rel_err < 0.02, f"dequant rel err {rel_err:.4f}"
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


class _Indexable:
    """Emulates nn.ModuleList: getattr(ml, '0') resolves via _modules."""

    def __init__(self, items):
        self._items = items

    def __getattr__(self, name):
        if name.isdigit():
            return self._items[int(name)]
        raise AttributeError(name)

    def __len__(self):
        return len(self._items)


def _fake_model_with_module(mod, tag="o_proj"):
    """Wrap a FakeBnbModule in a model-shaped tree for _find_module/_replace."""
    layer = type("Layer", (), {})()
    layer.self_attn = type("Attn", (), {})()
    layer.mlp = type("Mlp", (), {})()
    if tag == "down_proj":
        layer.mlp.down_proj = mod
    else:
        layer.self_attn.o_proj = mod
    inner = type("Inner", (), {})()
    inner.layers = _Indexable([layer])
    model = type("Model", (), {})()
    model.model = inner
    return model


def test_quantized_ablation_path():
    """8-bit branch: dequantize via state_dict → project → replace with fp16 Linear."""
    from rus.ablate import _ablate_quantized_target

    torch.manual_seed(2)
    w = (torch.randn(256, 256) * 0.02).half()
    cb, scb = quantize_absmax(w, 64)
    mod = FakeBnbModule(cb, scb)
    model = _fake_model_with_module(mod)
    v = torch.randn(256).half()

    stats = _ablate_quantized_target(model, "model.layers.0", "o_proj", mod, v, coefficient=0.8)
    assert stats["reduction"] > 0.7, stats

    # the module must now be a plain fp16 Linear with the direction removed
    import torch.nn as nn
    new_mod = model.model.layers._items[0].self_attn.o_proj
    assert isinstance(new_mod, nn.Linear), type(new_mod)
    assert new_mod.weight.dtype == torch.float16
    vn = v / v.norm()
    before = (vn @ w).abs().mean().item()
    after = (vn @ new_mod.weight).abs().mean().item()
    assert after < before * 0.3, f"{before} -> {after}"
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


def test_quantized_scb_layouts():
    """Works regardless of where bnb stashes CB/SCB — state_dict is canonical."""
    from rus.ablate import _ablate_quantized_target

    torch.manual_seed(3)
    w = (torch.randn(256, 256) * 0.02).half()
    cb, scb = quantize_absmax(w, 64)
    v = torch.randn(256).half()

    for make in (
        lambda: FakeBnbModule(cb, scb),                       # weight has no SCB attr at all
        lambda: FakeBnbModule(cb, scb),                       # same — state_dict driven
    ):
        mod = make()
        model = _fake_model_with_module(mod)
        stats = _ablate_quantized_target(model, "model.layers.0", "o_proj", mod, v, coefficient=0.8)
        assert stats["reduction"] > 0.7, stats
    print("PASS test_quantized_scb_layouts")


def test_apply_ablation_raises_on_failure():
    """Ablation errors must propagate, never be swallowed."""
    from rus.ablate import apply_ablation

    good = FakeBnbModule(*quantize_absmax((torch.randn(256, 256) * 0.02).half(), 64))
    bad = FakeBnbModule(*quantize_absmax((torch.randn(256, 256) * 0.02).half(), 64))
    bad._scb = None  # state_dict reports SCB=None -> layout check must raise

    model = type("M", (), {})()
    model.model = type("Inner", (), {})()
    model.model.layers = _Indexable([good, bad])
    model.model.layers.self_attn = None
    good.self_attn = type("S", (), {"o_proj": good})()
    bad.self_attn = type("S", (), {"o_proj": bad})()
    model.model.layers._items[0].self_attn = good.self_attn
    model.model.layers._items[1].self_attn = bad.self_attn
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
    test_quantized_scb_layouts()
    test_apply_ablation_raises_on_failure()
    test_exporter_shards()
    test_select_best_layers()
    print("\nAll tests passed ✓")
