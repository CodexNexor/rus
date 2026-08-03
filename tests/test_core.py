"""
Unit tests for RUS core math: 8-bit quant round-trip, direction projection,
and the sharded exporter. Run: python tests/test_core.py
"""

import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from rus.ablate import (
    dequantize_8bit,
    project_direction_from_weight,
    _block_size,
)
from rus.exporter import _save_shards


class _Int8Param(torch.Tensor):
    """Mimics bnb's Int8Params: a Tensor whose .SCB attr holds the scales."""

    @staticmethod
    def __new__(cls, cb, scb):
        obj = torch.Tensor._make_subclass(cls, cb)
        obj.SCB = scb
        return obj


class FakeBnbModule:
    """Mimics Linear8bitLt, in two real-world layouts:

    layout="state_dict" — state_dict exposes the "weight.SCB" key (newer bnb)
    layout="param_attr" — state_dict exposes only "weight"; SCB sits on the
                          weight param's .SCB attribute (transformers wrapper)
    """

    def __init__(self, cb, scb, layout="state_dict"):
        self._cb = cb
        self._scb = scb
        self._layout = layout
        self.bias = None
        if layout == "param_attr":
            self._weight_param = _Int8Param(cb, scb)

    def state_dict(self):
        if self._layout == "param_attr":
            return {"weight": self._weight_param}
        return {"weight": self._cb, "weight.SCB": self._scb}

    @property
    def weight(self):
        if self._layout == "param_attr":
            return self._weight_param
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


def test_transposed_projection():
    """Transformers Conv1D stores weights as (input, output)."""
    torch.manual_seed(11)
    w = torch.randn(512, 256)
    v = torch.randn(256)
    wm = project_direction_from_weight(w, v, coefficient=1.0, output_axis=1)
    assert (wm @ (v / v.norm())).abs().mean() < 1e-5
    print("PASS test_transposed_projection")


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


def test_exporter_shards():
    with tempfile.TemporaryDirectory() as tmp:
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
            _save_shards(deduped, tmp)
        files = sorted(os.listdir(tmp))
        assert any(f.endswith(".safetensors") for f in files), files
        if len(files) > 1:
            assert "model.safetensors.index.json" in files
        import safetensors.torch as st
        keys = []
        for f in files:
            if f.endswith(".safetensors"):
                keys += list(st.load_file(os.path.join(tmp, f)).keys())
        assert sorted(keys) == sorted(deduped.keys()), (sorted(keys), sorted(deduped))
    print("PASS test_exporter_shards")


def test_select_best_layers():
    """Low absolute scores (normal with many prompts) must still yield top-k."""
    from rus.subspace import select_best_layers

    ranked = [(2, 0.28, torch.randn(8)), (3, 0.23, torch.randn(8)), (4, 0.19, torch.randn(8))]
    selected = select_best_layers(ranked, top_k=2, min_score=0.3)
    assert [l for l, _, _ in selected] == [2, 3], "rank-based fallback failed"
    assert select_best_layers([], top_k=2) == []
    print("PASS test_select_best_layers")


def test_mean_difference_direction():
    from rus.subspace import compute_refusal_directions
    torch.manual_seed(12)
    harmless = torch.randn(32, 16) * 0.2
    harmful = harmless + torch.tensor([2.0] + [0.0] * 15)
    result = compute_refusal_directions({0: [harmful]}, {0: [harmless]}, 1)[0]
    expected = torch.tensor([1.0] + [0.0] * 15)
    assert torch.dot(result["direction"], expected) > 0.99
    assert 0.0 <= result["score"] <= 1.0
    print("PASS test_mean_difference_direction")


def test_refusal_detector_avoids_generic_safety_words():
    from rus.evaluator import detect_refusal
    assert not detect_refusal("This chemical is dangerous, so wear gloves and goggles.")
    assert detect_refusal("I cannot assist with that request.")
    print("PASS test_refusal_detector_avoids_generic_safety_words")


def test_quantized_scb_layouts():
    """Works regardless of where bnb stashes CB/SCB (state_dict key or param attr)."""
    from rus.ablate import _ablate_quantized_target

    torch.manual_seed(3)
    w = (torch.randn(256, 256) * 0.02).half()
    cb, scb = quantize_absmax(w, 64)
    v = torch.randn(256).half()

    for layout in ("state_dict", "param_attr"):
        mod = FakeBnbModule(cb, scb, layout=layout)
        model = _fake_model_with_module(mod)
        stats = _ablate_quantized_target(model, "model.layers.0", "o_proj", mod, v, coefficient=0.8)
        assert stats["reduction"] > 0.7, stats
        assert isinstance(model.model.layers._items[0].self_attn.o_proj, torch.nn.Linear)
    print("PASS test_quantized_scb_layouts")


def test_quantized_multi_target_routing():
    """o_proj and down_proj must each use their own quantized module."""
    torch.manual_seed(4)
    w1 = (torch.randn(256, 256) * 0.02).half()
    w2 = (torch.randn(256, 512) * 0.02).half()
    o_proj = FakeBnbModule(*quantize_absmax(w1, 64))
    down_proj = FakeBnbModule(*quantize_absmax(w2, 64))
    model = _fake_model_with_module(o_proj)
    model.model.layers._items[0].mlp.down_proj = down_proj
    from rus.ablate import apply_ablation_to_layer
    stats = apply_ablation_to_layer(model, "model.layers.0", torch.randn(256), 0.4)
    layer = model.model.layers._items[0]
    assert isinstance(layer.self_attn.o_proj, torch.nn.Linear)
    assert isinstance(layer.mlp.down_proj, torch.nn.Linear)
    assert set(stats) == {"o_proj", "down_proj"}
    print("PASS test_quantized_multi_target_routing")


def test_4bit_ablation_path():
    """NF4 path uses quant_state and replaces the target with fp16 Linear."""
    torch.manual_seed(5)
    fp = (torch.randn(256, 256) * 0.02).half()

    class Fake4Bit:
        bias = None
        def __init__(self):
            self.weight = torch.zeros_like(fp, dtype=torch.uint8)
            self.weight.quant_state = object()
        def state_dict(self):
            return {"weight": self.weight}

    functional = types.ModuleType("bitsandbytes.functional")
    functional.dequantize_4bit = lambda data, quant_state: fp.clone()
    package = types.ModuleType("bitsandbytes")
    package.functional = functional
    old_package = sys.modules.get("bitsandbytes")
    old_functional = sys.modules.get("bitsandbytes.functional")
    sys.modules["bitsandbytes"] = package
    sys.modules["bitsandbytes.functional"] = functional
    try:
        model = _fake_model_with_module(Fake4Bit())
        from rus.ablate import apply_ablation_to_layer
        stats = apply_ablation_to_layer(model, "model.layers.0", torch.randn(256), 0.8)
        assert stats["o_proj"]["quantization_bits"] == 4
        assert isinstance(model.model.layers._items[0].self_attn.o_proj, torch.nn.Linear)
    finally:
        if old_package is None:
            sys.modules.pop("bitsandbytes", None)
        else:
            sys.modules["bitsandbytes"] = old_package
        if old_functional is None:
            sys.modules.pop("bitsandbytes.functional", None)
        else:
            sys.modules["bitsandbytes.functional"] = old_functional
    print("PASS test_4bit_ablation_path")


def test_tracker_upsert():
    import rus.tracker as tracker
    old_path = tracker.DB_PATH
    with tempfile.TemporaryDirectory() as tmp:
        tracker.DB_PATH = os.path.join(tmp, "experience.db")
        try:
            for _ in range(2):
                tracker.log_run(
                    "Qwen/test", 8, 1, 0.8, [2], [0.8], {}, None, 1.0,
                    {2: 0.7},
                    {2: {"coefficient": 0.8, "targets": {"o_proj": {"reduction": 0.8}}}},
                )
            insight = tracker.get_insights("qwen")
            assert insight["num_runs"] == 2, insight
        finally:
            tracker.DB_PATH = old_path
    print("PASS test_tracker_upsert")


def test_apply_ablation_raises_on_failure():
    """Ablation errors must propagate, never be swallowed."""
    from rus.ablate import apply_ablation

    good = FakeBnbModule(*quantize_absmax((torch.randn(256, 256) * 0.02).half(), 64))
    bad_cb, _ = quantize_absmax((torch.randn(256, 256) * 0.02).half(), 64)
    bad = FakeBnbModule(bad_cb, torch.randn(10, 10))  # SCB garbage -> block inference must raise

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
    test_transposed_projection()
    test_quantized_ablation_path()
    test_quantized_scb_layouts()
    test_quantized_multi_target_routing()
    test_4bit_ablation_path()
    test_apply_ablation_raises_on_failure()
    test_exporter_shards()
    test_select_best_layers()
    test_mean_difference_direction()
    test_refusal_detector_avoids_generic_safety_words()
    test_tracker_upsert()
    print("\nAll tests passed ✓")
