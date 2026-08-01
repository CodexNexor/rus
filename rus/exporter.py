"""
RUS Exporter — saves the abliterated model and tokenizer to disk.

Saves via the model's state_dict (sharded safetensors + index) instead of
save_pretrained. This is required because save_pretrained refuses 8-bit
quantized (bitsandbytes) models, and it works identically for fp16 models.
The resulting checkpoint is a standard transformers multi-shard safetensors
folder loadable with from_pretrained(..., load_in_8bit=True/False).
"""

import os
import json
from datetime import datetime

import torch

try:
    import safetensors.torch as st
    _HAS_SAFETENSORS = True
except ImportError:
    st = None
    _HAS_SAFETENSORS = False

MAX_SHARD_SIZE = 5 * 1024 ** 3  # 5GB per shard, same as transformers default


def _dedupe_tied(state_dict: dict) -> dict:
    """
    Remove duplicate keys that share the same underlying tensor (tied weights,
    e.g. embed_tokens <-> lm_head). Safetensors refuses shared storage.
    Transformers re-ties them on load via config.tie_word_embeddings.
    """
    deduped, seen = {}, {}
    for key, tensor in state_dict.items():
        storage_id = (tensor.data_ptr(), tensor.numel(), tensor.element_size())
        if storage_id in seen:
            continue
        seen[storage_id] = key
        deduped[key] = tensor
    return deduped


def _shard_state_dict(state_dict: dict) -> list:
    """Split a state dict into ~5GB shards. Returns list of {key: tensor}."""
    shards, current, current_size = [], {}, 0
    for key, tensor in state_dict.items():
        size = tensor.numel() * tensor.element_size()
        if current and current_size + size > MAX_SHARD_SIZE:
            shards.append(current)
            current, current_size = {}, 0
        current[key] = tensor
        current_size += size
    if current:
        shards.append(current)
    return shards


def _save_shards(state_dict: dict, export_path: str) -> dict:
    """Write sharded safetensors files + index. Returns {key: filename} map."""
    shards = _shard_state_dict(state_dict)
    weight_map = {}

    if len(shards) == 1 and _HAS_SAFETENSORS:
        fname = "model.safetensors"
        st.save_file(shards[0], os.path.join(export_path, fname), metadata={"format": "pt"})
        for key in shards[0]:
            weight_map[key] = fname
        return weight_map

    if _HAS_SAFETENSORS:
        for i, shard in enumerate(shards):
            fname = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
            st.save_file(shard, os.path.join(export_path, fname), metadata={"format": "pt"})
            for key in shard:
                weight_map[key] = fname
    else:
        fname = "pytorch_model.bin"
        torch_dict = {key: v for shard in shards for key, v in shard.items()}
        torch.save(torch_dict, os.path.join(export_path, fname))
        for key in torch_dict:
            weight_map[key] = fname

    total_size = sum(v.numel() * v.element_size() for v in state_dict.values())
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    with open(os.path.join(export_path, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)

    return weight_map


def export_model(
    model,
    tokenizer,
    output_dir: str,
    model_name: str,
    ablation_stats: dict,
    comparison_results: dict,
):
    """
    Save the modified model, tokenizer, and metadata to disk.

    Args:
        model: The ablated model (fp16 or bitsandbytes-quantized)
        tokenizer: The tokenizer
        output_dir: Base output directory
        model_name: Original model name/ID
        ablation_stats: Stats from the ablation process
        comparison_results: Before/after comparison metrics
    """
    safe_name = model_name.replace("/", "_").replace(":", "_")
    export_path = os.path.join(output_dir, f"{safe_name}_rus")

    os.makedirs(export_path, exist_ok=True)

    model.config.save_pretrained(export_path)
    tokenizer.save_pretrained(export_path)

    # state_dict() returns CPU copies — safe for both fp16 and 8-bit models
    _save_shards(_dedupe_tied(model.state_dict()), export_path)

    metadata = {
        "tool": "RUS — Remove Ur Refusal",
        "version": "1.0.5",
        "original_model": model_name,
        "exported_at": datetime.now().isoformat(),
        "quantized": any(
            p.dtype in (torch.int8, torch.uint8) for p in model.parameters()
        ) if hasattr(model, "parameters") else False,
        "ablation_stats": {
            str(k): {
                "layer_path": v.get("layer_path", ""),
                "coefficient": v.get("coefficient", 0),
                "refusal_score": v.get("refusal_score", 0),
                "targets": {
                    tag: {
                        "projection_before": tv.get("projection_before", 0),
                        "projection_after": tv.get("projection_after", 0),
                        "reduction": tv.get("reduction", 0),
                        "quantized": tv.get("quantized", False),
                    }
                    for tag, tv in v.get("targets", {}).items()
                },
            }
            for k, v in ablation_stats.items()
        },
        "comparison": {
            "refusal_rate_before": comparison_results.get("refusal_rate_before", 0),
            "refusal_rate_after": comparison_results.get("refusal_rate_after", 0),
            "compliance_before": comparison_results.get("compliance_before", 0),
            "compliance_after": comparison_results.get("compliance_after", 0),
            "quality_before": comparison_results.get("quality_before", 0),
            "quality_after": comparison_results.get("quality_after", 0),
        },
    }

    metadata_path = os.path.join(export_path, "rus_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return export_path
