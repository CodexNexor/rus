"""
RUS Exporter — saves the abliterated model and tokenizer to disk.

Uses Transformers' quantizer-aware ``save_pretrained`` implementation. Manual
serialization of bitsandbytes internals produces checkpoints whose config and
weight representation can disagree on reload.
"""

import os
import json
from datetime import datetime

import torch

from .ablate import TARGET_SUFFIXES

try:
    import safetensors.torch as st
    _HAS_SAFETENSORS = True
except ImportError:
    st = None
    _HAS_SAFETENSORS = False

MAX_SHARD_SIZE = 5 * 1024 ** 3  # 5GB per shard, same as transformers default


def _configure_quantization_skip_modules(model, ablation_stats: dict) -> list:
    """Keep fp16 replacement modules unquantized when a mixed model reloads.

    bitsandbytes replaces ordinary Linear modules during ``from_pretrained``.
    Ablated quantized targets were intentionally replaced by fp16 Linear, so
    they must be named in the serialized quantizer skip list or they reload as
    incomplete Linear8bitLt modules without CB/SCB state.
    """
    module_names = []
    for layer in ablation_stats.values():
        layer_path = layer.get("layer_path")
        if not layer_path:
            continue
        for tag, target_stats in layer.get("targets", {}).items():
            if not isinstance(target_stats, dict) or not target_stats.get("quantized"):
                continue
            suffix = TARGET_SUFFIXES.get(tag)
            if suffix:
                module_names.append(f"{layer_path}.{suffix}")

    quant_config = getattr(getattr(model, "config", None), "quantization_config", None)
    if not module_names or quant_config is None:
        return sorted(set(module_names))

    if isinstance(quant_config, dict):
        existing = quant_config.get("llm_int8_skip_modules") or []
        quant_config["llm_int8_skip_modules"] = sorted(set(existing) | set(module_names))
    else:
        existing = getattr(quant_config, "llm_int8_skip_modules", None) or []
        quant_config.llm_int8_skip_modules = sorted(set(existing) | set(module_names))
    return sorted(set(module_names))


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
    method_metadata: dict = None,
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

    if not ablation_stats:
        raise RuntimeError("Refusing to export a model with no recorded ablation")

    quantization_skip_modules = _configure_quantization_skip_modules(
        model, ablation_stats
    )

    # Transformers owns the serialization contract for quantized checkpoints.
    # Modern bitsandbytes checkpoints include quantization state that a raw
    # state_dict writer cannot safely reconstruct.
    model.save_pretrained(
        export_path,
        safe_serialization=True,
        max_shard_size="5GB",
    )
    tokenizer.save_pretrained(export_path)

    metadata = {
        "tool": "RUS — Remove Ur Refusal",
        "version": "1.3.0",
        "original_model": model_name,
        "exported_at": datetime.now().isoformat(),
        "quantized": any(
            p.dtype in (torch.int8, torch.uint8) for p in model.parameters()
        ) if hasattr(model, "parameters") else False,
        "method": method_metadata or {},
        "quantization_skip_modules": quantization_skip_modules,
        "ablation_stats": {
            str(k): {
                "layer_path": v.get("layer_path", ""),
                "coefficient": v.get("coefficient", 0),
                "refusal_score": v.get("refusal_score", 0),
                "preserve_norm": v.get("preserve_norm", False),
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
            "harmless_kl_divergence": comparison_results.get("harmless_kl_divergence"),
        },
    }

    metadata_path = os.path.join(export_path, "rus_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    card_path = os.path.join(export_path, "README.md")
    with open(card_path, "w") as f:
        f.write(
            f"# {safe_name} (RUS)\n\n"
            f"Derived from `{model_name}` with RUS refusal-direction ablation.\n\n"
            "## Important use notice\n\n"
            "This transformation can weaken model safeguards and does not make "
            "outputs accurate, lawful, or safe. Evaluate the checkpoint in an "
            "isolated environment before deployment. See `rus_metadata.json` "
            "for transformation and evaluation details.\n"
        )

    return export_path
