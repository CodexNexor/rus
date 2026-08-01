"""
RUS Exporter — saves the abliterated model and tokenizer to disk.
"""

import os
import json
from datetime import datetime


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
        model: The ablated model
        tokenizer: The tokenizer
        output_dir: Base output directory
        model_name: Original model name/ID
        ablation_stats: Stats from the ablation process
        comparison_results: Before/after comparison metrics
    """
    safe_name = model_name.replace("/", "_").replace(":", "_")
    export_path = os.path.join(output_dir, f"{safe_name}_rus")

    os.makedirs(export_path, exist_ok=True)

    model.save_pretrained(
        export_path,
        safe_serialization=True,
        max_shard_size="5GB",
    )

    tokenizer.save_pretrained(export_path)

    metadata = {
        "tool": "RUS — Remove Ur Refusal",
        "version": "1.0.0",
        "original_model": model_name,
        "exported_at": datetime.now().isoformat(),
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
