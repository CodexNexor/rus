"""
RUS Model Loader — downloads from HuggingFace, discovers architecture, returns hooks-ready model.
"""

import logging
import torch
import gc
from typing import Tuple, List, Dict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    BitsAndBytesConfig,
)

from .config import DEFAULT_DTYPE, DEFAULT_DEVICE_MAP, TRUST_REMOTE_CODE

# Stop bitsandbytes flooding output with "inputs will be cast from bf16 to fp16"
# style warnings. With fp16 compute dtype (set below) these are unnecessary.
logging.getLogger("bitsandbytes").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)


def load_model_and_tokenizer(
    model_name: str,
    dtype: torch.dtype = DEFAULT_DTYPE,
    device_map: str = DEFAULT_DEVICE_MAP,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Download (if needed) and load a HuggingFace causal LM + its tokenizer.
    Returns (model, tokenizer) ready for activation collection.
    """
    gc.collect()
    torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=TRUST_REMOTE_CODE,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {
        "device_map": device_map,
        "trust_remote_code": TRUST_REMOTE_CODE,
    }

    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        # fp16 compute dtype stops bnb casting bf16 inputs on every forward
        kwargs["torch_dtype"] = torch.float16
    elif load_in_8bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16,
        )
        # MUST pass torch_dtype=float16: without it transformers uses the
        # config's bf16 and bitsandbytes re-casts to fp16 on every matmul,
        # flooding logs and wasting time on unsupported T4 bf16 paths.
        kwargs["torch_dtype"] = torch.float16
    else:
        kwargs["torch_dtype"] = dtype

    kwargs["output_hidden_states"] = False

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()

    return model, tokenizer


def discover_layers(model: PreTrainedModel) -> Tuple[List[str], int]:
    """
    Auto-discover the transformer block layers and their weight targets.
    Returns (layer_paths, num_layers) where layer_paths are dotted paths like
    'model.layers.0', 'model.layers.1', etc.
    """
    candidates = [
        "model.layers",
        "transformer.h",
        "model.decoder.layers",
        "transformer.blocks",
        "gpt_neox.layers",
        "model.model.layers",
    ]

    for prefix in candidates:
        try:
            obj = model
            for part in prefix.split("."):
                obj = getattr(obj, part)
            num_layers = len(obj)
            return [f"{prefix}.{i}" for i in range(num_layers)], num_layers
        except (AttributeError, TypeError, IndexError):
            continue

    for name, module in model.named_modules():
        if name.endswith(".0") and hasattr(module, "self_attn"):
            parent = ".".join(name.split(".")[:-1])
            count = 0
            while True:
                try:
                    obj = model
                    for part in f"{parent}.{count}".split("."):
                        obj = getattr(obj, part)
                    count += 1
                except (AttributeError, IndexError):
                    break
            return [f"{parent}.{i}" for i in range(count)], count

    raise RuntimeError(
        "Could not auto-discover transformer layers. "
        "Please report the model architecture."
    )


def get_weight_targets(model: PreTrainedModel, layer_path: str) -> Dict[str, torch.Tensor]:
    """
    For a given layer path (e.g. 'model.layers.5'), return the weight matrices
    that should be projected away from the refusal direction.

    Priority targets: self_attn.o_proj.weight, mlp.down_proj.weight
    """
    targets = {}
    layer_module = model
    for part in layer_path.split("."):
        layer_module = getattr(layer_module, part)

    weight_candidates = [
        ("self_attn.o_proj.weight", "o_proj"),
        ("self_attn.o_proj", "o_proj"),
        ("attn.c_proj.weight", "c_proj"),
        ("attn.c_proj", "c_proj"),
        ("attention.wo.weight", "wo"),
        ("mlp.down_proj.weight", "down_proj"),
        ("mlp.down_proj", "down_proj"),
        ("mlp.c_proj.weight", "c_proj_mlp"),
        ("mlp.c_proj", "c_proj_mlp"),
        ("mlp.fc2.weight", "fc2"),
        ("mlp.fc2", "fc2"),
    ]

    for attr_path, tag in weight_candidates:
        try:
            obj = layer_module
            for part in attr_path.split("."):
                obj = getattr(obj, part)
            if isinstance(obj, torch.Tensor) or hasattr(obj, "weight"):
                w = obj.weight if hasattr(obj, "weight") else obj
                if w.dim() == 2 and w.shape[0] >= 64 and w.shape[1] >= 64:
                    targets[tag] = w
        except (AttributeError, TypeError):
            continue

    return targets


def get_hidden_size(model: PreTrainedModel) -> int:
    """Extract the model's hidden dimension."""
    if hasattr(model.config, "hidden_size"):
        return model.config.hidden_size
    if hasattr(model.config, "d_model"):
        return model.config.d_model
    if hasattr(model.config, "n_embd"):
        return model.config.n_embd
    raise RuntimeError("Cannot determine model hidden size from config.")


def get_device(model: PreTrainedModel) -> torch.device:
    """Find which device the model is on."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")
