"""
RUS High-Level API — one-line model ablation.

Usage:
    import rus
    rus.ablate("Qwen/Qwen2.5-7B-Instruct", load_in_8bit=True)

Or step-by-step:
    engine = rus.RusEngine("Qwen/Qwen2.5-7B-Instruct", load_in_8bit=True)
    engine.analyze()
    engine.show_refusal()
    engine.ablate(k=5, coefficient=0.8)
    engine.compare()
    engine.save()
"""

import time
import copy
import torch
from typing import List, Optional, Dict, Tuple

from .loader import load_model_and_tokenizer, discover_layers, get_device
from .collector import collect_pairwise_activations
from .subspace import compute_refusal_directions, rank_layers, select_best_layers
from .ablate import apply_ablation
from .evaluator import run_comparison
from .exporter import export_model
from .tracker import log_run
from .prompts import HARMFUL_PROMPTS, HARMLESS_PROMPTS
from .config import (
    DEFAULT_NUM_PROMPTS,
    DEFAULT_COEFFICIENT,
    DEFAULT_COEFFICIENT_DECAY,
    DEFAULT_OUTPUT_DIR,
    TOP_K_LAYERS,
    LAYER_BLACKLIST_FIRST,
    LAYER_BLACKLIST_LAST,
)


class RusEngine:
    """
    RUS Ablation Engine — high-level interface.

    Example:
        engine = RusEngine("Qwen/Qwen2.5-7B-Instruct", load_in_8bit=True)
        engine.analyze(num_prompts=64)
        engine.show_refusal()
        engine.ablate(k=5)
        engine.compare()
        path = engine.save()
        print(f"Model saved to {path}")
    """

    def __init__(
        self,
        model_name: str,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        output_dir: str = None,
    ):
        self.model_name = model_name
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.output_dir = output_dir or DEFAULT_OUTPUT_DIR

        self.model = None
        self.tokenizer = None
        self.model_before = None
        self.layer_paths = []
        self.num_layers = 0
        self.directions = {}
        self.ranked_layers = []
        self.selected_layers = []
        self.ablation_stats = {}
        self.comparison_results = {}
        self.export_path = ""

    def load(self) -> "RusEngine":
        """Load the model and tokenizer."""
        self.model, self.tokenizer = load_model_and_tokenizer(
            self.model_name,
            load_in_8bit=self.load_in_8bit,
            load_in_4bit=self.load_in_4bit,
        )
        self.layer_paths, self.num_layers = discover_layers(self.model)
        return self

    def analyze(self, num_prompts: int = None) -> "RusEngine":
        """
        Collect activations and compute the refusal subspace.
        Call after load().
        """
        if self.model is None:
            self.load()

        n = num_prompts or min(DEFAULT_NUM_PROMPTS, len(HARMFUL_PROMPTS), len(HARMLESS_PROMPTS))

        harmful_acts, harmless_acts = collect_pairwise_activations(
            self.model, self.tokenizer,
            HARMFUL_PROMPTS[:n],
            HARMLESS_PROMPTS[:n],
        )

        self.directions = compute_refusal_directions(
            harmful_acts, harmless_acts, self.num_layers
        )

        self.ranked_layers = rank_layers(
            self.directions,
            LAYER_BLACKLIST_FIRST,
            LAYER_BLACKLIST_LAST,
        )
        return self

    def show_refusal(self, top_n: int = 16) -> List[Tuple[int, float]]:
        """Print per-layer refusal scores. Returns top layers."""
        from .cli import build_layer_table
        from rich.console import Console
        console = Console()

        selected = select_best_layers(self.ranked_layers, TOP_K_LAYERS)
        table = build_layer_table(self.directions, selected, top_n=top_n)
        console.print(table)
        return [(l, s) for l, s, _ in selected]

    def ablate(
        self,
        k: int = None,
        coefficient: float = None,
        layers: List[int] = None,
    ) -> "RusEngine":
        """
        Apply weight projection ablation to top-k layers.

        Args:
            k: Number of layers to ablate (default from config)
            coefficient: Steering strength (default from config)
            layers: Manual layer list (overrides auto-detection)
        """
        if self.model is None:
            self.load()
        if not self.directions:
            self.analyze()

        top_k = k or TOP_K_LAYERS
        coeff = coefficient or DEFAULT_COEFFICIENT

        if layers:
            self.selected_layers = [
                (l, self.directions[l]["score"], self.directions[l]["direction"])
                for l in layers
                if l in self.directions
            ]
        else:
            self.selected_layers = select_best_layers(self.ranked_layers, top_k)

        self.ablation_stats = apply_ablation(
            self.model,
            self.selected_layers,
            self.layer_paths,
            coefficient=coeff,
            coefficient_decay=DEFAULT_COEFFICIENT_DECAY,
        )
        return self

    def compare(self) -> Dict:
        """Run before/after comparison. Loads fresh copy of original model."""
        if self.model is None:
            raise RuntimeError("No ablated model. Call load() + ablate() first.")

        print("Loading original model for comparison...")
        self.model_before, _ = load_model_and_tokenizer(
            self.model_name,
            load_in_8bit=self.load_in_8bit,
            load_in_4bit=self.load_in_4bit,
        )

        self.comparison_results = run_comparison(
            self.model_before, self.model, self.tokenizer,
            HARMFUL_PROMPTS, HARMLESS_PROMPTS,
        )

        del self.model_before
        self.model_before = None
        torch.cuda.empty_cache()

        from .cli import build_comparison_table, build_sample_comparison
        from rich.console import Console
        console = Console()
        console.print(build_comparison_table(self.comparison_results))
        console.print(build_sample_comparison(self.comparison_results))

        return self.comparison_results

    def save(self, output_dir: str = None) -> str:
        """Export the abliterated model to disk. Returns export path."""
        out = output_dir or self.output_dir
        self.export_path = export_model(
            self.model, self.tokenizer, out, self.model_name,
            self.ablation_stats, self.comparison_results,
        )

        try:
            layer_scores = {l: self.directions[l]["score"] for l in self.directions}
            log_run(
                model_name=self.model_name,
                num_prompts=len(HARMFUL_PROMPTS),
                top_k=len(self.selected_layers),
                coefficient=DEFAULT_COEFFICIENT,
                layers_modified=[s[0] for s in self.selected_layers],
                coefficients_used=[DEFAULT_COEFFICIENT * (DEFAULT_COEFFICIENT_DECAY ** r) for r in range(len(self.selected_layers))],
                comparison=self.comparison_results,
                export_path=self.export_path,
                duration_seconds=0,
                layer_refusal_scores=layer_scores,
            )
        except Exception:
            pass

        return self.export_path

    def test(self, prompts: List[str] = None) -> List[Dict]:
        """Test the ablated model on a list of prompts. Returns results."""
        if self.model is None:
            raise RuntimeError("No model loaded.")

        from .evaluator import detect_refusal, generate_response

        if prompts is None:
            prompts = HARMFUL_PROMPTS[:8] + HARMLESS_PROMPTS[:4]

        results = []
        for p in prompts:
            resp = generate_response(self.model, self.tokenizer, p)
            refused = detect_refusal(resp)
            results.append({
                "prompt": p,
                "response": resp[:300],
                "refused": refused,
            })
        return results


def ablate(
    model_name: str,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    num_prompts: int = None,
    k: int = None,
    coefficient: float = None,
    output_dir: str = None,
    skip_compare: bool = False,
) -> str:
    """
    One-shot: load, analyze, ablate, compare, save.

    Args:
        model_name: HuggingFace model ID
        load_in_8bit: Use 8-bit quantization for low VRAM
        load_in_4bit: Use 4-bit quantization for very low VRAM
        num_prompts: Number of prompts for subspace analysis
        k: Number of layers to ablate
        coefficient: Steering strength
        output_dir: Where to save
        skip_compare: Skip before/after comparison

    Returns:
        Path to the saved abliterated model.

    Example:
        >>> path = rus.ablate("Qwen/Qwen2.5-7B-Instruct", load_in_8bit=True)
        >>> print(path)
    """
    engine = RusEngine(
        model_name=model_name,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
        output_dir=output_dir,
    )

    engine.analyze(num_prompts=num_prompts)
    engine.show_refusal()

    engine.ablate(k=k, coefficient=coefficient)

    if not skip_compare:
        engine.compare()

    path = engine.save()
    print(f"\n✓ Model saved to: {path}")
    return path
