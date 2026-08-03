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
import warnings
from typing import List, Dict, Tuple

from .loader import (
    load_model_and_tokenizer, discover_layers, get_device_report, release_memory,
)
from .collector import collect_pairwise_activations
from .subspace import (
    compute_refusal_directions, rank_layers, select_best_layers,
    build_consensus_direction,
)
from .ablate import apply_ablation
from .evaluator import evaluate_suite, compare_suites
from .exporter import export_model
from .tracker import log_run
from .prompts import (
    HARMFUL_PROMPTS, HARMLESS_PROMPTS,
    EVAL_HARMFUL_PROMPTS, EVAL_HARMLESS_PROMPTS,
)
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
        engine.analyze(num_prompts=48)
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
        trust_remote_code: bool = False,
        device_map="auto",
        max_memory: Dict = None,
        offload_folder: str = None,
    ):
        self.model_name = model_name
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.output_dir = output_dir or DEFAULT_OUTPUT_DIR
        self.trust_remote_code = trust_remote_code
        self.device_map = device_map
        self.max_memory = max_memory
        self.offload_folder = offload_folder

        self.model = None
        self.tokenizer = None
        self.layer_paths = []
        self.num_layers = 0
        self.directions = {}
        self.ranked_layers = []
        self.selected_layers = []
        self.source_layers = []
        self.ablation_strategy = ""
        self.ablation_stats = {}
        self.comparison_results = {}
        self.baseline_results = None
        self.export_path = ""
        self._coefficient = DEFAULT_COEFFICIENT
        self._coefficient_decay = DEFAULT_COEFFICIENT_DECAY
        self._protect_harmless = True
        self._started_at = time.time()

    def load(self) -> "RusEngine":
        """Load the model and tokenizer."""
        self.model, self.tokenizer = load_model_and_tokenizer(
            self.model_name,
            load_in_8bit=self.load_in_8bit,
            load_in_4bit=self.load_in_4bit,
            trust_remote_code=self.trust_remote_code,
            device_map=self.device_map,
            max_memory=self.max_memory,
            offload_folder=self.offload_folder,
        )
        self.layer_paths, self.num_layers = discover_layers(self.model)
        return self

    def device_report(self) -> Dict:
        """Return GPU memory and Accelerate module-placement diagnostics."""
        return get_device_report(self.model)

    def analyze(
        self, num_prompts: int = None, protect_harmless: bool = True
    ) -> "RusEngine":
        """
        Collect activations and compute the refusal subspace.
        Call after load().
        """
        if self.model is None:
            self.load()

        requested = DEFAULT_NUM_PROMPTS if num_prompts is None else num_prompts
        if requested < 3:
            raise ValueError("num_prompts must be at least 3")
        n = min(requested, len(HARMFUL_PROMPTS), len(HARMLESS_PROMPTS))

        harmful_acts, harmless_acts = collect_pairwise_activations(
            self.model, self.tokenizer,
            HARMFUL_PROMPTS[:n],
            HARMLESS_PROMPTS[:n],
        )

        self.directions = compute_refusal_directions(
            harmful_acts, harmless_acts, self.num_layers,
            protect_harmless=protect_harmless,
        )
        del harmful_acts, harmless_acts
        self._protect_harmless = protect_harmless
        self._analyzed_prompts = n

        self.ranked_layers = rank_layers(
            self.directions,
            LAYER_BLACKLIST_FIRST,
            LAYER_BLACKLIST_LAST,
        )
        release_memory()
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
        capture_baseline: bool = True,
        strategy: str = "global",
        preserve_norm: bool = True,
    ) -> "RusEngine":
        """
        Apply weight projection ablation to top-k layers.

        Args:
            k: Number of layers to ablate (default from config)
            coefficient: Steering strength (default from config)
            layers: Manual layer list (overrides auto-detection)
            capture_baseline: Cache held-out behavior before in-place modification
            strategy: ``global`` uses a consensus direction at every eligible
                destination layer; ``per_layer`` keeps the legacy top-k edit.
            preserve_norm: Restore each projected weight vector's original norm.
        """
        if self.model is None:
            self.load()
        if not self.directions:
            self.analyze()

        top_k = k if k is not None else TOP_K_LAYERS
        coeff = coefficient if coefficient is not None else DEFAULT_COEFFICIENT
        if top_k <= 0:
            raise ValueError("k must be positive")
        if not 0.0 < coeff <= 1.0:
            raise ValueError("coefficient must be in (0, 1]")

        if capture_baseline and self.baseline_results is None:
            print("Capturing held-out baseline before in-place ablation...")
            self.baseline_results = evaluate_suite(
                self.model, self.tokenizer,
                EVAL_HARMFUL_PROMPTS, EVAL_HARMLESS_PROMPTS,
            )
        self._coefficient = coeff

        if strategy not in {"global", "per_layer"}:
            raise ValueError("strategy must be 'global' or 'per_layer'")

        if layers:
            invalid = [l for l in layers if l not in self.directions]
            if invalid:
                raise ValueError(f"No analyzed direction for layers: {invalid}")
            source_candidates = [
                (l, self.directions[l]["score"], self.directions[l]["direction"])
                for l in layers
                if l in self.directions
            ]
        else:
            source_candidates = select_best_layers(self.ranked_layers, top_k)

        if not source_candidates:
            raise RuntimeError("No compatible layers were selected for ablation")

        self.source_layers = [item[0] for item in source_candidates]
        self.ablation_strategy = strategy
        if strategy == "global":
            consensus, self.source_layers = build_consensus_direction(
                source_candidates, len(source_candidates)
            )
            self.selected_layers = [
                (layer_idx, score, consensus)
                for layer_idx, score, _ in self.ranked_layers
            ]
            coefficient_decay = 1.0
        else:
            self.selected_layers = source_candidates
            coefficient_decay = DEFAULT_COEFFICIENT_DECAY
        self._coefficient_decay = coefficient_decay

        self.ablation_stats = apply_ablation(
            self.model,
            self.selected_layers,
            self.layer_paths,
            coefficient=coeff,
            coefficient_decay=coefficient_decay,
            preserve_norm=preserve_norm,
        )
        release_memory()
        return self

    def compare(self) -> Dict:
        """Compare against the held-out baseline captured before ablation."""
        if self.model is None:
            raise RuntimeError("No ablated model. Call load() + ablate() first.")

        if not self.ablation_stats:
            raise RuntimeError("No ablation has been applied. Call ablate() first.")
        if self.baseline_results is None:
            raise RuntimeError("Missing baseline evaluation; run ablate() again.")

        after = evaluate_suite(
            self.model, self.tokenizer,
            EVAL_HARMFUL_PROMPTS, EVAL_HARMLESS_PROMPTS,
        )
        self.comparison_results = compare_suites(self.baseline_results, after)

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
            method_metadata={
                "strategy": self.ablation_strategy,
                "source_layers": self.source_layers,
                "destination_layers": [item[0] for item in self.selected_layers],
                "protect_harmless": self._protect_harmless,
            },
        )

        try:
            layer_scores = {l: self.directions[l]["score"] for l in self.directions}
            log_run(
                model_name=self.model_name,
                num_prompts=getattr(self, "_analyzed_prompts", len(HARMFUL_PROMPTS)),
                top_k=len(self.source_layers),
                coefficient=self._coefficient,
                layers_modified=[s[0] for s in self.selected_layers],
                coefficients_used=[self._coefficient * (self._coefficient_decay ** r) for r in range(len(self.selected_layers))],
                comparison=self.comparison_results,
                export_path=self.export_path,
                duration_seconds=time.time() - self._started_at,
                layer_refusal_scores=layer_scores,
                ablation_stats=self.ablation_stats,
            )
        except Exception as exc:
            warnings.warn(f"Model exported, but run tracking failed: {exc}")

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
    trust_remote_code: bool = False,
    strategy: str = "global",
    preserve_norm: bool = True,
    protect_harmless: bool = True,
    device_map="auto",
    max_memory: Dict = None,
    offload_folder: str = None,
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
        trust_remote_code: Allow execution of custom code from the model repository
        strategy: Global consensus or legacy per-layer ablation
        preserve_norm: Restore projected weight-vector magnitudes
        protect_harmless: Orthogonalize estimates against harmless mean activations
        device_map: Accelerate placement strategy or explicit module map
        max_memory: Per-device memory limits such as ``{0: '12GiB', 1: '13GiB'}``
        offload_folder: Optional disk-offload directory

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
        trust_remote_code=trust_remote_code,
        device_map=device_map,
        max_memory=max_memory,
        offload_folder=offload_folder,
    )

    engine.analyze(num_prompts=num_prompts, protect_harmless=protect_harmless)
    engine.show_refusal()

    engine.ablate(
        k=k, coefficient=coefficient,
        capture_baseline=not skip_compare,
        strategy=strategy, preserve_norm=preserve_norm,
    )

    if not skip_compare:
        engine.compare()

    path = engine.save()
    print(f"\n✓ Model saved to: {path}")
    return path
