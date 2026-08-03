"""
RUS CLI — Beautiful terminal interface for the Remove Ur Refusal engine.
"""

import sys
import time
import argparse
import warnings
from typing import List

import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
)
from rich import box

from .config import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_NUM_PROMPTS,
    DEFAULT_COEFFICIENT,
    DEFAULT_COEFFICIENT_DECAY,
    TOP_K_LAYERS,
    LAYER_BLACKLIST_FIRST,
    LAYER_BLACKLIST_LAST,
    EVAL_HARMFUL_PROMPTS,
)
from .prompts import (
    HARMFUL_PROMPTS, HARMLESS_PROMPTS,
    EVAL_HARMFUL_PROMPTS as HELDOUT_HARMFUL_PROMPTS,
    EVAL_HARMLESS_PROMPTS as HELDOUT_HARMLESS_PROMPTS,
)
from .loader import load_model_and_tokenizer, discover_layers, get_device, get_hidden_size
from .collector import collect_pairwise_activations
from .subspace import (
    compute_refusal_directions, rank_layers, select_best_layers,
    build_consensus_direction,
)
from .ablate import apply_ablation
from .evaluator import evaluate_suite, compare_suites
from .exporter import export_model
from .tracker import log_run, get_insights

console = Console()


LOGO = r"""[bold bright_magenta]
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   ██████╗  ██╗   ██╗ ███████╗                                ║
║   ██╔══██╗ ██║   ██║ ██╔════╝                                ║
║   ██████╔╝ ██║   ██║ ███████╗                                ║
║   ██╔══██╗ ██║   ██║ ╚════██║                                ║
║   ██║  ██║ ╚██████╔╝ ███████║                                ║
║   ╚═╝  ╚═╝  ╚═════╝  ╚══════╝                                ║
║                                                              ║
║   Remove Ur Refusal — Living Ablation Engine v1.2.1           ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
[/bold bright_magenta]"""


def splash():
    """Print the RUS logo."""
    console.print(LOGO)
    console.print()


def build_layer_table(
    directions: dict,
    selected: List[tuple],
    top_n: int = 20,
) -> Table:
    """Build a Rich table showing per-layer refusal analysis."""
    table = Table(
        title="\n[bold]Refusal Subspace Analysis[/bold]",
        box=box.ROUNDED,
        border_style="bright_magenta",
        header_style="bold white",
    )
    table.add_column("Layer", justify="right", style="cyan")
    table.add_column("Refusal Score", justify="right", style="yellow")
    table.add_column("Bar", justify="left")
    table.add_column("Status", justify="center")

    selected_layers = {s[0] for s in selected}

    total_layers = max(directions.keys()) + 1 if directions else 0
    skip_first = LAYER_BLACKLIST_FIRST
    skip_last = LAYER_BLACKLIST_LAST

    eligible = [
        (layer_idx, info)
        for layer_idx, info in directions.items()
        if skip_first <= layer_idx < total_layers - skip_last
    ]
    eligible.sort(key=lambda item: item[1]["score"], reverse=True)

    for layer_idx, info in eligible[:top_n]:
        score = info["score"]
        bar_len = int(score * 20)
        bar = "[bright_green]" + "█" * bar_len + "[/]" + "░" * (20 - bar_len)

        if layer_idx in selected_layers:
            status = "[bold bright_yellow]★ SELECTED[/]"
        elif score >= 0.5:
            status = "[dim]candidate[/]"
        else:
            status = "[dim]low signal[/]"

        table.add_row(
            str(layer_idx),
            f"{score:.4f}",
            bar,
            status,
        )

    return table


def build_comparison_table(results: dict) -> Table:
    """Build before/after comparison table."""
    table = Table(
        title="\n[bold]Before vs After Ablation[/bold]",
        box=box.ROUNDED,
        border_style="bright_magenta",
        header_style="bold white",
    )
    table.add_column("Metric", style="cyan")
    table.add_column("BEFORE", justify="center", style="red")
    table.add_column("AFTER", justify="center", style="green")
    table.add_column("Delta", justify="center")

    r_before = results.get("refusal_rate_before", 0)
    r_after = results.get("refusal_rate_after", 0)
    c_before = results.get("compliance_before", 0)
    c_after = results.get("compliance_after", 0)
    q_before = results.get("quality_before", 0)
    q_after = results.get("quality_after", 0)

    table.add_row(
        "Refusal Rate",
        f"{r_before:.1%}",
        f"{r_after:.1%}",
        f"[{'green' if r_after < r_before else 'red'}]{r_after - r_before:+.1%}[/]",
    )
    table.add_row(
        "Compliance Rate",
        f"{c_before:.1%}",
        f"{c_after:.1%}",
        f"[{'green' if c_after > c_before else 'red'}]{c_after - c_before:+.1%}[/]",
    )
    table.add_row(
        "Quality Score",
        f"{q_before:.2f}",
        f"{q_after:.2f}",
        f"[{'green' if q_after >= q_before * 0.9 else 'yellow'}]{q_after - q_before:+.2f}[/]",
    )

    refusal_reduction = results.get("refusal_reduction", 0)
    table.add_row(
        "Refusal ↓",
        "",
        "",
        f"[bold green]{refusal_reduction:+.1%}[/]",
    )
    kl = results.get("harmless_kl_divergence")
    if kl is not None:
        table.add_row(
            "Harmless KL drift",
            "0.0000",
            f"{kl:.4f}",
            "[green]low[/]" if kl < 0.1 else "[yellow]review[/]",
        )

    return table


def build_sample_comparison(results: dict) -> Panel:
    """Build a sample output comparison panel."""
    samples_before = results.get("sample_results_before", [])
    samples_after = results.get("sample_results_after", [])

    text_parts = []
    for i in range(min(3, len(samples_before), len(samples_after))):
        prompt = samples_before[i]["prompt"][:80]
        text_parts.append(f"[bold]Prompt:[/] {prompt}...\n")
        text_parts.append(f"[red]BEFORE:[/] {samples_before[i]['response'][:120]}...\n")
        text_parts.append(f"[green]AFTER:[/]  {samples_after[i]['response'][:120]}...\n")
        text_parts.append("─" * 60 + "\n")

    return Panel(
        "".join(text_parts),
        title="[bold]Sample Output Comparison[/]",
        border_style="bright_magenta",
    )


def build_export_panel(export_path: str) -> Panel:
    """Build success/export panel."""
    text = f"""
[bold green]Model abliterated successfully![/]

[bold]Export path:[/] [cyan]{export_path}[/]

[bold]To use:[/]
  from transformers import AutoModelForCausalLM, AutoTokenizer
  model = AutoModelForCausalLM.from_pretrained("[cyan]{export_path}[/]")
  tokenizer = AutoTokenizer.from_pretrained("[cyan]{export_path}[/]")

[bold]For Ollama:[/]
  Create a Modelfile pointing to: [cyan]{export_path}[/]
"""
    return Panel(
        text.strip(),
        title="[bold]Export Complete[/]",
        border_style="bright_green",
    )


def run_pipeline(
    model_name: str,
    num_prompts: int = DEFAULT_NUM_PROMPTS,
    coefficient: float = DEFAULT_COEFFICIENT,
    top_k: int = TOP_K_LAYERS,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    skip_eval: bool = False,
    selected_layers_override: List[int] = None,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    trust_remote_code: bool = False,
    strategy: str = "global",
    preserve_norm: bool = True,
    protect_harmless: bool = True,
):
    """
    Execute the complete RUS pipeline:
    1. Load model
    2. Collect activations
    3. Compute refusal subspace
    4. Ablate selected layers
    5. Compare before/after
    6. Export model
    """
    start_time = time.time()
    splash()

    # ── Step 1: Load Model ─────────────────────────────
    console.print("[bold bright_magenta](1/6) Loading model...[/]")
    console.print(f"  Model: [cyan]{model_name}[/]")
    console.print(f"  Device: [yellow]{'CUDA' if torch.cuda.is_available() else 'CPU'}[/]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task("Downloading & loading model...", total=None)
        model, tokenizer = load_model_and_tokenizer(
            model_name,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            trust_remote_code=trust_remote_code,
        )

    device = get_device(model)
    layer_paths, num_layers = discover_layers(model)
    hidden_size = get_hidden_size(model)
    console.print(f"  Architecture: [cyan]{num_layers} layers[/], [cyan]{hidden_size}d hidden[/]")
    console.print(f"  ✓ Model loaded on [green]{device}[/]\n")

    # ── Step 2: Collect Activations ────────────────────
    console.print("[bold bright_magenta](2/6) Collecting activations...[/]")
    n_use = min(num_prompts, len(HARMFUL_PROMPTS), len(HARMLESS_PROMPTS))
    console.print(f"  Using [cyan]{n_use}[/] harmful + [cyan]{n_use}[/] harmless prompts")

    harmful_acts, harmless_acts = collect_pairwise_activations(
        model, tokenizer,
        HARMFUL_PROMPTS[:n_use],
        HARMLESS_PROMPTS[:n_use],
    )
    console.print("  ✓ Activations collected\n")

    # ── Step 3: Compute Refusal Subspace ───────────────
    console.print("[bold bright_magenta](3/6) Computing refusal subspace...[/]")

    directions = compute_refusal_directions(
        harmful_acts, harmless_acts, num_layers,
        protect_harmless=protect_harmless,
    )
    ranked = rank_layers(directions, LAYER_BLACKLIST_FIRST, LAYER_BLACKLIST_LAST)

    if selected_layers_override:
        selected = [
            (l, directions[l]["score"], directions[l]["direction"])
            for l in selected_layers_override
            if l in directions
        ]
        console.print(f"  Using [yellow]manually specified[/] layers: {selected_layers_override}\n")
    else:
        selected = select_best_layers(ranked, top_k)
        console.print(f"  Found [cyan]{len(directions)}[/] analyzable layers")
        console.print(f"  Selected [bright_yellow]{len(selected)}[/] layers for ablation\n")

    table = build_layer_table(directions, selected, top_n=16)
    console.print(table)
    console.print()

    baseline_results = None
    if not skip_eval:
        console.print("  Capturing held-out baseline before modifying weights...")
        baseline_results = evaluate_suite(
            model, tokenizer, HELDOUT_HARMFUL_PROMPTS, HELDOUT_HARMLESS_PROMPTS
        )

    # ── Step 4: Apply Ablation ─────────────────────────
    console.print("[bold bright_magenta](4/6) Applying weight projection ablation...[/]")

    if not selected:
        console.print("[red]No suitable layers found for ablation. Exiting.[/]")
        return

    if strategy == "global":
        consensus, source_layers = build_consensus_direction(selected, len(selected))
        ablation_selected = [(l, s, consensus) for l, s, _ in ranked]
        coefficient_decay = 1.0
        console.print(
            f"  Global consensus from layers [cyan]{source_layers}[/] -> "
            f"[cyan]{len(ablation_selected)}[/] destination layers"
        )
    else:
        ablation_selected = selected
        coefficient_decay = DEFAULT_COEFFICIENT_DECAY

    for rank, (layer_idx, score, direction) in enumerate(ablation_selected):
        coeff = coefficient * (coefficient_decay ** rank)
        console.print(
            f"  Layer [cyan]{layer_idx}[/] | "
            f"score=[yellow]{score:.4f}[/] | "
            f"coeff=[bright_magenta]{coeff:.3f}[/]"
        )

    ablation_stats = apply_ablation(
        model,
        ablation_selected,
        layer_paths,
        coefficient=coefficient,
        coefficient_decay=coefficient_decay,
        preserve_norm=preserve_norm,
    )

    reductions = []
    for layer_idx, stats in ablation_stats.items():
        for tag, tstats in stats.get("targets", {}).items():
            if isinstance(tstats, dict):
                reductions.append(tstats.get("reduction", 0))

    avg_reduction = sum(reductions) / len(reductions) if reductions else 0
    console.print(f"\n  ✓ Average projection reduction: [green]{avg_reduction:.1%}[/]\n")

    # ── Step 5: Before/After Comparison ────────────────
    if skip_eval:
        console.print("[bold bright_magenta](5/6) Skipping evaluation (--skip-eval)\n")
        comparison_results = {
            "refusal_rate_before": 0,
            "refusal_rate_after": 0,
            "compliance_before": 0,
            "compliance_after": 0,
            "quality_before": 0,
            "quality_after": 0,
            "refusal_reduction": 0,
        }
    else:
        console.print("[bold bright_magenta](5/6) Running before/after comparison...")
        console.print(f"  Testing [cyan]{EVAL_HARMFUL_PROMPTS}[/] harmful + harmless prompts")

        after_results = evaluate_suite(
            model, tokenizer, HELDOUT_HARMFUL_PROMPTS, HELDOUT_HARMLESS_PROMPTS
        )
        comparison_results = compare_suites(baseline_results, after_results)

        comp_table = build_comparison_table(comparison_results)
        console.print(comp_table)
        console.print()

        sample_panel = build_sample_comparison(comparison_results)
        console.print(sample_panel)
        console.print()

    # ── Step 6: Export Model ───────────────────────────
    console.print("[bold bright_magenta](6/6) Exporting abliterated model...[/]")

    export_path = export_model(
        model, tokenizer, output_dir, model_name,
        ablation_stats, comparison_results,
        method_metadata={
            "strategy": strategy,
            "source_layers": [item[0] for item in selected],
            "destination_layers": [item[0] for item in ablation_selected],
            "protect_harmless": protect_harmless,
        },
    )

    duration = time.time() - start_time

    layer_refusal_scores = {
        layer_idx: directions[layer_idx]["score"]
        for layer_idx in directions
    }
    try:
        log_run(
            model_name=model_name,
            num_prompts=n_use,
            top_k=top_k,
            coefficient=coefficient,
            layers_modified=[s[0] for s in ablation_selected],
            coefficients_used=[coefficient * (coefficient_decay ** r) for r in range(len(ablation_selected))],
            comparison=comparison_results,
            export_path=export_path,
            duration_seconds=duration,
            layer_refusal_scores=layer_refusal_scores,
            ablation_stats=ablation_stats,
        )
    except Exception as exc:
        warnings.warn(f"Model exported, but run tracking failed: {exc}")

    export_panel = build_export_panel(export_path)
    console.print(export_panel)

    console.print(f"\n[dim]Completed in {duration:.1f}s[/]")
    return export_path


def main():
    parser = argparse.ArgumentParser(
        prog="rus",
        description="RUS — Remove Ur Refusal. Living refusal ablation engine.",
    )
    parser.add_argument(
        "model",
        nargs="?",
        help="HuggingFace model name (e.g. meta-llama/Meta-Llama-3-8B-Instruct)",
    )
    parser.add_argument(
        "--num-prompts", "-n",
        type=int,
        default=DEFAULT_NUM_PROMPTS,
        help=f"Number of harmful/harmless prompts for activation collection (default: {DEFAULT_NUM_PROMPTS})",
    )
    parser.add_argument(
        "--coefficient", "-c",
        type=float,
        default=DEFAULT_COEFFICIENT,
        help=f"Steering coefficient for ablation (default: {DEFAULT_COEFFICIENT})",
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=TOP_K_LAYERS,
        help=f"Number of layers to ablate (default: {TOP_K_LAYERS})",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for abliterated model (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip before/after comparison evaluation",
    )
    parser.add_argument(
        "--layers", "-l",
        type=str,
        default=None,
        help="Comma-separated list of layer indices to ablate (overrides auto-detection)",
    )
    parser.add_argument(
        "--8bit",
        action="store_true",
        dest="load_in_8bit",
        help="Load model in 8-bit quantized (for Colab / low-VRAM GPUs)",
    )
    parser.add_argument(
        "--4bit",
        action="store_true",
        dest="load_in_4bit",
        help="Load model in 4-bit quantized (for very low VRAM)",
    )
    parser.add_argument(
        "--insights",
        action="store_true",
        help="Show learned insights for the model family and exit",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom Python code from the model repository (security-sensitive)",
    )
    parser.add_argument(
        "--strategy",
        choices=("global", "per_layer"),
        default="global",
        help="Global consensus ablation (paper-aligned) or legacy per-layer top-k",
    )
    parser.add_argument(
        "--no-norm-preserve",
        action="store_false",
        dest="preserve_norm",
        help="Disable post-projection weight-vector norm restoration",
    )
    parser.add_argument(
        "--no-protect-harmless",
        action="store_false",
        dest="protect_harmless",
        help="Keep refusal-direction components parallel to the harmless mean",
    )

    args = parser.parse_args()

    if args.load_in_8bit and args.load_in_4bit:
        parser.error("--8bit and --4bit are mutually exclusive")
    if not 0.0 < args.coefficient <= 1.0:
        parser.error("--coefficient must be in (0, 1]")
    if args.top_k <= 0 or args.num_prompts < 3:
        parser.error("--top-k must be positive and --num-prompts must be at least 3")

    if args.insights and args.model:
        family = args.model.lower()
        for fam in ["llama", "mistral", "qwen", "gemma", "phi", "deepseek"]:
            if fam in family:
                insights = get_insights(fam)
                if insights:
                    console.print(f"[bold]Insights for [cyan]{fam}[/] family:[/]")
                    console.print(f"  Best layers: {insights.get('best_layers', 'N/A')}")
                    console.print(f"  Best coefficient: {insights.get('best_coefficient', 'N/A')}")
                    console.print(f"  Avg refusal score: {insights.get('avg_refusal_score_best', 'N/A')}")
                    console.print(f"  Total runs: {insights.get('num_runs', 0)}")
                else:
                    console.print(f"[yellow]No insights yet for [cyan]{fam}[/] family.[/]")
                return

    if not args.model:
        parser.print_help()
        console.print("\n[yellow]Example:[/] python -m rus meta-llama/Meta-Llama-3-8B-Instruct")
        return

    selected_layers = None
    if args.layers:
        try:
            selected_layers = [int(x.strip()) for x in args.layers.split(",")]
        except ValueError:
            console.print("[red]Invalid --layers format. Use comma-separated integers.[/]")
            sys.exit(1)

    run_pipeline(
        model_name=args.model,
        num_prompts=args.num_prompts,
        coefficient=args.coefficient,
        top_k=args.top_k,
        output_dir=args.output,
        skip_eval=args.skip_eval,
        selected_layers_override=selected_layers,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        trust_remote_code=args.trust_remote_code,
        strategy=args.strategy,
        preserve_norm=args.preserve_norm,
        protect_harmless=args.protect_harmless,
    )


if __name__ == "__main__":
    main()
