# RUS — Remove Ur Refusal

RUS is an experimental representation-engineering toolkit that measures a
harmful-vs-harmless activation direction and projects it out of selected output
weights in a Hugging Face causal language model.

> **Research warning:** this transformation can weaken model safeguards. It does
> not make outputs correct, lawful, or safe. Test only in an isolated environment
> and perform independent capability and safety evaluation before deployment.

## Quick start

```bash
pip install -e '.[gpu,test]'
rus Qwen/Qwen2.5-7B-Instruct --8bit --num-prompts 48 --top-k 5
```

Or from Python:

```python
import rus

path = rus.ablate(
    "Qwen/Qwen2.5-7B-Instruct",
    load_in_8bit=True,
    num_prompts=48,
    k=5,
    coefficient=0.8,
)
```

The pipeline captures a held-out baseline before modifying the in-memory model,
so quantized evaluation does not require loading a second model onto the GPU.

RUS 1.3 defaults to a sign-aligned, score-weighted global direction applied at
all compatible residual-writing output projections with norm preservation. See
[`RESEARCH.md`](RESEARCH.md) for the derivation, assumptions, and references.

## Kaggle: Qwen2.5 7B on two T4 GPUs

Open [`rus_kaggle_dual_t4_7b.ipynb`](rus_kaggle_dual_t4_7b.ipynb) in a Kaggle
GPU notebook and select the **GPU T4 x2** accelerator. The notebook loads the
7B checkpoint in 8-bit mode, balances transformer modules over both GPUs,
prints per-GPU memory use, verifies placement, runs the pipeline, then reloads
the exported checkpoint and performs a real forward pass. If Kaggle assigns
only one GPU, the notebook automatically uses a single-GPU 7B fallback.

The equivalent CLI settings are:

```bash
rus Qwen/Qwen2.5-7B-Instruct --8bit \
  --device-map balanced \
  --max-memory 0=12GiB,1=13GiB,cpu=24GiB \
  --num-prompts 48 --top-k 5
```

This is Accelerate model parallelism: layers are split between GPUs to increase
available memory. A normal forward pass still executes those layers in order,
so this is not tensor-parallel speed scaling.

## Supported paths

- Standard fp16/bf16 `nn.Linear` output projections.
- bitsandbytes 8-bit output projections, replaced with projected fp16 modules.
- bitsandbytes NF4 4-bit output projections, dequantized and replaced with fp16.
- GPT-style `Conv1D` output-axis orientation.
- Llama/Qwen/Mistral/Gemma/Phi projections, plus OPT, GPT-NeoX,
  BLOOM/Falcon-style output-projection names.
- Accelerate `auto`, `balanced`, `balanced_low_0`, explicit device maps,
  per-device memory limits, and optional CPU/disk offload.
- Global consensus and legacy per-layer ablation strategies.
- Harmless-direction protection, norm preservation, and held-out KL drift.

Architectures vary. RUS fails explicitly when it cannot discover compatible
layers or output projections. Successful export is not proof that refusal was
removed; inspect `rus_metadata.json` and perform independent evaluation.

## Validation

```bash
python tests/test_core.py
python -m pytest -q
```

The older single-GPU Colab workflow remains in `rus_colab_llama.ipynb`; use the
Kaggle notebook above for the dual-T4 7B test.
