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
rus Qwen/Qwen2.5-3B-Instruct --8bit --num-prompts 48 --top-k 5
```

Or from Python:

```python
import rus

path = rus.ablate(
    "Qwen/Qwen2.5-3B-Instruct",
    load_in_8bit=True,
    num_prompts=48,
    k=5,
    coefficient=0.8,
)
```

The pipeline captures a held-out baseline before modifying the in-memory model,
so quantized evaluation does not require loading a second model onto the GPU.

RUS 1.2 defaults to a sign-aligned, score-weighted global direction applied at
all compatible residual-writing output projections with norm preservation. See
[`RESEARCH.md`](RESEARCH.md) for the derivation, assumptions, and references.

## Supported paths

- Standard fp16/bf16 `nn.Linear` output projections.
- bitsandbytes 8-bit output projections, replaced with projected fp16 modules.
- bitsandbytes NF4 4-bit output projections, dequantized and replaced with fp16.
- GPT-style `Conv1D` output-axis orientation.
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

The Colab workflow in `rus_colab_llama.ipynb` uses the ungated
`Qwen/Qwen2.5-3B-Instruct` checkpoint, which fits a free T4 in 8-bit mode.
