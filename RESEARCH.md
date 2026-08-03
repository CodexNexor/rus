# Mathematical design of RUS

This document records the mathematical assumptions behind RUS 1.2. The methods
are experimental: refusal is not guaranteed to be one-dimensional or invariant
across prompt categories and model families.

## 1. Residual-stream measurements

Let \(h_{l,t}(x) \in \mathbb{R}^d\) be the residual-stream activation after
transformer block \(l\), at the final non-padding instruction token \(t\), for
prompt \(x\). For harmful set \(H\) and harmless set \(B\), RUS estimates

\[
\mu^H_l = \frac{1}{|H|}\sum_{x\in H}h_{l,t}(x),\qquad
\mu^B_l = \frac{1}{|B|}\sum_{x\in B}h_{l,t}(x).
\]

The raw difference-in-means direction is

\[
r_l = \mu^H_l-\mu^B_l.
\]

This follows Arditi et al. (2024). Centered PCA is deliberately not used for the
primary vector: PCA of centered pair differences finds variation among pairs and
need not point between the two class centroids.

## 2. Protecting the harmless direction

The optional harmless reference is \(b_l=\mu^B_l/\|\mu^B_l\|_2\). RUS removes
the component of the refusal estimate parallel to it:

\[
\tilde r_l=r_l-b_l(b_l^\top r_l),\qquad
\hat r_l=\tilde r_l/\|\tilde r_l\|_2.
\]

This is a rank-one nuisance projection. It is related to projected abliteration
variants and to the broader concept-erasure goal of removing class covariance
while minimizing changes to unrelated representation geometry. It is not full
LEACE, which uses feature covariance whitening and an affine eraser.

## 3. Layer score

RUS projects both classes onto \(\hat r_l\). With projected values \(z_H,z_B\),
it computes a pooled standardized separation

\[
d_l=\frac{|\mathbb{E}z_H-\mathbb{E}z_B|}
{\sqrt{(\mathrm{Var}(z_H)+\mathrm{Var}(z_B))/2}+\epsilon}.
\]

The displayed bounded score is \(s_l=d_l/(1+d_l)\). This is a ranking statistic,
not a probability and not proof that the direction is causally necessary.

## 4. Cross-layer consensus

Residual-stream coordinates are shared across ordinary decoder blocks, while
finite prompt samples make individual \(\hat r_l\) noisy. Starting with the
highest-scoring candidate \(q\), RUS sign-aligns candidate directions:

\[
r'_l=\operatorname{sign}(q^\top\hat r_l)\hat r_l.
\]

For source layer set \(S\), the global direction is

\[
r_* = \frac{\sum_{l\in S}s_l r'_l}
{\left\|\sum_{l\in S}s_l r'_l\right\|_2}.
\]

Sign alignment matters because \(r\) and \(-r\) define the same erased
one-dimensional subspace but would cancel in a naive average.

## 5. Weight orthogonalization

For a linear output map \(y=Wx\), where \(W\in\mathbb{R}^{d_{out}\times d_{in}}\)
and unit refusal direction \(r\in\mathbb{R}^{d_{out}}\), define

\[
P_r=I-rr^\top.
\]

Full orthogonalization uses

\[
W'=P_rW=W-r(r^\top W).
\]

Then \(r^\top W'=0\), so the module cannot write into direction \(r\). Partial
ablation with strength \(\alpha\in(0,1]\) is

\[
W'=W-\alpha r(r^\top W),
\quad r^\top W'=(1-\alpha)r^\top W.
\]

Transformers `Conv1D` stores \(W\) transposed; RUS therefore applies the
equivalent right projection \(W'=W-\alpha(Wr)r^\top\). The global strategy uses
the consensus direction at every compatible non-boundary block, matching the
paper's argument that orthogonalizing every residual-writing output map is
equivalent to residual-stream directional ablation.

## 6. Norm preservation

Projection shortens each output weight vector. For column \(w_j\), RUS can
restore its original magnitude without changing its projected ray:

\[
\bar w_j=P_rw_j,\qquad
w'_j=\bar w_j\frac{\|w_j\|_2}{\|\bar w_j\|_2+\epsilon}.
\]

For full projection, \(r^\top w'_j=0\) still holds. For partial projection this
preserves magnitude but slightly changes the exact \((1-\alpha)\) scaling.

## 7. Capability-preservation measurement

Response length is not a capability metric. RUS also stores the original and
modified next-token distributions on held-out harmless prompts and reports

\[
D_{KL}(p_{before}\|p_{after})=
\sum_v p_{before}(v)\log\frac{p_{before}(v)}{p_{after}(v)}.
\]

Lower KL means the local output distribution changed less. It complements, but
does not replace, task benchmarks such as MMLU, ARC, GSM8K, perplexity, or human
evaluation.

## 8. Why RUS remains rank one by default

Later work finds multiple refusal directions and concept-cone geometry. Blindly
removing several singular vectors increases edit rank and capability damage.
RUS therefore uses a denoised rank-one consensus by default and exposes the
legacy per-layer strategy for comparison. Multi-direction erasure should be
introduced only with category-stratified data, causal validation, and a
quality-constrained selection objective.

## Primary references

- Arditi et al., *Refusal in Language Models Is Mediated by a Single Direction*
  (2024): https://arxiv.org/abs/2406.11717
- Belrose et al., *LEACE: Perfect Linear Concept Erasure in Closed Form*
  (2023, revised 2025): https://arxiv.org/abs/2306.03819
- Wollschläger et al., *The Geometry of Refusal in Large Language Models:
  Concept Cones and Representational Independence* (2025):
  https://arxiv.org/abs/2502.17420
- Arditi et al. reference implementation:
  https://github.com/andyrdt/refusal_direction
