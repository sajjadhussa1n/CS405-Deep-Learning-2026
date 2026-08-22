# Handout — Linear Algebra and Calculus Primer

This handout collects, in one place, the specific pieces of linear algebra and calculus that CS-405 actually leans on, at the level of detail needed to follow the derivations in [`lectures/`](../lectures/). It is not a substitute for a full linear algebra or calculus course — it's a targeted refresher of exactly what comes up repeatedly in this course, written so you can look something up quickly rather than re-derive it under time pressure.

## Vectors and matrices

A **vector** `x ∈ ℝⁿ` is an ordered list of `n` real numbers; we write it as a column: `x = [x_1, x_2, ..., x_n]ᵀ`. A **matrix** `W ∈ ℝᵐˣⁿ` is a rectangular grid of numbers with `m` rows and `n` columns. Throughout the course, weight matrices transform vectors from one space into another — a matrix `W ∈ ℝᵐˣⁿ` maps an `n`-dimensional input vector to an `m`-dimensional output vector via matrix-vector multiplication.

**Matrix-vector multiplication:** if `W` is `m×n` and `x` is `n×1`, then `z = Wx` is `m×1`, where each entry `z_i = Σ_j W_ij x_j` — the dot product of row `i` of `W` with `x`. This single operation is what every "weighted sum" in the course — a perceptron's `Σw_ix_i`, a fully connected layer's `Wx+b`, a convolution's local dot product — ultimately reduces to.

**Matrix-matrix multiplication:** if `A` is `m×k` and `B` is `k×n`, then `C = AB` is `m×n`, where `C_ij = Σ_l A_il B_lj`. In the course, this shows up whenever an entire *batch* of examples is processed at once (`Z = WX` where `X`'s columns are individual examples), and centrally in self-attention (`QK^T`).

**The dot product** of two vectors `u, v ∈ ℝⁿ` is `u·v = Σ_i u_i v_i`, a single scalar. Geometrically, `u·v = ‖u‖‖v‖cos(θ)`, where `θ` is the angle between them — this is why a large dot product between two vectors indicates they point in similar directions (used directly in Week 11's cosine similarity between word embeddings, and in Week 13's attention scores between queries and keys).

**Transpose:** `Aᵀ` flips a matrix across its diagonal, turning an `m×n` matrix into `n×m`. It appears constantly in the course simply to make matrix-multiplication shapes line up correctly (e.g., `QKᵀ` in self-attention).

## Norms

The **L2 norm** (Euclidean length) of a vector is `‖x‖₂ = √(Σ x_i²)`. The **L1 norm** is `‖x‖₁ = Σ|x_i|`. These appear directly in Week 2's L2 and L1 regularization terms, and the L2 norm specifically underlies mean squared error loss (`‖x - x̂‖²`).

## Derivatives, the chain rule, and gradients

A **derivative** `df/dx` measures how much a function's output changes for a tiny change in its input — the slope of the function at a point. Every "how much does the loss change if I nudge this weight" question in the course is a derivative question.

The **chain rule** is the single most important calculus tool in this course: if `y = f(u)` and `u = g(x)`, then `dy/dx = (dy/du) · (du/dx)`. It lets you break a complicated derivative into a chain of simpler, local derivatives multiplied together — this is *literally* what backpropagation is (see the [Backpropagation Derivation](handout02-backpropagation-derivation.md) handout for a fully worked example). For a function of several intermediate steps, `y = f(g(h(x)))`, the chain rule extends naturally: `dy/dx = (dy/df)·(df/dg)·(dg/dh)·(dh/dx)`.

For a function of *multiple* inputs, `f(x_1, ..., x_n)`, the **partial derivative** `∂f/∂x_i` treats every other input as constant while differentiating with respect to `x_i`. The **gradient**, `∇f = [∂f/∂x_1, ..., ∂f/∂x_n]`, collects all the partial derivatives into a vector, and it points in the direction of steepest *increase* of `f` — which is why gradient *descent* (Week 1) moves in the *negative* gradient direction, `θ ← θ - η∇f`, to decrease a loss.

## Common derivatives used throughout the course

| Function | Derivative | Where it's used |
|---|---|---|
| `f(x) = x²` | `f'(x) = 2x` | mean squared error |
| `f(x) = e^x` | `f'(x) = e^x` | softmax, sigmoid |
| `σ(x) = 1/(1+e^-x)` | `σ'(x) = σ(x)(1-σ(x))` | Week 1–2 activation derivatives |
| `tanh(x)` | `1 - tanh²(x)` | Week 6–7 RNN/LSTM/GRU derivatives |
| `ReLU(x) = max(0,x)` | `1 if x>0 else 0` | Week 1–4 |
| `log(x)` | `1/x` | cross-entropy loss derivatives |

## Probability basics used in the course

An **expectation** `E[X]` is the probability-weighted average value of a random quantity `X` — it appears throughout as `E_{x~p}[...]` notation (e.g., in the GAN objective, Week 10, and the VAE's ELBO, Week 9), meaning "average this quantity over samples drawn from distribution `p`."

The **Gaussian (normal) distribution**, `N(μ, σ²)`, is fully described by its mean `μ` and variance `σ²`; its probability density is highest at `μ` and falls off symmetrically. It underlies weight initialization schemes (Week 2), the VAE's latent prior (Week 9), and GAN noise vectors (Week 10).

**Cross-entropy** between a true distribution `y` (usually one-hot) and a predicted distribution `ŷ` is `-Σ y_i log(ŷ_i)`; when `y` is one-hot with the 1 at index `c`, this collapses to `-log(ŷ_c)` — the loss function used almost everywhere a network makes a classification-style prediction (Weeks 1, 6, 11, 13).

**KL divergence** `D_KL(q‖p)` measures how different one probability distribution `q` is from a reference distribution `p`; it is 0 when the two distributions are identical and grows as they diverge. It's central to the VAE loss (Week 9).

## Where this shows up in the course

If you can comfortably read `z = Wx + b`, compute a dot product by hand, apply the chain rule through 2–3 nested functions, and know what `∇f` and `argmax` mean, you have everything you need for the derivations in every week's lecture notes. When a specific week's notes use something beyond this handout, they explain it in place — this handout is meant to cover the recurring background, not everything.
