# Handout — Backpropagation, Derived From Scratch

The [Week 1 lecture notes](../lectures/week01-introduction-to-neural-networks.md) walk through backpropagation using a small 2-2-2-1 network. This handout goes one level deeper: a complete, self-contained derivation for the simplest possible non-trivial case — a two-layer network with one input, one hidden neuron, and one output neuron — so that every single step is explicit and nothing is skipped. Once you're comfortable with this fully-worked, minimal example, generalizing to wider layers (as in the lecture notes) is just a matter of repeating the same steps once per neuron.

## The network

Consider a network with a single input `x`, one hidden neuron with weight `w1`, bias `b1`, and a sigmoid activation, feeding into a single output neuron with weight `w2`, bias `b2`, and a linear (identity) activation:

```
z1 = w1 · x + b1
a1 = σ(z1) = 1 / (1 + e^(-z1))
z2 = w2 · a1 + b2
ŷ = z2                          (linear output, for a regression task)
```

We'll use mean squared error as the loss for a single training example `(x, y)`:

```
L = (1/2)(ŷ - y)²
```

## Goal

We want `∂L/∂w1`, `∂L/∂b1`, `∂L/∂w2`, `∂L/∂b2` — the gradient of the loss with respect to every parameter — so we can update each one with gradient descent, `θ ← θ - η · ∂L/∂θ`.

## Step 1 — The easy ones: gradients for the output layer

Since `w2` and `b2` only affect the loss *directly* through `z2` (there's no further chain beyond that), we apply the chain rule through just two steps: `L → ŷ → z2 → w2` (and similarly for `b2`).

**`∂L/∂ŷ`:** since `L = (1/2)(ŷ-y)²`, treating `ŷ` as the variable, `∂L/∂ŷ = ŷ - y`. (This is the single most common gradient in the whole course — it shows up every time MSE is the loss.)

**`∂ŷ/∂z2`:** since `ŷ = z2` (linear/identity activation), `∂ŷ/∂z2 = 1`.

**`∂z2/∂w2`:** since `z2 = w2·a1 + b2`, treating `w2` as the variable and everything else as constant, `∂z2/∂w2 = a1`.

**`∂z2/∂b2`:** similarly, `∂z2/∂b2 = 1`.

Chaining these together:

```
∂L/∂w2 = (∂L/∂ŷ) · (∂ŷ/∂z2) · (∂z2/∂w2) = (ŷ - y) · 1 · a1 = (ŷ - y) · a1
∂L/∂b2 = (∂L/∂ŷ) · (∂ŷ/∂z2) · (∂z2/∂b2) = (ŷ - y) · 1 · 1 = (ŷ - y)
```

Notice `(ŷ - y)` appears in both — it's the *error signal* at the output, and it's worth naming: `δ2 = ŷ - y`. So `∂L/∂w2 = δ2 · a1` and `∂L/∂b2 = δ2`.

## Step 2 — The harder ones: gradients for the hidden layer

Now we want `∂L/∂w1` and `∂L/∂b1`. The catch: `w1` doesn't affect `L` directly — it affects `z1`, which affects `a1` (through the sigmoid), which affects `z2`, which affects `ŷ`, which affects `L`. The chain rule handles arbitrarily long chains like this by simply multiplying every link together:

```
∂L/∂w1 = (∂L/∂ŷ) · (∂ŷ/∂z2) · (∂z2/∂a1) · (∂a1/∂z1) · (∂z1/∂w1)
```

We already have the first two factors from Step 1: `∂L/∂ŷ = ŷ-y` and `∂ŷ/∂z2 = 1`. We need three new ones.

**`∂z2/∂a1`:** since `z2 = w2·a1 + b2`, treating `a1` as the variable, `∂z2/∂a1 = w2`. (Compare this to `∂z2/∂w2 = a1` from Step 1 — same equation, different variable of interest, so a different one of the two factors "drops out" and the other survives.)

**`∂a1/∂z1`:** this is the derivative of the sigmoid function itself. A standard (and worth memorizing) result is `σ'(z) = σ(z)(1-σ(z))`, so `∂a1/∂z1 = a1(1-a1)` (since `a1 = σ(z1)`).

**`∂z1/∂w1`:** since `z1 = w1·x + b1`, treating `w1` as the variable, `∂z1/∂w1 = x`.

Multiplying every factor together:

```
∂L/∂w1 = (ŷ - y) · 1 · w2 · a1(1-a1) · x
```

And, since `∂z1/∂b1 = 1` (by the same logic as `∂z2/∂b2` in Step 1):

```
∂L/∂b1 = (ŷ - y) · 1 · w2 · a1(1-a1) · 1 = (ŷ - y) · w2 · a1(1-a1)
```

Notice the pattern: `∂L/∂w1` and `∂L/∂b1` share every factor except the very last one (`x` vs. `1`) — exactly mirroring how `∂L/∂w2` and `∂L/∂b2` shared everything except their last factor (`a1` vs. `1`) in Step 1.

## Step 3 — Naming the reusable piece: the "delta" trick

Look closely at `∂L/∂w1` and notice it contains the *entire* output-layer error signal `δ2 = (ŷ-y)` as a sub-expression, multiplied by some *new* hidden-layer-specific terms. Define:

```
δ1 = δ2 · w2 · a1(1-a1)     (the "error signal" backpropagated into the hidden layer)
```

Then, cleanly:

```
∂L/∂w1 = δ1 · x
∂L/∂b1 = δ1
```

Compare this to Step 1's `∂L/∂w2 = δ2 · a1` and `∂L/∂b2 = δ2`. **The pattern is identical at every layer:** a weight's gradient is always *that layer's error signal, multiplied by that layer's input*, and a bias's gradient is always *just the error signal*. This is exactly why backpropagation is efficient: rather than recomputing the full chain-rule product from scratch for every single weight, you compute one `δ` per layer (working backward from the output), and every weight in that layer's gradient reuses the same `δ`. In a network with wide layers (many neurons per layer, as in the Week 1 lecture's 2-2-2-1 example), this same idea generalizes directly: each neuron has its own `δ`, computed from the `δ`s of the layer *after* it (weighted by the connecting weights and multiplied by the local activation derivative), and every weight feeding into that neuron reuses its `δ`.

## Step 4 — Putting it all together as an algorithm

1. **Forward pass:** compute `z1, a1, z2, ŷ` in order, given the current weights and an input `x`.
2. **Compute the loss:** `L = (1/2)(ŷ-y)²`.
3. **Backward pass, output layer:** `δ2 = ŷ - y`; gradients are `δ2·a1` and `δ2`.
4. **Backward pass, hidden layer:** `δ1 = δ2 · w2 · a1(1-a1)`; gradients are `δ1·x` and `δ1`.
5. **Update:** `w2 ← w2 - η·δ2·a1`, `b2 ← b2 - η·δ2`, `w1 ← w1 - η·δ1·x`, `b1 ← b1 - η·δ1`.

## A worked numerical example

Let `x=2`, `y=5`, `w1=0.5`, `b1=0`, `w2=1.0`, `b2=0`.

Forward: `z1 = 0.5·2 + 0 = 1.0`; `a1 = σ(1.0) ≈ 0.7311`; `z2 = 1.0·0.7311 + 0 = 0.7311`; `ŷ = 0.7311`.

Loss: `L = 0.5·(0.7311-5)² ≈ 9.128`.

Backward: `δ2 = ŷ - y = 0.7311 - 5 = -4.2689`. Gradients for layer 2: `∂L/∂w2 = δ2·a1 = -4.2689 × 0.7311 ≈ -3.1214`; `∂L/∂b2 = δ2 = -4.2689`.

`δ1 = δ2 · w2 · a1(1-a1) = -4.2689 × 1.0 × 0.7311×(1-0.7311) = -4.2689 × 0.1966 ≈ -0.8393`. Gradients for layer 1: `∂L/∂w1 = δ1·x = -0.8393 × 2 ≈ -1.6786`; `∂L/∂b1 = δ1 ≈ -0.8393`.

With `η = 0.1`: `w2 ← 1.0 - 0.1×(-3.1214) = 1.3121`; `w1 ← 0.5 - 0.1×(-1.6786) = 0.6679`. Try re-running the forward pass with these updated weights — you should find `ŷ` has moved closer to `y=5`.

## Generalizing to wider layers and more layers

Everything above generalizes directly to the vector/matrix form used in the [Week 1 lecture notes](../lectures/week01-introduction-to-neural-networks.md): each layer's `δ` becomes a *vector* (one entry per neuron in that layer) instead of a scalar, `∂L/∂W = δ · aᵀ_prev` (an outer product) instead of a simple product, and computing the `δ` for one layer from the next layer's `δ` involves a matrix-vector product with that layer's weight matrix (`δ_current = (W_nextᵀ δ_next) ⊙ g'(z_current)`) instead of a simple scalar multiplication. The underlying logic — propagate an error signal backward one layer at a time, reusing it for every weight in that layer — is exactly what you just derived by hand above.
