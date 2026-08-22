# Week 7 — Long Short-Term Memory (LSTM)

*Companion notes for [`slides/lecture_week07.pdf`](../slides/lecture_week07.pdf)*

## Why this week matters

Last week's GRU fixed most of the vanishing gradient problem with a single update gate. LSTM is the original solution to the same problem, published by Hochreiter & Schmidhuber all the way back in 1997 — long before GRU existed — and it takes a more elaborate but even more powerful approach: instead of one combined memory, it keeps **two separate memories** and uses **three gates**, rather than two, to control them with finer precision. Understanding LSTM in detail also pays off directly in Weeks 11–13, where gated recurrent ideas underlie many sequence-to-sequence and attention-based models.

## 1. The core idea: a separate, protected long-term memory

The single biggest structural difference between LSTM and GRU is this: **GRU combines short-term and long-term memory into one hidden state `h⟨t⟩`; LSTM keeps them separate.** LSTM maintains a **cell state** `c⟨t⟩`, which acts as a long-term memory "highway," alongside a **hidden state** `h⟨t⟩`, which acts as the short-term, externally visible output. The key design intuition is that the cell state should be updated through *only simple linear operations* (element-wise addition and multiplication by gate values) as it moves from one time step to the next — no matrix multiplication and no repeated squashing through `tanh` sits directly in its path — which is exactly what lets gradients flow through it largely undiminished, forming a genuine "gradient highway."

Three gates, each a vector of values between 0 and 1 (via sigmoid), control what happens to this memory:

- **Forget gate `f⟨t⟩`** — controls what to discard from the existing long-term memory.
- **Input gate `i⟨t⟩`** — controls how much new information to add to the long-term memory.
- **Output gate `o⟨t⟩`** — controls how much of the long-term memory to reveal as this step's output.

All three gates, plus a candidate for new memory content, are computed from the same two inputs at every time step: the previous hidden state `h⟨t-1⟩` and the current input `x⟨t⟩`.

## 2. The four LSTM equations

```
f⟨t⟩ = σ(W_f · [h⟨t-1⟩, x⟨t⟩] + b_f)      # forget gate
i⟨t⟩ = σ(W_i · [h⟨t-1⟩, x⟨t⟩] + b_i)      # input gate
c̃⟨t⟩ = tanh(W_c · [h⟨t-1⟩, x⟨t⟩] + b_c)   # candidate cell content
o⟨t⟩ = σ(W_o · [h⟨t-1⟩, x⟨t⟩] + b_o)      # output gate
```

Notice the pattern: every *gate* uses a sigmoid, because a gate's job is to represent "how much" (a fraction between 0 and 1) — while the *candidate* content `c̃⟨t⟩` uses `tanh`, exactly like a plain RNN's hidden-state update, because its job is to represent an actual proposed *value* (bounded between -1 and 1), not a proportion.

### Updating the cell state (long-term memory)

```
c⟨t⟩ = f⟨t⟩ ⊙ c⟨t-1⟩ + i⟨t⟩ ⊙ c̃⟨t⟩
        \_________/    \________/
         forget part     new part
```

This single line is the heart of LSTM. Reading it in words: the new long-term memory equals (how much of the old memory we keep) plus (how much of the new candidate we add in). If `f⟨t⟩ ≈ 1` and `i⟨t⟩ ≈ 0`, the cell simply carries the old memory forward unchanged, ignoring the new input entirely. If `f⟨t⟩ ≈ 0` and `i⟨t⟩ ≈ 1`, the old memory is discarded and replaced with the new candidate. If both are around 0.5, old and new memory are blended. Note that unlike GRU's update gate (where `z` and `1-z` are forced to sum to exactly 1), LSTM's forget and input gates are computed *independently* — the network is free to, say, forget very little *and* add very little, or forget a lot *and* add a lot, giving it more flexible control than GRU's single interpolation knob.

### Producing the hidden state (short-term, visible output)

```
h⟨t⟩ = o⟨t⟩ ⊙ tanh(c⟨t⟩)
```

The cell state is first squashed into `[-1, 1]` with `tanh`, and then the output gate decides how much of that to actually reveal as this step's hidden state. If `o⟨t⟩ ≈ 0`, the hidden state (and therefore this step's externally visible output) is near zero, *even if the cell state itself contains rich information* — the network can deliberately keep something in long-term memory without exposing it at every intermediate step. If `o⟨t⟩ ≈ 1`, the hidden state directly reveals the (squashed) cell state. This separation — "remember something internally" versus "show it externally right now" — is something a plain RNN or even a GRU cannot do as cleanly, since GRU's hidden state *is* its only memory, with no private, unexposed channel.

Putting it all together, one LSTM step has three logical stages: **forget** (`f⟨t⟩ ⊙ c⟨t-1⟩` — what to remove from long-term memory), **input** (`i⟨t⟩ ⊙ c̃⟨t⟩` — what to add), and **output** (`o⟨t⟩ ⊙ tanh(c⟨t⟩)` — what to reveal).

## 3. Why LSTM solves the vanishing gradient problem

Compare the gradient path for a plain RNN's hidden state against LSTM's cell state. In a plain RNN, `∂h⟨t⟩/∂h⟨t-1⟩ = diag(1-(h⟨t⟩)²) · W_hh^T` — a purely *multiplicative* path involving both a squashed `tanh` derivative and a full weight matrix, which is exactly what causes gradients to shrink (or explode) exponentially over many steps, as we saw last week. In LSTM, differentiating the cell-state update gives (to first approximation) `∂c⟨t⟩/∂c⟨t-1⟩ ≈ diag(f⟨t⟩) + (additional terms from the gates)`. The key term here is `diag(f⟨t⟩)` — if the forget gate is close to 1, this factor is close to the identity, meaning **the gradient passes through that time step essentially unchanged**, with no squashing nonlinearity and no weight matrix sitting directly in the cell state's own path. This is precisely the "gradient highway" idea: the cell state's path across time is dominated by simple, near-identity linear operations, controlled by a learned gate, rather than by a fixed multiplicative transformation applied at every step regardless of what's useful to remember.

### The "France...French" example, revisited

Using the same running example from last week — "I grew up in France. I moved to Germany for work. After 10 years, I now speak fluent French." — at `t=5` ("France"), suppose the forget gate for a "country" dimension of the cell state is `f_5 ≈ 0.95` (mostly keep what's there) with a moderate input gate `i_5 ≈ 0.3` adding new information, so the cell state now strongly encodes "France." Through the filler words from `t=6` to `t=18`, that same dimension keeps `f⟨t⟩ ≈ 0.95` at each step, so the memory decays only as `0.95^14 ≈ 0.49` — still strong after 14 steps. At `t=19` ("French"), the input gate opens (`i_19 ≈ 0.4`) to blend in the new word, and the output gate opens (`o_19 ≈ 0.9`) to actually reveal the combined "France + French → language" representation in the hidden state, ready to inform the prediction.

A direct numerical comparison across 50 steps, tracking a single important dimension, illustrates the gap between architectures: a plain RNN (each step multiplying by roughly 0.8) decays to about `0.00001` of its original strength after 50 steps — completely gone; a GRU or LSTM (each step effectively multiplying by a gate value of `0.95`) retains about `0.08` — dramatically stronger, and comparable to each other when their gates are tuned similarly. LSTM's edge over GRU in practice comes from three additional degrees of freedom: a genuinely separate cell state that lets information be "hidden" from the output at intermediate steps (via the output gate), independently-controlled forget and input gates for finer-grained control (rather than one gate implicitly controlling both), and, as a result, the ability to handle even longer dependencies when the extra control is actually needed.

## 4. LSTM vs. GRU, side by side

| Feature | RNN | GRU | LSTM |
|---|---|---|---|
| Separate cell state | no | no | **yes** (`c⟨t⟩`) |
| Number of gates | 0 | 2 (`z`, `r`) | 3 (`f`, `i`, `o`) |
| Explicit forget gate | no | no (implicit via `z`) | **yes** |
| Output gate (hide info) | no | no | **yes** |
| Gradient path | multiplicative only | additive (via `z`) | additive (via cell state) |
| Vanishing gradients | severe | mild | mildest |
| Parameter count | lowest | medium | highest |
| Training speed | fastest | fast | slowest |
| Long-range memory | poor | good | excellent |

Practical guidance: choose **LSTM** for very long sequences (100+ steps), when you need fine-grained control over memory or the ability to hide information internally, and when you have enough data and compute to support its larger parameter count — LSTM is also often preferred for speech/audio tasks. Choose **GRU** for moderate-length sequences, faster training, more limited data or compute, or simply a simpler model — GRU is frequently sufficient for text-based tasks. A sensible rule of thumb from the slides: **start with GRU for quick prototyping, and move to LSTM only if you need more capacity or actually observe vanishing-gradient-style symptoms.**

## Key takeaways

LSTM was the original (1997) fix for the vanishing gradient problem in recurrent networks, predating GRU by 17 years, and it solves the problem by maintaining a separate cell state `c⟨t⟩` that is updated through simple, gated linear operations — `c⟨t⟩ = f⟨t⟩ ⊙ c⟨t-1⟩ + i⟨t⟩ ⊙ c̃⟨t⟩` — creating a near-identity "gradient highway" whenever the forget gate is close to 1. A separate, learned output gate then controls how much of that internal memory gets exposed as the visible hidden state, `h⟨t⟩ = o⟨t⟩ ⊙ tanh(c⟨t⟩)`, letting the network keep information in long-term memory without necessarily broadcasting it at every step. Compared to GRU's single update gate, LSTM's three independent gates (forget, input, output) give it finer-grained, more flexible control at the cost of more parameters and slower training — a trade-off worth making specifically when sequences are very long or memory control needs to be precise.
