# Week 6 — Sequential Models: From RNNs to GRUs

*Companion notes for [`slides/lecture_week06.pdf`](../slides/lecture_week06.pdf)*

## Why this week matters

Everything up through Week 5 assumed inputs are independent of each other — a batch of unrelated images. Language, time series, audio, video, and DNA are not like that: order carries meaning, and the current output usually depends on everything that came before it. This week introduces the recurrent neural network (RNN), the algorithm used to train it (backpropagation through time), and the vanishing-gradient problem that limits plain RNNs — motivating the Gated Recurrent Unit (GRU), a first fix that we'll extend further with LSTM next week.

## 1. What makes data "sequential," and why plain networks fail on it

A sequential model is designed for data where **order matters** — inputs are not independent, the current output depends on previous inputs, and sequences can have variable length. As the slides put it: you can't understand Chapter 10 of a book without having read Chapters 1–9. This directly contradicts the "i.i.d." (independent and identically distributed) assumption that underlies most traditional machine learning and the feedforward networks of Weeks 1–2. Sequential data is everywhere: words in a sentence (NLP), stock prices over days (time series), frames in a video, audio signals (speech), and DNA sequences (biology) — and the applications built on top are equally broad: machine translation, sentiment analysis, named entity recognition, stock and weather prediction, video classification, gene prediction, and speech recognition.

Standard feedforward networks (FNNs) fail on this kind of data for three concrete reasons. First, **fixed input size**: an FNN needs a fixed-length input vector, but sentences have variable length. Second, **no parameter sharing across positions**: an FNN learns a separate set of weights for each input position, so a pattern like "France implies French" would have to be learned separately depending on *where* in the sentence "France" happens to appear — it can't generalize across positions the way a convolution's shared weights generalize across image locations (Week 3). Third, **no memory**: each prediction is made independently, so the network has no way to use context from earlier in the sequence — a word is treated in total isolation from every word around it.

## 2. The RNN: adding memory

The recurrent neural network's core idea is to process a sequence **step by step** while maintaining a **hidden state** — a vector that acts as the network's memory. At each time step `t`, the RNN reads the current input `x⟨t⟩`, updates its memory using both that input and the *previous* memory `h⟨t-1⟩`, and produces an output `ŷ⟨t⟩` based on the current memory. Crucially, `h⟨0⟩` (the initial memory, usually all zeros) evolves into `h⟨1⟩` after seeing `x⟨1⟩`, into `h⟨2⟩` after seeing `x⟨1⟩` and `x⟨2⟩`, and so on — each hidden state is a running summary of everything seen so far.

It helps to think of an RNN in two equivalent views: a **compact, looped** diagram, where a single RNN "cell" feeds its own hidden state back into itself, and an **unfolded, through-time** diagram, where you draw a separate copy of the cell for each time step, connected by the flow of the hidden state. The critical insight of the unfolded view is that it is the *same* RNN cell — the same weights `W_hx`, `W_hh`, `W_yh` — reused at every single time step. This weight sharing across time is directly analogous to the weight sharing across space that made convolution so parameter-efficient in Week 3.

### Notation and equations

| Symbol | Meaning |
|---|---|
| `x⟨t⟩` | input at time step `t` |
| `h⟨t⟩` | hidden state (memory) after processing up to time `t` |
| `ŷ⟨t⟩` | prediction at time `t` |
| `h⟨0⟩` | initial hidden state, usually zeros |
| `T_x` | sequence length |
| `n_x`, `n_h`, `n_y` | dimensions of the input, hidden state, and output vectors |

Three weight matrices, shared across every time step, define the RNN: `W_hx` (`n_h × n_x`, maps the input into hidden-state space), `W_hh` (`n_h × n_h`, transforms the previous hidden state), and `W_yh` (`n_y × n_h`, maps the hidden state to an output), along with biases `b_h` and `b_y`. The forward pass at each step is:

```
h⟨t⟩ = tanh(W_hx · x⟨t⟩ + W_hh · h⟨t-1⟩ + b_h)          [update memory]
ŷ⟨t⟩ = σ(W_yh · h⟨t⟩ + b_y)                              [binary output]
ŷ⟨t⟩ = softmax(W_yh · h⟨t⟩ + b_y)                        [multi-class output]
```

The hidden-state equation is worth reading term by term: `W_hx · x⟨t⟩` processes what we're seeing right now, `W_hh · h⟨t-1⟩` brings in what we remember from before, the two are added together and passed through `tanh` (squashing the combined signal into `[-1, 1]`) to produce the new memory. In short: *new memory = a learned combination of "what we see now" and "what we remember."*

A small implementation trick worth knowing: `W_hx` and `W_hh` can be concatenated into a single matrix `W_h = [W_hh, W_hx]`, and `h⟨t-1⟩` and `x⟨t⟩` can be stacked into a single vector, so the two separate matrix multiplications collapse into one: `h⟨t⟩ = tanh(W_h · [h⟨t-1⟩; x⟨t⟩] + b_h)` — mathematically identical, but a single matrix multiplication instead of two.

### Worked example: named entity recognition

Consider labeling each word of "Martin Luther King gave a speech" as a person name (1) or not (0). At `t=1` ("Martin"), starting from `h⟨0⟩ = 0`, the network computes `z⟨1⟩ = W_hx·x⟨1⟩ + W_hh·h⟨0⟩ + b_h`, then `h⟨1⟩ = tanh(z⟨1⟩)`, then a prediction `ŷ⟨1⟩ = σ(W_yh·h⟨1⟩ + b_y) = 0.82` — close to the true label `y⟨1⟩=1`, giving a small binary cross-entropy loss of about 0.20. At `t=2` ("Luther"), the *same* weights are reused, but now `h⟨1⟩` (which already encodes "we just saw what looks like the start of a name") feeds into the update, so `h⟨2⟩` combines "Luther" with the memory of "Martin," producing `ŷ⟨2⟩ = 0.79`, again close to the true label 1. Summing the per-step losses over the whole sentence gives the total loss the network is trained to minimize.

## 3. Training an RNN: Backpropagation Through Time (BPTT)

The challenge in training an RNN is that the *same* weights are reused at every time step, so a single weight like `W_hh` influences the loss through many different paths — one for each time step. **Backpropagation Through Time (BPTT)** handles this by: unfolding the RNN across all time steps, computing the loss at each step, backpropagating through every step, and then **summing** the gradients contributed by each time step before finally updating the shared weights once: `∂L/∂W = ∂L_1/∂W + ∂L_2/∂W + ∂L_3/∂W + ...`.

Why summation? Because `W_hh` affects the final hidden state `h⟨3⟩` (say) through *multiple independent pathways*: a direct effect at `t=3`, an effect at `t=2` that then propagates forward to `t=3`, an effect at `t=1` that propagates through `t=2` and then `t=3`, and so on. The chain rule, applied to a variable that influences the output through several separate paths, says you add the contribution of every path — exactly like `∂h⟨3⟩/∂W_hh = (direct term at t=3) + (∂h⟨3⟩/∂h⟨2⟩)·(direct term at t=2) + (∂h⟨3⟩/∂h⟨2⟩)·(∂h⟨2⟩/∂h⟨1⟩)·(direct term at t=1) + ...`.

The building block that repeats in every one of these paths is `∂h⟨t⟩/∂h⟨t-1⟩`. Writing `a⟨t⟩ = W_hx·x⟨t⟩ + W_hh·h⟨t-1⟩ + b_h` (the pre-activation), the chain rule gives `∂h⟨t⟩/∂h⟨t-1⟩ = (∂h⟨t⟩/∂a⟨t⟩) · (∂a⟨t⟩/∂h⟨t-1⟩) = diag(1 - (h⟨t⟩)²) · W_hh^T` — the derivative of `tanh` (which is at most 1, and often much smaller) multiplied by the recurrent weight matrix.

## 4. The vanishing (and exploding) gradient problem in RNNs

To connect a gradient all the way from a late time step `T` back to an early one, this same `∂h⟨t⟩/∂h⟨t-1⟩` term gets multiplied together once per intervening step:

```
∂h_T/∂h_1 = Π_{t=2}^{T} diag(1 - (h⟨t⟩)²) · W_hh^T
```

Two things shrink this product. The `diag(1 - (h⟨t⟩)²)` factor is always between 0 and 1 (it's a `tanh` derivative), so multiplying many of them together shrinks the result — this is exactly the same mechanism as sigmoid's vanishing gradient from Week 2, just recurring over *time steps* instead of over *layers*. And if `W_hh`'s eigenvalues are also less than 1, that shrinks the product even further. The slides show this concretely: for a 50-step sequence, the gradient reaching back from `t=50` to `t=1` can shrink to about 0.005 of its original size — the network effectively has **no memory of early inputs** at that range.

The concrete consequence, worked out for the sentence "I grew up in France. I moved to Germany for work. After 10 years, I now speak fluent French," is that "France" at position 5 should influence the prediction of "French" at position 19 — 14 steps later. If each step shrinks the gradient by roughly 0.8, then after 14 steps the surviving gradient is `0.8^14 ≈ 0.044` — barely 4% of the original signal. The network receives almost no training signal telling it that "France" is relevant to predicting "French."

The opposite failure mode is the **exploding gradient problem**: if `W_hh`'s norm is greater than 1 (say 1.1), the same repeated multiplication instead *grows* the gradient exponentially — after 100 steps, a growth factor of roughly 13,780×. Symptoms include gradients becoming `NaN`, wild parameter updates, and training that diverges outright; the standard practical fix is **gradient clipping** — capping the gradient's magnitude at some maximum value before applying the update.

In summary, plain RNNs suffer from vanishing gradients (repeated multiplication by values below 1, which prevents learning long-range dependencies), exploding gradients (repeated multiplication by values above 1, causing instability), and, as a consequence of the first problem, an effective memory window of only around 5–10 time steps in practice, however long the "official" sequence is. This motivates a more sophisticated cell design: the GRU.

## 5. GRU: intelligent memory management with gates

The **Gated Recurrent Unit (GRU)** gives the network explicit, *learned* control over what to remember and what to forget, using two gates:

- **Update gate `z⟨t⟩`** decides how much of the old memory to keep versus how much to overwrite with new information: `z⟨t⟩ = σ(W_z · [h⟨t-1⟩, x⟨t⟩] + b_z)`. Since sigmoid outputs values in `(0,1)`, `z⟨t⟩ ≈ 1` means "keep the old memory, mostly ignore the new input," while `z⟨t⟩ ≈ 0` means "mostly replace old memory with new information."
- **Reset gate `r⟨t⟩`** decides how much of the past memory is even relevant when forming a new candidate memory right now: `r⟨t⟩ = σ(W_r · [h⟨t-1⟩, x⟨t⟩] + b_r)`. `r⟨t⟩ ≈ 1` means "use the full past memory when forming the new candidate," while `r⟨t⟩ ≈ 0` means "start mostly fresh, ignoring the past."

Think of it as a smart notebook: important pages get kept (update gate near 1), irrelevant pages get erased (reset gate near 0), and new information gets written in only where it's needed.

The **candidate hidden state** blends the (possibly reset-filtered) past with the current input, exactly like a plain RNN's update but using the *reset-gated* memory instead of the raw memory: `h̃⟨t⟩ = tanh(W_h · [r⟨t⟩ ⊙ h⟨t-1⟩, x⟨t⟩] + b_h)`, where `⊙` denotes element-wise multiplication. If `r⟨t⟩ ≈ 0`, the past is essentially ignored and the candidate is based mostly on the current input; if `r⟨t⟩ ≈ 1`, the full past memory is used.

Finally, the **new hidden state** is a *weighted average* between the old memory and the new candidate, controlled entirely by the update gate:

```
h⟨t⟩ = (1 - z⟨t⟩) ⊙ h̃⟨t⟩ + z⟨t⟩ ⊙ h⟨t-1⟩
```

If `z⟨t⟩ = 0.9`, the network keeps 90% of the old memory and blends in only 10% new information; if `z⟨t⟩ = 0.1`, it's the reverse. This is the single equation that gives GRU its power: **the network can choose, per dimension of the hidden state, to almost perfectly preserve information across many time steps by driving that dimension's update gate close to 1.**

### Why this actually fixes the vanishing gradient problem

Revisiting the "France...French" example: at `t=5` ("France"), suppose the update gate for a "country" dimension of the hidden state activates strongly, `z⟨5⟩ ≈ 0.95`. For the filler words between `t=6` and `t=18`, that same dimension keeps `z⟨t⟩ ≈ 0.95` at every step, so the country information decays only as `0.95^14 ≈ 0.49` — still a strong signal by the time "French" appears at `t=19`, in stark contrast to the plain RNN's `0.8^14 ≈ 0.044`. At `t=19`, the reset gate `r⟨19⟩ ≈ 0.9` lets the past be used freely, and the candidate combines "French" with the preserved memory of "France."

Mathematically, this is because the gradient path through the update gate is qualitatively different from a plain RNN's. In a plain RNN, `∂h⟨t⟩/∂h⟨t-1⟩ = diag(1-(h⟨t⟩)²) · W_hh^T` — a *purely multiplicative* path that has no way to avoid shrinking (or exploding). In a GRU, `∂h⟨t⟩/∂h⟨t-1⟩ ≈ diag(z⟨t⟩) + (additional terms from the reset gate and candidate)` — there's now an **additive**, gate-controlled path where, for any dimension where `z⟨t⟩ ≈ 1`, the corresponding entry of the derivative is close to 1 too, so the gradient for that dimension survives essentially undiminished across many steps: `∂h_i⟨T⟩/∂h_i⟨1⟩ ≈ Π z_i⟨t⟩ ≈ 1`. Dimensions where `z⟨t⟩ ≈ 0`, by contrast, are allowed to update and "forget" normally. The key structural point is that **different dimensions of the hidden state can have different, learned forgetting rates** — the network decides, per feature, what's worth preserving over the long run and what can be safely overwritten at each step.

## 6. RNN vs. GRU, and when to use each

| Feature | Plain RNN | GRU |
|---|---|---|
| Memory control | none (always overwritten) | gates control what to keep/forget |
| Gradient path | multiplicative only | additive + multiplicative |
| Long-range dependencies | fails after roughly 10 steps | can remember hundreds of steps |
| Vanishing gradients | essentially inevitable | preventable by learning `z ≈ 1` |
| Training stability | difficult on long sequences | much more stable |
| Parameters | fewer | somewhat more (extra gates) |

As a rule of thumb: use a plain RNN only for short sequences (fewer than about 10 steps) where long-range dependencies aren't critical, computational resources are very tight, or you specifically want a simple baseline. Use a GRU whenever sequences are long, long-range dependencies matter, you need stable training, and you have enough data to learn the extra gate parameters — which, in practice, describes most real sequence modeling problems.

## Key takeaways

Sequential data breaks the core assumptions behind feedforward networks — fixed size, no parameter sharing across positions, and no memory — so RNNs process a sequence one step at a time while carrying a hidden state forward as memory, reusing the *same* weights at every step (an idea directly parallel to convolution's weight sharing across space). Training uses backpropagation through time, summing gradient contributions from every time step, but this same mechanism causes gradients to shrink (or occasionally explode) exponentially with sequence length, because the recurrent path is purely multiplicative — a plain RNN effectively forgets anything more than about 10 steps in the past. The GRU fixes this by introducing an *update gate* that creates an additive, learnable gradient pathway: by driving a dimension's update gate close to 1, the network can choose to preserve that piece of information across arbitrarily many time steps, while a *reset gate* controls how much past context is used when forming new candidate memories. Next week's LSTM takes this gating idea even further, with separate short-term and long-term memory pathways.
