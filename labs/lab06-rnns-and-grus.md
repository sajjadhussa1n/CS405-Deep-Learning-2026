# Lab 6 — RNNs and GRUs

**Matches:** [Week 6 — Sequential Models: From RNNs to GRUs](../lectures/week06-sequential-models-rnn-gru.md)
**Goal:** Implement a plain RNN cell and a GRU cell from scratch, and empirically observe the vanishing-gradient gap between them.

## Setup

```bash
pip install torch matplotlib
```

## Step 1 — A plain RNN cell from scratch

```python
import torch
import torch.nn as nn

class MyRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.Whx = nn.Linear(input_size, hidden_size, bias=False)
        self.Whh = nn.Linear(hidden_size, hidden_size)

    def forward(self, x_t, h_prev):
        return torch.tanh(self.Whx(x_t) + self.Whh(h_prev))
```

Wrap it in a small loop over a sequence to confirm it produces a hidden state of the right shape at every time step:

```python
cell = MyRNNCell(input_size=8, hidden_size=16)
h = torch.zeros(1, 16)
for t in range(5):
    x_t = torch.randn(1, 8)
    h = cell(x_t, h)
print(h.shape)  # (1, 16)
```

## Step 2 — A GRU cell from scratch

```python
class MyGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.Wz = nn.Linear(input_size + hidden_size, hidden_size)
        self.Wr = nn.Linear(input_size + hidden_size, hidden_size)
        self.Wh = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x_t, h_prev):
        combined = torch.cat([h_prev, x_t], dim=-1)
        z = torch.sigmoid(self.Wz(combined))
        r = torch.sigmoid(self.Wr(combined))
        combined_reset = torch.cat([r * h_prev, x_t], dim=-1)
        h_candidate = torch.tanh(self.Wh(combined_reset))
        h_new = (1 - z) * h_candidate + z * h_prev
        return h_new
```

Confirm it runs the same shape-check as Step 1.

## Step 3 — A synthetic long-range-dependency task

Build a dataset where the label depends only on the *first* token of a sequence, with everything else being irrelevant "noise" tokens (e.g., a sequence of random integers 0–9, where the label is 1 if the first token is even, 0 if odd — the network must remember the very first token all the way to the end):

```python
import random

def make_example(seq_len, vocab_size=10):
    seq = [random.randrange(vocab_size) for _ in range(seq_len)]
    label = seq[0] % 2
    return seq, label
```

## Step 4 — Train plain-RNN vs. GRU classifiers, at increasing sequence length

For each of `seq_len in [5, 15, 30, 60]`, train a small classifier (embedding → your `MyRNNCell` or `MyGRUCell` run over the sequence → linear classifier on the final hidden state) on a training set of a few thousand generated examples, for a fixed small number of epochs. Record final test accuracy for both cell types at each sequence length, and plot accuracy vs. sequence length as two lines (RNN, GRU) on one graph.

You should see the plain RNN's accuracy degrade toward chance (50%) as sequence length grows, while the GRU holds up much better — a direct, empirical version of the "France...French" example from the lecture notes.

## Step 5 — Measure the gradient reaching the first time step

After one backward pass on a batch with `seq_len=60`, inspect the gradient with respect to the *first* input embedding (`x_t.grad` for `t=0`, which requires setting `requires_grad_(True)` on that input) versus the *last* (`t=59`). Compute the ratio `|grad_first| / |grad_last|` for both your RNN-based and GRU-based classifiers, and plot this ratio across a few sequence lengths.

## Checkpoint questions

1. At what sequence length does the plain RNN's accuracy in Step 4 start to visibly degrade? Does this roughly match the "effective memory window of about 5–10 steps" claim from the lecture notes?
2. In Step 5, does the gradient ratio you measured for the GRU stay closer to 1 (undiminished) than the RNN's, across all tested sequence lengths? If not, what might explain the discrepancy (e.g., untuned initialization, the update gate not learning to open fully)?
3. What happens to your GRU's performance in Step 4 if you initialize the update-gate bias to a large positive value (encouraging `z≈1`, i.e., "remember by default") at the start of training? Try it and discuss.
