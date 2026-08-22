# Lab 7 — LSTM

**Matches:** [Week 7 — Long Short-Term Memory (LSTM)](../lectures/week07-lstm.md)
**Goal:** Implement an LSTM cell from scratch, inspect its gate activations directly, and compare it against your Lab 6 RNN/GRU on the same task.

## Setup

```bash
pip install torch matplotlib
```

## Step 1 — An LSTM cell from scratch

Implement the four LSTM equations exactly as given in the lecture notes:

```python
import torch
import torch.nn as nn

class MyLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        combined_size = input_size + hidden_size
        self.Wf = nn.Linear(combined_size, hidden_size)  # forget gate
        self.Wi = nn.Linear(combined_size, hidden_size)  # input gate
        self.Wc = nn.Linear(combined_size, hidden_size)  # candidate
        self.Wo = nn.Linear(combined_size, hidden_size)  # output gate

    def forward(self, x_t, state):
        h_prev, c_prev = state
        combined = torch.cat([h_prev, x_t], dim=-1)
        f = torch.sigmoid(self.Wf(combined))
        i = torch.sigmoid(self.Wi(combined))
        c_candidate = torch.tanh(self.Wc(combined))
        o = torch.sigmoid(self.Wo(combined))
        c_new = f * c_prev + i * c_candidate
        h_new = o * torch.tanh(c_new)
        return h_new, c_new
```

Confirm shapes with a quick smoke test, tracking both `h` and `c` across a short sequence.

## Step 2 — Visualize gate activations over a sequence

Feed a single hand-constructed sequence through your `MyLSTMCell` (e.g., 20 random input steps) and, at every time step, record the *mean* value (averaged across the hidden dimension) of `f`, `i`, and `o`. Plot all three as separate lines over time. Even with random, untrained weights, you should see the gates producing values spread across `(0,1)` — after training (Step 4), revisit this plot and see whether the forget gate values shift toward the extremes (closer to 0 or 1) as the network learns to be more decisive about what to keep.

## Step 3 — The same long-range task as Lab 6

Reuse the synthetic "remember the first token" dataset from [Lab 6](lab06-rnns-and-grus.md). Build an LSTM-based classifier (embedding → `MyLSTMCell` run over the sequence → linear head on the final `h`) and train/evaluate it at the same sequence lengths (`[5, 15, 30, 60]`) used in Lab 6.

## Step 4 — Three-way comparison

Combine your RNN, GRU (Lab 6), and LSTM (this lab) accuracy-vs-sequence-length results onto a single plot. Also combine the "gradient reaching the first time step" measurements from Lab 6, Step 5, adding the LSTM's cell-state gradient (`c_0.grad` after a backward pass, if you make the initial cell state a leaf tensor with `requires_grad_(True)`).

## Step 5 — Revisit the trained gate-activation plot

Re-run Step 2's gate-activation visualization using your now-**trained** LSTM from Step 3 (at `seq_len=60`), feeding it a real test example. Compare the forget-gate trace before and after training. Discuss whether the forget gate appears to have learned to stay close to 1 for the dimension(s) carrying the "remember the first token" signal.

## Checkpoint questions

1. Does your trained LSTM's accuracy in Step 3 clearly exceed the GRU's at the longest sequence length (60), or are they roughly comparable? The lecture notes suggest LSTM's advantage is most visible on *very* long sequences and when fine-grained control is needed — does a length of 60 seem long enough to show a clear gap in your experiment?
2. In Step 5, what would you expect to see in the forget-gate trace of a *well-trained* LSTM that has successfully learned this task, at the specific hidden-state dimension(s) responsible for remembering the first token? Does your plot match that expectation?
3. LSTM has more parameters than GRU for the same hidden size (four gate matrices vs. two gate matrices plus a candidate). Measure the actual parameter count of your `MyLSTMCell` vs. `MyGRUCell` from Lab 6 for the same `hidden_size`, and confirm it roughly matches the `4n_h² + ...` vs. `3n_h² + ...` scaling mentioned in the lecture.
