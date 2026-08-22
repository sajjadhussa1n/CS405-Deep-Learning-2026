# Lab 14 — Reinforcement Learning

**Matches:** [Week 15 — Reinforcement Learning](../lectures/week15-reinforcement-learning.md)
**Goal:** Reproduce the grid-world value/Q-value computations from the lecture by hand-coded backward induction, then replace the table with a small neural network (a minimal DQN).

## Setup

```bash
pip install numpy torch matplotlib
```

## Step 1 — Encode the grid world from the lecture

```python
import numpy as np

grid_size = 3
rewards = {
    (0,0): -1, (0,1): -1, (0,2): -1,
    (1,0): -1, (1,1): -2, (1,2): -1,
    (2,0): -1, (2,1): -2, (2,2): 100,
}
policy = {
    (0,0): (1,0), (1,0): (2,0), (2,0): (2,1),
    (2,1): (1,1), (1,1): (0,1), (0,1): (0,2),
    (0,2): (1,2), (1,2): (2,2),
}
terminal = (2, 2)
```

## Step 2 — Compute V^π(s) by backward induction

Implement the backward-induction procedure from the lecture (start at the terminal state, walk backward along the policy path) and confirm your computed values exactly match the table in the Week 15 notes: `(2,2)=100, (1,2)=99, (0,2)=98, (0,1)=97, (1,1)=95, (2,1)=93, (2,0)=92, (1,0)=91, (0,0)=90`.

```python
def compute_values(policy, rewards, terminal):
    # Build the reverse path from terminal back to start
    reverse_map = {v: k for k, v in policy.items()}
    values = {terminal: rewards[terminal]}
    current = terminal
    while current in reverse_map:
        prev_state = reverse_map[current]
        values[prev_state] = rewards[prev_state] + values[current]
        current = prev_state
    return values
```

## Step 3 — Compute Q^π(s, a) for the deterministic environment

For every state and every one of the four actions (Up, Down, Left, Right — clip moves that would leave the grid so they land back on the same cell), compute `Q^π(s,a) = r(s) + V^π(next_state)` using your Step 2 values. Reproduce the full table from the lecture notes (including the `argmax` action per state) and confirm it matches.

## Step 4 — Stochastic transitions

Extend Step 3 to a stochastic environment where the intended action succeeds with probability 0.7 and each of the other three directions occurs with probability 0.1 (matching the lecture's setup). Reproduce the `Q^π((0,0), Down) = 90.4` calculation from the lecture notes exactly, and then compute the full stochastic Q-table. Compare your deterministic and stochastic Q-values for at least 3 states and discuss which actions become relatively better or worse under uncertainty, and why.

## Step 5 — Replace the table with a small neural network (DQN)

Build the tiny DQN architecture described in the lecture (2 → 64 ReLU → 4) and train it, from random initial weights, to approximate the Q-values you computed by hand in Step 3/4 — treating your computed Q-table as supervised regression targets for now (this isolates "can a small network represent this Q-function" from "can Q-learning discover it through trial and error," which is a natural follow-up if you want to go further).

```python
import torch
import torch.nn as nn

class TinyDQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 4))

    def forward(self, state):
        return self.net(state)  # returns Q-values for [Up, Down, Left, Right]
```

Train with MSE loss against your hand-computed Q-table (one training example per `(state, action)` pair), and confirm the trained network's predicted Q-values closely match your table (plot predicted vs. true Q-value for every state-action pair as a scatter plot; points should lie close to the diagonal `y=x` line).

## Step 6 (optional, for the ambitious) — Actual Q-learning

Rather than regressing directly onto pre-computed Q-values, implement the standard tabular or DQN-style **Q-learning** update rule, where the agent actually explores the grid world (with an epsilon-greedy policy) and updates its Q-estimates online using the Bellman equation as a bootstrapped target: `Q(s,a) ← Q(s,a) + α[r + γ·max_a' Q(s',a') - Q(s,a)]`. Confirm that, after enough episodes, your learned Q-table (or trained DQN) converges close to the values you computed analytically in Steps 2–4.

## Checkpoint questions

1. In Step 4, which action(s) changed their relative ranking (i.e., which action was `argmax` in the deterministic case vs. the stochastic case) for any state? Explain the change in terms of the *distribution* of possible outcomes, not just the intended one.
2. In Step 5, how many hand-computed `(state, action, Q-value)` training examples did you have in total, for this 3×3 grid with 4 actions? Given how few examples that is, why might a neural network still work reasonably well here, compared to a much larger/continuous state space?
3. If you attempted Step 6: how many training episodes did it take before your learned Q-values started closely matching the analytically-computed ones from Step 3? What does this tell you about the practical cost of learning through trial and error (Q-learning) versus having a known model of the environment to compute values directly (as in Steps 2–4)?
