# Week 15 — Reinforcement Learning

*Companion notes for [`slides/lecture_week15.pdf`](../slides/lecture_week15.pdf)*

## Why this week matters

Every model in this course so far has been trained with **supervised** learning: given the right answer for each training example, adjust the network to reproduce it. Reinforcement Learning (RL) is a fundamentally different paradigm — there is no "right answer" provided at each step, only a sparse, delayed **reward** signal, and the model (called an **agent**) has to learn, through trial and error, which sequences of decisions lead to good long-term outcomes. This closing week introduces the core vocabulary and mathematics of RL, and shows, at the end, how the deep learning techniques from the rest of the course (neural networks as function approximators) connect directly into it — the combination is known as **Deep Reinforcement Learning**.

## 1. How RL differs from supervised learning

| Aspect | Supervised learning | Reinforcement learning |
|---|---|---|
| Data | (input, label) pairs | (state, action, reward) |
| Feedback | immediate, correct label given | delayed, scalar reward only |
| Goal | predict the correct label | maximize cumulative reward |
| Interaction with environment | no | yes (agent ↔ environment) |

The slides offer a nice analogy: supervised learning is like teaching with flashcards — you're shown the input and immediately told the correct answer. Reinforcement learning is more like training a dog — you don't tell it exactly which muscle to move; you give it a treat (reward) when it does something good, and it has to figure out, through repeated interaction with its environment, what sequence of actions tends to earn treats over time. There's no ground-truth label being handed out at every step — only feedback in the form of a scalar reward, often only available well after the actions that actually caused it.

## 2. The RL loop

RL is framed as a continuous loop of interaction between an **agent** (the learner/decision-maker) and its **environment** (everything outside the agent). At each time step, the agent observes the environment's current **state** `s` and chooses an **action** `a`; the environment responds with a new state and an immediate **reward** `r` (a scalar signal indicating how good or bad that step was). This loop — agent takes action, environment returns new state and reward, repeat — is the basic unit of every RL problem, whatever the specific domain (a game, a robot, a recommendation system, and so on).

### The discount factor

Because rewards can arrive over a long horizon, RL defines the quantity an agent actually cares about — the **return** `G_t` — as the sum of *all* future rewards from time `t` onward, but with each future reward **discounted** by how far away it is:

```
G_t = r_{t+1} + γ·r_{t+2} + γ²·r_{t+3} + ... = Σ_{k=0}^∞ γ^k · r_{t+k+1}
```

The **discount factor** `γ ∈ [0, 1]` controls how "impatient" the agent is: a `γ` close to 0 makes the agent almost entirely short-sighted, essentially ignoring rewards more than a step or two into the future; a `γ` close to 1 makes the agent nearly as concerned about rewards far in the future as about immediate ones. Discounting also reflects a natural intuition that far-future rewards are inherently less certain than immediate ones.

### Policy

A **policy** `π` defines the agent's behavior — it's the (possibly probabilistic) rule mapping states to actions. A **deterministic** policy always picks the same action in a given state, `π(s) = a`. A **stochastic** policy instead assigns a probability to each possible action, `π(a|s)` = the probability of taking action `a` in state `s` — for example, `π(a_1|s) = 0.7, π(a_2|s) = 0.3`. Formally, the agent's entire objective in RL is to find the policy that maximizes its expected discounted return.

## 3. The state-value function

The **state-value function** `V^π(s)` answers the question "how good is it to be in state `s`, if I then follow policy `π` from here on?" — formally, the expected discounted return starting from state `s` and following `π` thereafter: `V^π(s) = E_π[G_t | S_t = s]`.

### Worked example: computing V^π on a grid world

Consider a 3×3 grid, with a prize of `+100` at cell `(2,2)`, a normal step penalty of `-1` for most cells, and two special "uphill" cells, `(1,1)` and `(2,1)`, which have a harsher penalty of `-2`. A fixed, deterministic policy walks the path `(0,0) → (1,0) → (2,0) → (2,1) → (1,1) → (0,1) → (0,2) → (1,2) → (2,2)`, and we use no discounting (`γ=1`) for simplicity.

`V^π(s)` for this policy is computed by **backward induction**: start at the terminal state and work backward, where each state's value equals its own immediate reward plus the value of whatever state the policy sends it to next. At the terminal state, `V^π(2,2) = 100` (just the prize, no future steps remain). Working backward: `V^π(1,2) = -1 + V^π(2,2) = -1+100 = 99`; `V^π(0,2) = -1+99 = 98`; `V^π(0,1) = -1+98 = 97`; `V^π(1,1) = -2+97 = 95` (note the harsher `-2` penalty for this uphill cell); `V^π(2,1) = -2+95 = 93`; `V^π(2,0) = -1+93 = 92`; `V^π(1,0) = -1+92 = 91`; and finally the start state, `V^π(0,0) = -1+91 = 90`. The complete table of values, laid out on the grid, decreases steadily as you move away from the prize, with an extra dip at the two uphill cells — exactly matching intuition: states that are farther from the goal, or that pass through costlier terrain, are worth less.

### The Bellman equation

The pattern used in every one of those backward-induction steps — *this state's value equals its own reward plus the value of the next state* — is a special (deterministic, undiscounted) case of the general **Bellman equation** for the state-value function:

```
V^π(s) = r(s, π(s)) + γ · V^π(π(s))
```

This recursive, self-consistent relationship — a state's value is defined in terms of the *next* state's value — is the mathematical foundation that essentially every RL algorithm builds on, whether solved exactly (as in this small grid-world example) or estimated iteratively via learning in larger problems.

## 4. The state-action value function (Q-function)

`V^π(s)` tells you how good a *state* is, but an agent making decisions really needs to know how good each available *action* is in that state. The **state-action value function**, `Q^π(s,a)`, answers exactly that: the expected discounted return if the agent takes action `a` in state `s` and then follows policy `π` thereafter: `Q^π(s,a) = E_π[G_t | S_t=s, A_t=a]`. This is directly useful for decision-making because the agent can compare `Q(s,a)` across every available action in a state and simply pick the best one: `argmax_a Q(s,a)`. In fact, once you have the *optimal* Q-function `Q*`, the optimal policy falls straight out of it: `π*(s) = argmax_a Q*(s,a)` — no separate policy representation is even needed.

Q also satisfies its own recursive Bellman equation, following the same logic as before — take action `a`, get an immediate reward, land in a next state `s'`, then continue following `π`:

```
Q^π(s,a) = r(s,a) + γ · Q^π(s', π(s'))
```

### Worked example: computing Q^π on the same grid

Given the `V^π(s)` table already computed above, `Q^π(s,a) = r(s) + γ·V^π(s')`, where `s'` is whatever state results from taking action `a` in state `s`. For example, to compute `Q^π((0,0), Right)`: taking "Right" from `(0,0)` lands the agent at `(0,1)`, the immediate reward at `(0,0)` is `-1`, and `V^π(0,1) = 97`, so `Q^π((0,0), Right) = -1 + 1×97 = 96`. Repeating this for every state and every one of the four actions (Left, Right, Up, Down — with "moving out of the grid" simply leaving the agent in the same cell) produces a full Q-table, and taking the `argmax` across actions in each row recovers the policy that would actually be optimal for this environment — which, in this particular grid, turns out to favor "Right" or "Down" moves that steer around the costly uphill cells wherever possible.

## 5. Beyond deterministic environments: Markov Decision Processes

Real environments are rarely perfectly deterministic — a robot's wheel might slip, wind might affect movement, sensors are noisy. To capture this, RL problems are formally modeled as **Markov Decision Processes (MDPs)**, defined by the tuple `(S, A, P, R, γ)`: a set of states `S`, a set of actions `A`, a **transition probability** `P(s'|s,a)` (the probability of landing in state `s'` after taking action `a` in state `s`), a reward function `R(s,a,s')` (or `R(s,a)`), and a discount factor `γ`.

The defining assumption is the **Markov property**: `P(S_{t+1} | S_t, A_t, S_{t-1}, A_{t-1}, ...) = P(S_{t+1} | S_t, A_t)` — informally, "given the present, the future is independent of the past." Everything relevant to predicting what happens next is assumed to be captured in the *current* state; there's no need to remember the entire history that led there.

### Worked example: Q-values under a stochastic transition model

Extending the grid-world example: suppose the intended action now only succeeds with probability `p=0.7`, and with probability `0.1` each, the agent instead "slips" into one of the three other directions (with any out-of-bounds move simply leaving the agent in place). The Q-value formula now needs to sum over *all* possible resulting states, weighted by their transition probabilities:

```
Q^π(s,a) = r(s) + Σ_{s'} P(s'|s,a) · V^π(s')
```

Computing `Q^π((0,0), Down)`: the intended "Down" move succeeds with probability 0.7, landing at `(1,0)` with `V^π=91`; a "Left" slip (probability 0.1) leaves the agent at `(0,0)` (out of bounds) with `V^π=90`; a "Right" slip (probability 0.1) lands at `(0,1)` with `V^π=97`; an "Up" slip (probability 0.1) leaves it at `(0,0)` (out of bounds) with `V^π=90`. Plugging in:

```
Q^π((0,0), Down) = -1 + [0.7×91 + 0.1×90 + 0.1×97 + 0.1×90]
                  = -1 + [63.7 + 9.0 + 9.7 + 9.0] = -1 + 91.4 = 90.4
```

Comparing deterministic and stochastic Q-values for `(0,0)` reveals an important qualitative effect: stochasticity tends to *smooth* the Q-values toward the mean — extreme, "lucky" outcomes get diluted by the possibility of slipping elsewhere. "Down" became slightly *better* under uncertainty here (there's a chance of accidentally slipping into the good state `(0,1)`), while "Right" became noticeably *worse* (there's now real risk of slipping into a worse state instead of reliably reaching the good one). The general lesson: in a genuine MDP, an action's Q-value depends on the *entire distribution* of possible outcomes, not just the single outcome you'd expect under a deterministic model — an agent has to reason about risk, not just the "intended" result of an action.

## 6. From tables to neural networks: Deep Q-Networks

Everything above worked by explicitly building and filling in a table of `V(s)` or `Q(s,a)` values, one entry per state (or state-action pair). This approach breaks down completely for any realistically large or continuous problem, for two related reasons: **state explosion** (an Atari game screen at `210×160` pixels with 256 possible values per pixel has an astronomically large number of possible states — `256^(210×160)` — utterly impossible to enumerate in a table; a robot arm with continuous joint angles has literally infinitely many states) and **no generalization** (a lookup table has no way to produce a sensible Q-value for a state it has never exactly seen before, even if that state is nearly identical to one it has seen — e.g., the coordinates `(2.1, 3.9)`).

The fix connects directly back to everything else in this course: replace the table with a **function approximator** — specifically, a neural network — that takes a state (and optionally an action) as input and outputs an approximate Q-value: `Q(s,a;θ) ≈ Q*(s,a)`, where `θ` is a comparatively small set of learnable network parameters. This is the essence of a **Deep Q-Network (DQN)**. Using a neural network instead of a table brings exactly the benefits you'd expect from the rest of the course: it handles continuous state spaces naturally (no need to discretize), it **generalizes** — similar states tend to produce similar Q-value estimates automatically, since that's precisely what a trained network does — and its number of parameters is vastly smaller than the number of possible states, making it tractable for problems where a table simply could never exist.

For the grid-world example, a simple DQN might take the `(x,y)` coordinates as a 2-dimensional input, pass them through a hidden layer of 64 ReLU-activated neurons (exactly the feedforward architecture from Week 1), and output 4 values — one Q-value estimate for each of the four possible actions (Up, Down, Left, Right). After training (using the Bellman equation as the target for a regression-style loss, updated via ordinary backpropagation and gradient descent, exactly as in every other network in this course), the network learns to approximate the optimal Q-values across the entire state space, without ever storing a single explicit table entry — the same core training machinery from Week 1 (forward pass, loss, backward pass, gradient descent) applied to an entirely different kind of learning problem.

## Key takeaways

Reinforcement learning departs from the supervised paradigm used everywhere else in this course: instead of learning from labeled (input, correct-output) pairs, an agent learns by interacting with an environment and receiving delayed, scalar rewards, with the goal of maximizing cumulative discounted return, `G_t = Σ γ^k r_{t+k+1}`. The state-value function `V^π(s)` and state-action value function `Q^π(s,a)` formalize "how good is this state/action," and both satisfy a recursive Bellman equation — the current value equals the immediate reward plus the (discounted) value of what comes next — which can be solved directly by backward induction in small, deterministic problems, and extended to weighted sums over transition probabilities in stochastic Markov Decision Processes. Real-world problems have far too many states (or continuous state spaces) to represent as an explicit table, which is where deep learning re-enters the picture: a neural network trained to approximate the Q-function (a Deep Q-Network) generalizes across similar states using a compact set of learned parameters, in exactly the same forward-pass/loss/backward-pass/gradient-descent training loop introduced all the way back in Week 1 — closing the loop on the course by showing that the fundamental training machinery you learned first turns out to power an entirely different class of learning problems as well.
