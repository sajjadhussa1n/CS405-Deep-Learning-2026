# Lab 2 — Training Deep Networks

**Matches:** [Week 2 — Training Neural Networks: A Deep Dive](../lectures/week02-training-neural-networks.md)
**Goal:** Feel, with your own hands and plots, why vanishing gradients happen and how initialization, batch norm, optimizers, and regularization fix training problems.

## Setup

```bash
pip install torch torchvision matplotlib
```

## Step 1 — Build a deep, fragile network

```python
import torch
import torch.nn as nn

class DeepNet(nn.Module):
    def __init__(self, depth=10, width=64, activation=nn.Sigmoid, use_bn=False):
        super().__init__()
        layers = []
        in_dim = 28 * 28
        for i in range(depth):
            layers.append(nn.Linear(in_dim, width))
            if use_bn:
                layers.append(nn.BatchNorm1d(width))
            layers.append(activation())
            in_dim = width
        layers.append(nn.Linear(width, 10))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))
```

Load Fashion-MNIST (`torchvision.datasets.FashionMNIST`) with a standard `DataLoader`.

## Step 2 — Reproduce vanishing gradients

Train a `DeepNet(depth=10, activation=nn.Sigmoid)` for a handful of batches. After each backward pass, record the mean absolute gradient of the first linear layer's weight versus the last linear layer's weight:

```python
first_grad = model.net[0].weight.grad.abs().mean().item()
last_grad = model.net[-1].weight.grad.abs().mean().item()
```

Plot both over training steps. You should see the first layer's gradient sitting far below the last layer's — the vanishing gradient problem from Week 2, caused by repeatedly multiplying by sigmoid derivatives (max 0.25) through the chain rule.

## Step 3 — Swap in ReLU

Repeat Step 2 with `activation=nn.ReLU`. Compare the first-layer vs. last-layer gradient plot. ReLU's derivative is exactly 1 for active units, so the gap should shrink substantially (though not disappear completely, since dead units still contribute exactly 0).

## Step 4 — Add proper initialization

By default, `nn.Linear` uses a Kaiming (He)-style initialization already, but confirm you understand the theory: manually re-initialize a `DeepNet`'s weights with `nn.init.xavier_normal_` and compare against `nn.init.kaiming_normal_` (matched to ReLU) — measure the standard deviation of the activations at each layer during a single forward pass with each initialization scheme, and relate what you see to the Week 2 discussion of why He initialization compensates for ReLU zeroing out half its inputs.

## Step 5 — Add BatchNorm

Train `DeepNet(depth=10, activation=nn.ReLU, use_bn=True)` and compare the training loss curve (first 200 steps) against the same network without BatchNorm. You should see faster, smoother convergence with BatchNorm.

## Step 6 — Compare optimizers

Using your best network configuration from Steps 3–5, train three separate copies with `torch.optim.SGD`, `torch.optim.SGD(momentum=0.9)`, and `torch.optim.Adam`, all with a comparable learning rate, for the same number of steps. Plot all three loss curves on one figure.

## Step 7 — Regularization

Add `weight_decay=1e-4` to your Adam optimizer (this is L2 regularization, Week 2) and separately try adding `nn.Dropout(p=0.3)` between hidden layers. For each, train for enough epochs to start overfitting on a small subset of the training data (e.g., 2,000 examples), and plot training vs. validation accuracy over epochs with and without each regularizer.

## Checkpoint questions

1. In Step 2, does the gap between first-layer and last-layer gradients grow or shrink as you increase `depth`? Why?
2. In Step 5, BatchNorm is described in the lecture as also having a mild *regularizing* effect. Design a small experiment (varying batch size) that tests whether this holds up empirically for your network.
3. Which optimizer in Step 6 converged fastest? Does this match the theoretical intuition from the lecture notes about momentum accumulating gradients in consistent directions?
