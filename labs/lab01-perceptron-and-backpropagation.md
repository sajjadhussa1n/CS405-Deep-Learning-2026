# Lab 1 — Perceptron and Backpropagation

**Matches:** [Week 1 — Introduction to Neural Networks](../lectures/week01-introduction-to-neural-networks.md)
**Goal:** Implement a perceptron and a tiny multi-layer network with backpropagation using only NumPy, and see the XOR problem fail and then succeed.

## Setup

```bash
pip install numpy matplotlib scikit-learn
```

## Step 1 — A single perceptron

Implement the perceptron learning rule from the lecture notes:

```python
import numpy as np

def step(z):
    return (z >= 0).astype(int)

class Perceptron:
    def __init__(self, n_inputs, lr=0.1):
        self.w = np.random.randn(n_inputs) * 0.01
        self.b = 0.0
        self.lr = lr

    def predict(self, x):
        return step(x @ self.w + self.b)

    def train_step(self, x, y_true):
        y_pred = self.predict(x)
        error = y_true - y_pred
        self.w += self.lr * error * x
        self.b += self.lr * error
        return error
```

Train it on AND and OR (4 examples each, 2 inputs). Confirm it converges to 100% accuracy within a few dozen epochs for both.

## Step 2 — Watch it fail on XOR

Train the same `Perceptron` on the XOR truth table for several hundred epochs. Plot accuracy vs. epoch. You should see it plateau well below 100% and never improve — exactly the limitation described in Week 1. Plot the four XOR points colored by class and try to draw a single straight line separating them by hand — you won't be able to.

## Step 3 — A 2-layer network that solves XOR

Implement a minimal 2-input → 2-hidden (sigmoid) → 1-output (sigmoid) network with manual forward and backward passes:

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_deriv(a):
    return a * (1 - a)  # a is already sigmoid(z)

class TinyMLP:
    def __init__(self):
        self.W1 = np.random.randn(2, 2) * 0.5
        self.b1 = np.zeros(2)
        self.W2 = np.random.randn(2, 1) * 0.5
        self.b2 = np.zeros(1)

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, lr=0.5):
        m = X.shape[0]
        dz2 = (self.a2 - y.reshape(-1, 1))  # d(MSE)/dz2 with sigmoid output
        dW2 = self.a1.T @ dz2 / m
        db2 = dz2.mean(axis=0)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * sigmoid_deriv(self.a1)
        dW1 = X.T @ dz1 / m
        db1 = dz1.mean(axis=0)
        self.W1 -= lr * dW1; self.b1 -= lr * db1
        self.W2 -= lr * dW2; self.b2 -= lr * db2
```

Train it on XOR for a few thousand epochs, plotting loss over time. Confirm it reaches near-zero loss.

## Step 4 — Visualize the decision boundary

Using `matplotlib.pyplot.contourf` over a grid of points in `[-0.5, 1.5]²`, plot your trained `TinyMLP`'s predicted class as a filled contour, with the 4 XOR points overlaid. You should see a non-linear (curved) decision boundary that correctly separates the classes — visual proof of Week 1's claim that a hidden layer with non-linear activations can do what a single perceptron cannot.

## Checkpoint questions

1. Why does the perceptron's accuracy on XOR plateau instead of slowly improving? What does that plateau tell you about the loss landscape it's stuck in?
2. In `backward()`, why is `dz2 = self.a2 - y` (rather than something involving a separate sigmoid derivative term) when using MSE loss with a sigmoid output? (Hint: work out `d(MSE)/da2 · da2/dz2` by hand.)
3. What happens to your `TinyMLP`'s training if you remove the `sigmoid` from the hidden layer and use a linear activation instead? Try it and connect the result to the "deep linear network collapses to one layer" argument in the Week 1 notes.
