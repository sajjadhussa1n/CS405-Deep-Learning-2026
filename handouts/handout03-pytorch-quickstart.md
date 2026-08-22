# Handout — PyTorch Quickstart

A minimal, practical introduction to the PyTorch features you'll actually use across the [`labs/`](../labs/) and [`assignments/`](../assignments/) in this course. This is not a complete PyTorch tutorial — it's the specific subset that shows up repeatedly, explained once so every later handout, lab, and assignment can assume you've seen it.

## Installation

```bash
pip install torch torchvision
```

(For GPU support, follow the install command generator at pytorch.org for your specific CUDA version — the plain `pip install torch` above gives you a CPU-only build, which is fine for every small example in this course's labs.)

## Tensors

A `torch.Tensor` is PyTorch's version of a NumPy array — an n-dimensional array with a specific data type, that can additionally live on a GPU and track gradients.

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0])
y = torch.zeros(3, 4)          # 3x4 tensor of zeros
z = torch.randn(2, 3)          # 2x3 tensor of standard-normal random values
w = torch.tensor([[1., 2.], [3., 4.]])

print(x.shape, y.shape, z.shape)
print(w @ w)                   # matrix multiplication
print(x.sum(), x.mean())
```

Moving between NumPy and PyTorch: `torch.from_numpy(np_array)` and `tensor.numpy()`.

## Autograd: automatic differentiation

This is the single most important PyTorch feature for this course — it computes the backpropagation gradients from Week 1 and the [Backpropagation Derivation handout](handout02-backpropagation-derivation.md) *automatically*, so you never have to hand-derive `∂L/∂w` for a real model.

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x + 1
y.backward()          # computes dy/dx and stores it in x.grad
print(x.grad)          # tensor(7.)  (dy/dx = 2x + 3, at x=2 -> 7)
```

For any tensor you want a gradient with respect to, set `requires_grad=True` (parameters created via `nn.Module`, covered next, get this automatically). Calling `.backward()` on a scalar output walks the computational graph *backward*, applying the chain rule at every operation — exactly the process derived by hand in the backpropagation handout, just automated.

**Important gotcha:** gradients *accumulate* by default. Always call `optimizer.zero_grad()` (or `tensor.grad = None`) before each new `.backward()` call in a training loop, or gradients from previous steps will incorrectly add onto the current step's gradients.

## Building a model with `nn.Module`

```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = MLP(input_dim=10, hidden_dim=32, output_dim=1)
```

Every layer used across the course's lecture notes has a direct `nn` counterpart: `nn.Linear` (a fully connected layer, Week 1), `nn.Conv2d`/`nn.MaxPool2d` (Week 3), `nn.BatchNorm1d`/`nn.BatchNorm2d` (Week 2), `nn.RNN`/`nn.GRU`/`nn.LSTM` (Weeks 6–7), `nn.Dropout` (Week 2), `nn.MultiheadAttention`/`nn.TransformerEncoderLayer` (Week 13). Several labs ask you to implement a layer type *yourself* (from raw tensor operations) precisely so you understand what these built-in layers are doing internally before you rely on them.

## Loss functions and optimizers

```python
import torch.optim as optim

criterion = nn.MSELoss()          # or nn.CrossEntropyLoss(), nn.BCEWithLogitsLoss(), etc.
optimizer = optim.Adam(model.parameters(), lr=1e-3)   # or optim.SGD(..., momentum=0.9)
```

`nn.CrossEntropyLoss` expects raw, unnormalized logits (it applies `softmax` and `log` internally) and integer class-index targets — a common source of bugs is applying softmax yourself before passing predictions into it, which double-applies the operation.

## The standard training loop

Nearly every lab and assignment in this course uses some variant of this loop:

```python
for epoch in range(num_epochs):
    model.train()
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()               # 1. clear old gradients
        predictions = model(batch_x)        # 2. forward pass
        loss = criterion(predictions, batch_y)
        loss.backward()                     # 3. backward pass (autograd)
        optimizer.step()                    # 4. update weights (gradient descent)

    model.eval()
    with torch.no_grad():                   # disable gradient tracking for evaluation
        val_loss = sum(criterion(model(x), y).item() for x, y in val_loader)
    print(f"Epoch {epoch}: val_loss={val_loss:.4f}")
```

This is the four-step training cycle from the [Week 1 lecture notes](../lectures/week01-introduction-to-neural-networks.md) — forward pass, loss, backward pass, weight update — written out in PyTorch. `model.train()` and `model.eval()` matter specifically for layers that behave differently at train vs. test time, like `nn.Dropout` and `nn.BatchNorm*` (Week 2) — always set the correct mode before running a batch through the model.

## Datasets and DataLoaders

```python
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(MyDataset(X_train, y_train), batch_size=32, shuffle=True)
```

`torchvision.datasets` provides ready-made `Dataset` classes for common image datasets (`MNIST`, `FashionMNIST`, `CIFAR10`, etc.) used throughout the labs; `datasets` (from Hugging Face, Week 14) plays the same role for common NLP datasets.

## GPU usage

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
batch_x, batch_y = batch_x.to(device), batch_y.to(device)
```

Every tensor involved in a computation must be on the *same* device — a common error message ("Expected all tensors to be on the same device") almost always means you forgot to `.to(device)` one of your inputs or your model.

## Saving and loading models

```python
torch.save(model.state_dict(), "model.pt")

model = MLP(input_dim=10, hidden_dim=32, output_dim=1)
model.load_state_dict(torch.load("model.pt"))
```

## Where to go next

This handout covers everything needed to get through [Lab 1](../labs/lab01-perceptron-and-backpropagation.md) through roughly [Lab 5](../labs/lab05-object-detection-segmentation.md). Later labs introduce a few additional PyTorch pieces (e.g., `nn.Embedding` in Week 11's labs, `torch.nn.functional.softmax`/masking for Week 13's attention labs) directly in place, since they're specific to those topics rather than general-purpose.
