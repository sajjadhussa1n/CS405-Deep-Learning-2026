# Lab 12 — Transformers

**Matches:** [Week 13 — Transformer: Attention Is All You Need](../lectures/week13-transformers.md)
**Goal:** Implement scaled dot-product and multi-head attention from scratch, add positional encoding, and assemble a minimal Transformer encoder.

## Setup

```bash
pip install torch matplotlib
```

## Step 1 — Scaled dot-product attention, verified by hand

```python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V):
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    weights = F.softmax(scores, dim=-1)
    return weights @ V, weights
```

Construct a tiny hand-designed example: 3 tokens, `d_k=2`. Choose `Q`, `K`, `V` values yourself, compute the expected `scores`, `weights`, and output **by hand** (or in a spreadsheet/calculator), and confirm your function's output matches to a few decimal places.

## Step 2 — The scaling factor matters

Repeat the computation from Step 1 but with `d_k=256` and `Q`, `K` entries drawn from `N(0,1)` (so dot products are naturally larger). Compute attention weights both with and without dividing by `√d_k`. Print (or plot) both resulting weight distributions for one query — you should see the unscaled version is far more "peaked" (close to one-hot), exactly as described in the lecture notes.

## Step 3 — Multi-head attention

```python
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        B, T, D = x.shape
        return x.view(B, T, self.num_heads, self.d_k).transpose(1, 2)  # (B, heads, T, d_k)

    def forward(self, x):
        Q = self.split_heads(self.W_q(x))
        K = self.split_heads(self.W_k(x))
        V = self.split_heads(self.W_v(x))
        out, weights = scaled_dot_product_attention(Q, K, V)
        out = out.transpose(1, 2).contiguous().view(x.size(0), x.size(1), -1)
        return self.W_o(out), weights
```

Confirm the shapes are correct for a batch of sequences: `MultiHeadAttention(64, 8)(torch.randn(2, 10, 64))[0].shape` should be `(2, 10, 64)`.

## Step 4 — Positional encoding

```python
def positional_encoding(seq_len, d_model):
    pos = torch.arange(seq_len).unsqueeze(1).float()
    i = torch.arange(d_model).unsqueeze(0).float()
    angle_rates = 1 / (10000 ** (2 * (i // 2) / d_model))
    angles = pos * angle_rates
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(angles[:, 0::2])
    pe[:, 1::2] = torch.cos(angles[:, 1::2])
    return pe
```

Visualize the resulting `positional_encoding(50, 64)` matrix as a heatmap. Confirm each row (position) is a unique pattern, and that nearby positions have more similar patterns than distant ones (check this quantitatively with cosine similarity between a few pairs of rows).

## Step 5 — Assemble a minimal Transformer encoder block

```python
class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, weights = self.attn(x)
        x = self.norm1(x + attn_out)        # Add & Norm
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)         # Add & Norm
        return x, weights
```

Stack 2 of these blocks, add an embedding layer plus your Step 4 positional encoding at the input, and a final classification head (e.g., mean-pool over the sequence dimension, then a linear layer) to build a small sequence classifier.

## Step 6 — Train and compare against your RNN-family results

Train your Transformer-encoder classifier on the sentiment (or similar) task you used in [Lab 6](lab06-rnns-and-grus.md)/[Lab 7](lab07-lstm.md), and compare accuracy and wall-clock training time directly against your best RNN/GRU/LSTM result on the same data.

## Checkpoint questions

1. In Step 2, roughly how much more "peaked" (e.g., measured by the maximum weight in the distribution) was the unscaled attention compared to the scaled version? Does the size of the effect roughly match what you'd predict from the `√d_k` argument in the lecture?
2. In Step 5, why is `LayerNorm` applied *after* adding the residual (`norm1(x + attn_out)`) rather than before? (This is called the "post-norm" convention used in the original Transformer paper — some later variants use "pre-norm" instead. Try swapping the order and see whether training stability changes.)
3. In Step 6, did the Transformer train faster (in wall-clock time) than your RNN-family models, given the same hardware? Does this match the "self-attention is highly parallelizable" claim from the lecture, and if not, what might explain the discrepancy on a small model/dataset where RNN sequential overhead is less of a bottleneck?
