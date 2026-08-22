# Lab 4 — Modern CNN Architectures and Transfer Learning

**Matches:** [Week 4 — Deep CNN Architectures](../lectures/week04-deep-cnn-architectures.md)
**Goal:** Build a residual block and a depthwise-separable convolution block from scratch, measure their computational cost, and practice transfer learning with a pre-trained backbone.

## Setup

```bash
pip install torch torchvision matplotlib
```

## Step 1 — A residual block from scratch

```python
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity          # the skip connection
        return self.relu(out)
```

## Step 2 — Degradation problem, reproduced

Build two networks of the same depth (e.g., 12 conv layers, 32 channels): one a plain stack of `Conv-BN-ReLU` blocks, one using your `ResidualBlock` throughout. Train both on CIFAR-10 for a fixed, modest number of epochs (enough to see a trend, not necessarily full convergence) and plot training loss for both on one graph. You should see the plain network's loss plateau higher than the residual network's — a small-scale echo of the degradation problem discussed in Week 4.

## Step 3 — A bottleneck block, and counting FLOPs

Implement the bottleneck block (1×1 compress → 3×3 → 1×1 expand) from the lecture notes:

```python
class BottleneckBlock(nn.Module):
    def __init__(self, channels, bottleneck_channels):
        super().__init__()
        self.reduce = nn.Conv2d(channels, bottleneck_channels, 1)
        self.conv = nn.Conv2d(bottleneck_channels, bottleneck_channels, 3, padding=1)
        self.expand = nn.Conv2d(bottleneck_channels, channels, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.relu(self.reduce(x))
        out = self.relu(self.conv(out))
        out = self.expand(out)
        return self.relu(out + identity)
```

Using `torch.numel()` on each layer's weight tensor (or a library like `thop`/`fvcore` if available), count the parameters (or FLOPs) of a plain 256→256 3×3 conv block versus a `BottleneckBlock(256, 64)`. Confirm your measured ratio is close to the ~8.5× reduction reported in the Week 4 notes.

## Step 4 — Depthwise separable convolution

```python
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))
```

Compare its parameter count against a standard `nn.Conv2d(in_channels, out_channels, 3, padding=1)` for a few `(in_channels, out_channels)` pairs, and confirm the savings grow as `out_channels` increases, matching the `1/C_out + 1/K²` formula from the lecture.

## Step 5 — Transfer learning

Load a pre-trained backbone (`torchvision.models.resnet18(weights="IMAGENET1K_V1")` or `mobilenet_v2`) and fine-tune it on a small image classification dataset (e.g., a subset of CIFAR-10, or a small custom dataset).

1. **Frozen backbone:** freeze all layers except the final classifier, train only the head. Report validation accuracy and training time.
2. **Full fine-tuning:** unfreeze everything and fine-tune the whole network with a small learning rate. Report validation accuracy and training time.

Compare the two, and connect your observation back to the Week 2/Week 4 guidance on when each strategy makes sense based on dataset size.

## Checkpoint questions

1. In Step 2, does the residual network's advantage over the plain network grow or shrink as you increase depth further? (Try a deeper variant if time permits.)
2. In Step 3, which part of the bottleneck block's cost dominates — the two 1×1 convolutions or the middle 3×3 convolution? Does this match the "the 3×3 now only processes 64 channels, not 256" explanation from the lecture?
3. In Step 5, which fine-tuning strategy converged faster, and which reached higher final accuracy? Was this what you expected given your dataset's size?
