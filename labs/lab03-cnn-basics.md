# Lab 3 — CNN Basics

**Matches:** [Week 3 — Introduction to Convolutional Neural Networks](../lectures/week03-introduction-to-cnns.md)
**Goal:** Implement 2D convolution and pooling by hand to build genuine intuition, then build and train a small CNN in PyTorch.

## Setup

```bash
pip install numpy torch torchvision matplotlib
```

## Step 1 — Convolution by hand (NumPy)

Implement plain 2D convolution (valid padding, configurable stride) exactly following the step-by-step procedure in the lecture notes:

```python
import numpy as np

def conv2d(image, kernel, stride=1):
    H, W = image.shape
    kH, kW = kernel.shape
    out_h = (H - kH) // stride + 1
    out_w = (W - kW) // stride + 1
    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            region = image[i*stride:i*stride+kH, j*stride:j*stride+kW]
            out[i, j] = np.sum(region * kernel)
    return out
```

Test it on a small 5×5 synthetic image with a hand-designed vertical-edge kernel (`[[1,0,-1],[1,0,-1],[1,0,-1]]`) and confirm the output highlights the vertical edge you placed in the image.

## Step 2 — Stride, padding, and the output-size formula

Add zero-padding support to your `conv2d`, and verify the output-size formula `O = floor((N - F + 2P)/S) + 1` from the lecture by testing several `(N, F, P, S)` combinations and checking your output array's shape matches the formula's prediction.

## Step 3 — Max pooling by hand

```python
def max_pool2d(feature_map, pool_size=2, stride=2):
    H, W = feature_map.shape
    out_h = (H - pool_size) // stride + 1
    out_w = (W - pool_size) // stride + 1
    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            region = feature_map[i*stride:i*stride+pool_size, j*stride:j*stride+pool_size]
            out[i, j] = region.max()
    return out
```

Reproduce the worked 4×4 → 2×2 max-pooling example from the lecture notes and confirm your function gives the same numbers (9, 7, 6, 8).

## Step 4 — Load a real image and apply your filters

Load a grayscale image (e.g., with `PIL` or `torchvision.io.read_image`, converted to grayscale and downsized to something like 128×128 for speed) and apply your hand-written `conv2d` with a few different hand-designed kernels (vertical edge, horizontal edge, a simple blur/box kernel). Display the original image and each filtered output side by side. Note how much slower your pure-Python loop is than a vectorized/library implementation — this motivates why real CNN implementations use optimized (often GPU) convolution routines rather than the naive loop.

## Step 5 — A small CNN in PyTorch

Build a small CNN (2 conv+pool blocks, 1 FC head) with `nn.Conv2d`/`nn.MaxPool2d` and train it on MNIST or Fashion-MNIST.

```python
import torch.nn as nn

class SmallCNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Linear(32 * 7 * 7, n_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
```

Train for a few epochs and report test accuracy.

## Step 6 — Visualize learned filters

After training, extract the first convolutional layer's weights (`model.features[0].weight`) and display each of the 16 learned 3×3 filters as a small image. Compare them qualitatively to the hand-designed edge filters from Step 4 — do any of the learned filters resemble edge detectors?

## Checkpoint questions

1. How does your hand-written `conv2d`'s runtime scale as you increase the image size or kernel size? Why do real frameworks avoid this naive triple-nested-loop approach?
2. Why does max pooling (rather than, say, always taking the top-left value of each window) give the network some robustness to small translations?
3. Looking at your visualized learned filters in Step 6, do they look like meaningful patterns, or mostly noise? What could you change (more training epochs, more data, different learning rate) to encourage cleaner, more interpretable filters?
