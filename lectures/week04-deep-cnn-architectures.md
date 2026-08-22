# Week 4 — Deep Convolutional Neural Network Architectures

*Companion notes for [`slides/lecture_week04.pdf`](../slides/lecture_week04.pdf)*

## Why this week matters

Week 3 gave us the basic convolution + pooling recipe. This week is a guided tour through the real history of how that recipe was scaled up and made efficient — from the 2012 breakthrough that kicked off the deep learning era, through the architectural tricks (small kernels, 1×1 convolutions, residual connections, multi-scale processing, depthwise-separable convolutions) that let networks become both much deeper and, eventually, small enough to run on a phone.

## 1. AlexNet (2012) — the breakthrough

Before 2012, CNNs were widely considered too slow and too shallow to be practical for large-scale image recognition. AlexNet changed that overnight by winning the 2012 ImageNet competition with a top-5 error of 15.3%, versus 26.2% for the second-place entry — a massive margin. Its key innovations, several of which we've already met, were: the **ReLU** non-linearity (Week 1–2) instead of sigmoid/tanh, a **GPU implementation** that made training at this scale computationally feasible, **dropout** regularization (Week 2), and **data augmentation** to artificially expand the training set. Architecturally, AlexNet takes a 224×224×3 image through five convolutional layers (with the first using a large 11×11 filter with stride 4, followed by smaller filters in later layers) interspersed with max-pooling, and finishes with three fully connected layers (4096, 4096, 1000 units) and a softmax over 1000 ImageNet classes. It had about 60 million parameters and was originally split across two GPUs because no single GPU had enough memory at the time.

## 2. VGGNet (2014) — simplicity and depth

VGGNet's key idea was refreshingly simple: **replace large filters (11×11, 5×5) with a stack of small 3×3 convolutions.** Two stacked 3×3 convolutional layers (stride 1) have an effective receptive field of 5×5 — they "see" the same area of the input as a single 5×5 filter — and three stacked 3×3 layers reach a 7×7 receptive field, matching still-larger filters. But stacking small filters is cheaper: two 3×3 layers over `C` channels cost `2 × 9C² = 18C²` parameters, versus `25C²` for a single 5×5 layer — about a 28% reduction — and, as a bonus, you get an extra non-linearity (ReLU) inserted between the two smaller convolutions, giving the network more representational power for a similar receptive field.

VGG-16 (the most famous variant) applies this idea rigorously: 13 convolutional layers, all 3×3 with stride 1 and padding 1, followed by 2×2 max-pooling with stride 2, doubling the number of channels after each pooling stage (64 → 128 → 256 → 512 → 512), and finishing with 3 fully connected layers. The uniformity of the design (literally the same 3×3 conv / 2×2 pool building block repeated) made VGG very easy to understand and extend, but at a cost: about 138 million parameters, the vast majority of which sit in the large fully connected layers, making VGG memory-hungry and relatively slow.

## 3. 1×1 convolution — the "channel mixer"

A 1×1 convolution is a filter that spans exactly one spatial pixel but *all* input channels. It doesn't look at any neighboring pixels at all — for every position, it just takes the vector of channel values at that one pixel (e.g., R, G, B) and linearly recombines them into a new vector of channel values, like adjusting the color-mixing dials on an old TV without looking at any other pixel. Mathematically, applying a 1×1×C_in×C_out filter to an H×W×C_in input is exactly equivalent to running a small fully connected layer independently at every pixel, with the same weights reused (shared) across all H×W positions.

This gives 1×1 convolutions two superpowers that show up constantly in later architectures. First, **dimensionality control**: choosing `C_out < C_in` creates a "bottleneck" that compresses the number of channels, while `C_out > C_in` expands them. Second, **cheapness**: since a standard KxK convolution costs `H × W × C_in × C_out × K²` operations, setting `K=1` removes the `K²` factor entirely, leaving just `H × W × C_in × C_out` — for a 3×3 filter, a 1×1 convolution costs 9× less per output pixel purely from removing the spatial extent. This efficiency is exactly what makes 1×1 convolutions the workhorse behind Inception's multi-branch design and ResNet's bottleneck blocks, both covered next.

## 4. ResNet (2015) — learning residuals

The intuitive expectation is "deeper networks should be at least as good as shallower ones" — a deeper network can, in principle, learn to make its extra layers do nothing (an identity mapping) and match a shallower network exactly. In practice, plain (non-residual) very deep networks showed *higher* training error as depth increased — this is the **degradation problem**. Crucially, this is not the vanishing-gradient problem from Week 2 (batch norm and ReLU had already mostly solved that) — the actual hypothesis is that it is simply *hard* for a stack of non-linear layers to learn to approximate a plain identity mapping, even though in principle it should be able to.

ResNet's fix is the **residual block**. Instead of asking a stack of layers `F` to learn the desired output `y` directly, we ask it to learn the *residual* — the difference between the desired output and the input — and then add the original input back in via a **skip (identity) connection**:

```
y = F(x, {W_i}) + x
```

Here `F` is typically two or three stacked convolutional layers, and `x` is passed around them unchanged and added to their output. If the optimal thing for a block to do really is "nothing" (an identity mapping), the block can now achieve that trivially by driving `F(x)` toward zero — which is much easier to learn than forcing several non-linear layers to reconstruct the identity exactly. The skip connection has a second, equally important benefit for training: it creates a direct path for gradients to flow backward without being repeatedly multiplied by layer weights and activation derivatives. Differentiating `y = F(x) + x` gives `∂L/∂x = (∂L/∂y)(1 + ∂F/∂x)` — that leading `1` means at least some of the gradient always flows straight through the skip connection, undiminished, no matter how deep the network is. This is why ResNets can be trained successfully at depths (50, 101, 152 layers, and beyond) that would be essentially untrainable as plain stacks.

### The bottleneck block

For very deep ResNets (ResNet-50/101/152), a plain residual block of two 3×3 convolutions on a wide (e.g., 256-channel) feature map is expensive. The **bottleneck block** uses the 1×1-convolution trick from above to make this cheap: first compress 256 channels down to 64 with a 1×1 convolution, then run the expensive 3×3 convolution on only those 64 channels, then expand back to 256 channels with another 1×1 convolution — like summarizing a long document before doing detailed analysis, then expanding the findings back out. Counting operations per pixel makes the saving concrete: a plain 3×3 conv on 256→256 channels costs `256 × 256 × 9 = 589,824` operations, while the bottleneck version costs `(256×64×1) + (64×64×9) + (64×256×1) = 16,384 + 36,864 + 16,384 = 69,632` operations — about **8.5× fewer**, simply because the expensive 3×3 convolution now only has to process 64 channels instead of 256.

## 5. Inception (GoogLeNet) — multi-scale processing

Objects in real images appear at very different scales — a face might fill the frame or occupy a tiny corner. Rather than committing to a single filter size and hoping it works for every scale, the **Inception module** runs several filter sizes *in parallel* on the same input (1×1, 3×3, 5×5 convolutions, plus a 3×3 max-pool branch) and concatenates all of their outputs together, letting the network effectively learn which scales matter through training. The naive version of this is very expensive — a 5×5 convolution on a high-channel feature map costs a lot — so the practical Inception module inserts a 1×1 bottleneck convolution *before* each expensive branch to first reduce the number of channels, exactly the same compress-then-process trick used in ResNet's bottleneck. GoogLeNet is the full network built by stacking these Inception modules.

## 6. Depthwise separable convolution

Everything so far has still used **standard convolution**, which does two things simultaneously in one expensive operation: it mixes information across *space* (looking at a K×K neighborhood) and across *channels* (combining all C_in input channels into each output value). **Depthwise separable convolution** splits these two jobs into two much cheaper sequential steps:

1. **Depthwise convolution (spatial only):** apply one separate small filter (e.g., 3×3) to *each* input channel independently — no mixing between channels at all. If there are `C_in` channels, you use `C_in` separate 2D filters, and the output still has `C_in` channels.
2. **Pointwise convolution (channel mixing only):** apply a 1×1 convolution to that result to mix the channels together and produce the desired number of output channels — exactly the "channel mixer" from Section 3.

The efficiency gain is large and easy to verify with a worked example. For a 7×7 input with 3 channels, producing 4 output channels with 3×3 filters: a standard convolution costs `H×W×C_in×C_out×K² = 7×7×3×4×9 = 5,292` operations. The depthwise-separable version costs `(H×W×C_in×K²) + (H×W×C_in×C_out) = (7×7×3×9) + (7×7×3×4) = 1,323 + 588 = 1,911` operations — a 2.8× reduction. In general, the ratio of costs is:

```
Cost_separable / Cost_standard = 1/C_out + 1/K²
```

The savings grow dramatically as the number of output channels increases: with `K=3` and `C_out=512` (typical of deeper layers), the ratio drops to about `1/512 + 1/9 ≈ 0.113` — an **8.8× reduction**. This is why depthwise separable convolutions become more, not less, valuable as networks get wider.

## 7. MobileNet V1 and V2 — efficient architectures for edge devices

**MobileNet V1** is the first architecture built almost entirely out of stacked depthwise-separable convolution blocks (each block: one depthwise 3×3 conv, then one pointwise 1×1 conv), rather than standard convolutions. It also introduces two simple scaling knobs — a **width multiplier α** that uniformly thins the number of channels in every layer, and a **resolution multiplier ρ** that reduces the input image size — letting a developer trade accuracy for speed and memory to fit a specific device's constraints.

**MobileNet V2** refines this with two further ideas. The **inverted residual** block flips the usual bottleneck pattern: instead of compress → process → expand (as in ResNet), it goes thin → **expand** (with a 1×1 conv, typically by a factor `t=6`) → depthwise 3×3 conv → **project** back down (with another 1×1 conv) — the intuition being that the depthwise convolution works better with more channels to play with, so the network temporarily expands into a "wide" representation for that step, then compresses back down, with a skip connection added when the stride is 1 (matching input/output shapes). The **linear bottleneck** is the second refinement: the final projecting 1×1 convolution deliberately has **no ReLU activation** afterward, because ReLU discards negative values, and doing that on an already-compressed (low-dimensional) representation can destroy useful information; keeping that last step linear preserves more of what the network has learned.

These design choices matter enormously for real deployment on **edge devices** — phones, drones, wearables — where power, memory, and compute are all tightly limited. Depthwise separable convolutions (8–9× cheaper than standard convolutions in FLOPs) plus MobileNet V2's linear bottleneck (preserving information) and inverted residual (improving gradient flow) together make it practical to run capable CNNs directly on such constrained hardware.

## 8. Comparing the architectures

| Model | Top-1 Accuracy | Parameters (M) | FLOPs (B) |
|---|---|---|---|
| AlexNet | 57.1% | 62.3 | 0.7 |
| VGG-16 | 71.5% | 138.3 | 15.5 |
| ResNet-50 | 75.2% | 25.6 | 3.9 |
| Inception-v3 | 78.8% | 23.8 | 5.7 |
| MobileNetV1 | 70.6% | 4.2 | 0.57 |
| MobileNetV2 | 72.0% | 3.4 | 0.30 |

The story the table tells: VGG achieves reasonable accuracy but at huge memory and compute cost; ResNet gets substantially better accuracy while actually *reducing* parameters and FLOPs relative to VGG, thanks to bottleneck blocks and skip connections that make depth practical; and the MobileNet family trades a modest amount of accuracy for a dramatic (10–50×) reduction in parameters and FLOPs, making them the right choice specifically for mobile/edge deployment rather than for chasing the absolute best accuracy.

## Key takeaways

The history of CNN architecture design is really a history of answering "how do we get the benefits of depth without the cost?" VGG showed that stacking small 3×3 filters is more parameter-efficient than using large filters directly. The 1×1 convolution turned out to be a remarkably versatile, cheap tool for mixing channels and controlling dimensionality, and became the key ingredient in both ResNet's bottleneck blocks and Inception's multi-scale modules. ResNet's skip connections solved the degradation problem and gave gradients a direct path through very deep networks, making 50+ layer networks trainable. Inception let the network choose its own effective filter size by processing multiple scales in parallel. And depthwise separable convolutions, used throughout MobileNet, split the expensive "spatial + channel mixing" operation into two cheap sequential steps, unlocking real-time deep learning on phones and other edge devices. Every one of these ideas — small kernels, 1×1 bottlenecks, residual connections, multi-scale branches, depthwise separability — reappears, in some form, in nearly every modern vision architecture you will encounter after this course.
