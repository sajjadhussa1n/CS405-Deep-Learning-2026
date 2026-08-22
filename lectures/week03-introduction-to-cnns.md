# Week 3 — Introduction to Convolutional Neural Networks

*Companion notes for [`slides/lecture_week03.pdf`](../slides/lecture_week03.pdf)*

## Why this week matters

Weeks 1 and 2 covered fully connected networks, where every neuron connects to every neuron in the next layer. That design breaks down completely when the input is an image. This week introduces the convolutional neural network (CNN) — an architecture built specifically around the idea that pixels close to each other are related, and that a useful visual pattern (an edge, a texture) can appear anywhere in an image. Everything in Weeks 4 and 5 (ResNet, MobileNet, object detection, segmentation) is built directly on top of the ideas introduced here.

## 1. Why fully connected networks fail on images

Imagine feeding a modest 1000×1000 image (1 million pixels) into a fully connected layer with 1 million neurons. Every one of those 1 million output neurons would need its own weight for every one of the 1 million input pixels — that's **one trillion** parameters in a single layer. This is not just expensive, it is essentially infeasible to train, and it gets worse: flattening the image into a long vector of pixels throws away the spatial structure (which pixels were next to which), and a fully connected network has no built-in notion of **translation invariance** — if a cat moves from the top-left of an image to the bottom-right, the network has to learn to recognize it in that new position essentially from scratch, because a different set of weights is responsible for that region.

CNNs solve all three problems at once with two structural ideas:

- **Local connectivity:** a neuron in a convolutional layer only looks at a small local region of the input (its "receptive field"), not the entire image.
- **Parameter sharing:** the *same* small set of weights (a "filter" or "kernel") is reused at every spatial position in the image, instead of learning a separate set of weights per location.

Local connectivity plus parameter sharing means the number of parameters in a convolutional layer depends only on the filter size, not on the size of the image — a filter that detects vertical edges works the same way whether that edge appears in the top-left or bottom-right of the picture, which is exactly the translation invariance a fully connected network lacked.

## 2. Biological inspiration: the visual cortex

This design is not just a mathematical convenience — it echoes real biology. Hubel and Wiesel's 1959 experiments on the cat visual cortex found a hierarchical organization of neurons: **simple cells** respond to edges at specific orientations in a specific location, while **complex cells** respond to the same kind of edge regardless of its exact position (some early translation invariance, built into biology). Processing proceeds hierarchically, from edges to shapes to whole objects, and each neuron only responds to a specific region of the visual field — its receptive field. CNNs mirror this almost directly: early convolutional layers behave like simple cells (edge detectors), pooling layers behave like complex cells (adding position invariance), and deep layers combine everything into object recognition.

## 3. The convolution operation

Mathematically, 2D discrete convolution slides a small filter `K` (typically 3×3, 5×5, or 7×7) across an input `X` and, at every position, computes the sum of element-wise products between the filter and the patch of the input it currently covers:

```
S[i, j] = Σ_u Σ_v  X[i+u, j+v] · K[u, v]
```

Concretely: place the filter over the top-left corner of the image, multiply each filter value by the pixel underneath it, sum all those products into one number, and write that number into the output. Then slide the filter one step to the right and repeat, continuing left-to-right and top-to-bottom until the entire image has been covered. The result, `S`, is called a **feature map**.

Different filters detect different things. A filter with positive weights on the left column and negative weights on the right column (`[[1,0,-1],[1,0,-1],[1,0,-1]]`) responds strongly to vertical edges; a filter with positive weights on top and negative on the bottom responds to horizontal edges. In practice we never hand-design these filters — a CNN *learns* them from data, and in doing so it discovers edge detectors, blob detectors, color detectors, and texture detectors on its own, because those turn out to be useful for the task. Every filter you use produces its own feature map, and stacking many filters' feature maps together forms the "depth" (channel dimension) of the convolutional layer's output — bright regions in a feature map indicate a strong presence of whatever that filter detects, dark regions indicate its absence.

### Stride and padding

Two hyperparameters control exactly how the filter slides and what happens at the borders:

- **Stride** is the step size the filter moves by. Stride 1 (move one pixel at a time) produces the most detailed output but costs the most computation; stride 2 roughly halves the output's height and width and cuts computation substantially, which is common in practice. Larger strides trade fine detail for speed.
- **Padding** addresses the fact that plain convolution shrinks the spatial size: an `N×N` input with an `F×F` filter produces an `(N-F+1)×(N-F+1)` output, and border pixels get used in fewer filter positions than interior pixels, so information near the edges is under-represented. Adding a border of extra pixels (most commonly zeros — "zero padding") around the input compensates for this. **Valid padding** means no padding at all (`P=0`), and the output legitimately shrinks. **Same padding** chooses `P` so that the output size exactly matches the input size (for stride 1, `P = (F-1)/2`).

Putting stride and padding together, the general formula for one spatial dimension of the output is:

```
O = floor((N - F + 2P) / S) + 1
```

and this formula is applied independently to width and height.

### Convolution over multi-channel (RGB) input

A real photo isn't a single 2D grid — an RGB image has three channels (red, green, blue). A convolutional filter over such an image must span **all** input channels: for a `k×k` spatial filter over an RGB input, the actual filter shape is `k × k × 3` — really three separate 2D filters, one per channel, stacked together. The convolution step then multiplies each channel of the input patch by its corresponding channel of the filter, sums the results across *all* channels and spatial positions into a single number, adds a bias, and applies an activation function — producing one scalar value in the output feature map. So one filter, no matter how many input channels it spans, always produces exactly one output feature map; the *number of filters* you choose determines the depth of the output volume.

It is worth explicitly noticing that this whole operation is really just a special, constrained case of the linear-plus-activation operation from Week 1: flatten the local patch of the input into a vector `X` and the filter weights into a vector `W`, and the convolution is simply `z = WX + b`, `a = g(z)`. **A CNN is nothing more than a neural network where the connections are local and the weights are shared across space.**

### Layer dimensions, formally

For layer `ℓ`, given an input of shape `n_H^[ℓ-1] × n_W^[ℓ-1] × n_C^[ℓ-1]`, and hyperparameters filter size `f^[ℓ]`, stride `s^[ℓ]`, padding `p^[ℓ]`, and number of filters `n_C^[ℓ]`, the filter weight tensor has shape `f^[ℓ] × f^[ℓ] × n_C^[ℓ-1] × n_C^[ℓ]`, the bias has shape `1×1×1×n_C^[ℓ]`, and the output has shape:

```
n_H^[ℓ] = floor((n_H^[ℓ-1] - f^[ℓ] + 2p^[ℓ]) / s^[ℓ]) + 1
n_W^[ℓ] = floor((n_W^[ℓ-1] - f^[ℓ] + 2p^[ℓ]) / s^[ℓ]) + 1
```

with the layer's activations computed exactly like a fully connected layer, `Z^[ℓ] = W^[ℓ] * A^[ℓ-1] + b^[ℓ]`, `A^[ℓ] = g(Z^[ℓ])` — just with `*` denoting the convolution operation instead of a plain matrix multiply.

## 4. Pooling: building in invariance

Convolution is *equivariant* to translation — if the input shifts, the feature map shifts with it, but it doesn't become invariant on its own. For classification, we usually want the final decision to be robust to small shifts in exactly where a feature appears. **Pooling** layers add that invariance while also reducing the spatial size (and therefore the parameter count and computation) of later layers, which also helps prevent overfitting and adds some robustness to small distortions and noise.

- **Max pooling** slides a small window (commonly 2×2, with stride equal to the pool size so windows don't overlap) across the feature map and keeps only the maximum value in each window. For example, over a 4×4 input split into four 2×2 blocks with values `{2,5,4,9}`, `{1,3,2,7}`, `{3,6,1,4}`, `{8,1,5,2}`, max pooling produces `9, 7, 6, 8` respectively — it keeps the strongest activation of each region, which tends to correspond to "was this feature present anywhere in this region," a useful and very common choice.
- **Average pooling** instead takes the mean of each window (the same four blocks above become `5.0, 3.25, 3.5, 4.0`), smoothing rather than picking out the peak.
- **Global average pooling** averages an entire feature map down to a single number — a technique we'll see used in later, more modern architectures (Week 4) to replace large fully connected layers.
- **Strided convolution** is sometimes used as a *learnable* alternative to fixed pooling, achieving downsampling as part of the convolution itself rather than as a separate step.

## 5. Putting it together: a simple CNN architecture

A typical CNN follows a repeating pattern: an input image goes through alternating blocks of **convolution + activation** followed by **pooling**, repeated multiple times with increasing depth (number of filters) as spatial size shrinks. Early conv–pool blocks learn simple features (edges, textures); deeper blocks combine those into shapes, object parts, and eventually whole objects — spatial resolution decreases while the "depth" (number of channels/feature types) increases at each stage. After the convolutional blocks, the resulting 3D volume of feature maps is **flattened** into a 1D vector, which is fed into one or more **fully connected layers** that perform higher-level reasoning and combine the learned features into a final decision, ending in an output layer with a sigmoid (binary classification) or softmax (multi-class classification) activation, exactly as described in Week 1.

## 6. LeNet-5: the first successful CNN

LeNet-5, developed by Yann LeCun and colleagues in 1998, was the first CNN to demonstrate real practical success — recognizing handwritten digits (the MNIST dataset) well enough to be deployed for reading handwritten checks in banking. It established the now-canonical **convolution → pooling → convolution → pooling → fully connected** pattern that essentially every later CNN architecture builds on. Technically, LeNet-5 took 32×32 grayscale images as input, used two convolutional layers and two pooling layers followed by two fully connected layers, had only about 60,000 parameters (tiny by modern standards), used sigmoid/tanh activations (this was long before ReLU became standard), and was trained with stochastic gradient descent. It achieved a 0.8% error rate on MNIST — state of the art at the time — proving that CNNs and backpropagation could solve real-world problems. Its broader impact was limited for over a decade afterward simply because the hardware and datasets of the late 1990s and 2000s weren't yet capable of training much larger CNNs — exactly the "second AI winter" story from Week 1. Next week picks up the story from AlexNet (2012) onward, when the hardware, data, and algorithmic pieces finally came together.

## Key takeaways

CNNs replace the "every neuron connects to every input" design of fully connected networks with **local connectivity** and **parameter (weight) sharing**, which drastically cuts the parameter count and gives the network a built-in notion of translation invariance — directly inspired by the hierarchical, locally-receptive organization of the biological visual cortex. A convolutional layer is defined by its filter size, stride, padding, and number of filters, and these four hyperparameters together determine the output's spatial dimensions via `O = floor((N - F + 2P)/S) + 1`. Pooling layers (typically max pooling) add further translation invariance and reduce computation. Stacking conv–pool blocks builds a feature hierarchy from edges to shapes to objects, and flattening the final feature maps into a fully connected classifier head produces the final prediction — the pattern first proven out by LeNet-5 in 1998 and reused, in far larger form, by every architecture we study next week.
