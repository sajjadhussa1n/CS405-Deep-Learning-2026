# Quiz 2 — Convolutional Neural Networks

*Covers: [Week 3](../lectures/week03-introduction-to-cnns.md), [Week 4](../lectures/week04-deep-cnn-architectures.md), and [Week 5](../lectures/week05-advanced-computer-vision.md).*

## Section A — Short answer

**A1.** Why is a fully connected layer generally a poor choice for processing raw high-resolution images directly, and what two properties of convolutional layers address this?

**A2.** Explain the difference between max pooling and average pooling, and give one reason a network might use max pooling.

**A3.** What problem does a residual (skip) connection solve, and how does the identity shortcut help with it mathematically?

**A4.** What is the purpose of a 1×1 convolution in architectures like Inception and ResNet's bottleneck blocks?

**A5.** Describe the difference between a classification task, an object detection task, and a semantic segmentation task, in terms of what the model outputs for a given image.

## Section B — Multiple choice

**B1.** Given a 32×32 input, a 5×5 convolution kernel, stride 1, and no padding, what is the spatial size of the output feature map?
(a) 32×32 (b) 28×28 (c) 27×27 (d) 30×30

**B2.** Which architecture first introduced residual/skip connections to enable training of very deep networks (100+ layers)?
(a) AlexNet (b) VGGNet (c) ResNet (d) LeNet-5

**B3.** Depthwise separable convolutions, the key efficiency trick in MobileNet, work by:
(a) Replacing convolution with fully connected layers
(b) Splitting a standard convolution into a per-channel spatial convolution followed by a 1×1 pointwise convolution
(c) Removing all pooling layers
(d) Doubling the number of filters at every layer

**B4.** In the YOLO object detection framework, Non-Maximum Suppression (NMS) is used to:
(a) Normalize bounding box coordinates
(b) Remove duplicate/overlapping detections of the same object, keeping the highest-confidence box
(c) Increase the number of anchor boxes
(d) Convert the classification head into a regression head

**B5.** U-Net's characteristic "skip connections" between the encoder and decoder paths primarily serve to:
(a) Speed up training by reducing parameter count
(b) Recover fine-grained spatial detail lost during downsampling, for pixel-accurate segmentation
(c) Prevent overfitting via regularization
(d) Replace the need for a bottleneck layer

## Section C — Calculation

**C1.** You have a 227×227×3 input image. It passes through a convolutional layer with 96 filters of size 11×11, stride 4, no padding. What is the shape (height × width × channels) of the output volume? (This mirrors AlexNet's first layer.)

**C2.** Compute the Intersection over Union (IoU) for two bounding boxes: box A = `[x1=0, y1=0, x2=10, y2=10]` (area 100) and box B = `[x1=5, y1=5, x2=15, y2=15]` (area 100). Their intersection is the region `[5,5]` to `[10,10]`.

**C3.** A convolutional layer has input with 64 channels and applies 128 filters of size 3×3 (with bias). How many learnable parameters does this layer have?

## Answer Key

**A1.** A fully connected layer applied directly to a raw image would need one weight per input pixel per neuron, so the parameter count explodes with image resolution and the layer has no notion that nearby pixels are related — it would have to relearn "edge detection" separately at every spatial location. Convolutional layers address this with (1) parameter sharing — the same small filter slides across the whole image, so the same weights detect the same pattern anywhere it appears — and (2) local connectivity — each output unit only looks at a small local receptive field, matching the local, spatially-correlated structure of images.

**A2.** Max pooling takes the maximum activation within each pooling window; average pooling takes the mean. Max pooling is used more often in practice because it retains the strongest activation for a detected feature (a sharp signal that a pattern was present somewhere in that region) and tends to give slightly better empirical performance for classification-style tasks, while being naturally robust to small spatial shifts.

**A3.** Residual connections solve the degradation/vanishing-gradient problem that makes very deep plain networks *harder* to optimize than shallower ones (not just prone to overfitting, but literally harder to fit even the training set). Mathematically, instead of a block learning a direct mapping `H(x)`, it learns a residual `F(x) = H(x) - x`, and the block's output is `F(x) + x`. The `+x` identity shortcut provides a direct additive path for the gradient to flow backward through, unimpeded by the block's (possibly poorly conditioned) weight layers, which keeps gradients from vanishing even in very deep stacks.

**A4.** A 1×1 convolution operates only across the channel dimension at each spatial location (it doesn't mix spatial neighbors), so it can cheaply change the number of channels — most often to *reduce* channels before an expensive larger convolution (a "bottleneck," saving computation) or to *combine* information across channels without altering spatial resolution, as in Inception's parallel-branch design.

**A5.** Classification: the model outputs a single label (or a probability distribution over labels) for the whole image, with no spatial information. Object detection: the model outputs a set of bounding boxes, each with a class label and confidence score, localizing every instance of every object in the image. Semantic segmentation: the model outputs a label for *every pixel* in the image, producing a full-resolution map of what class each pixel belongs to (without distinguishing separate instances of the same class, unlike instance segmentation).

**B1.** (b) 28×28. (`(32-5)/1 + 1 = 28`)

**B2.** (c) ResNet.

**B3.** (b) Splitting a standard convolution into a per-channel (depthwise) spatial convolution followed by a 1×1 (pointwise) convolution.

**B4.** (b) Remove duplicate/overlapping detections of the same object, keeping the highest-confidence box.

**B5.** (b) Recover fine-grained spatial detail lost during downsampling, for pixel-accurate segmentation.

**C1.** Output spatial size: `(227-11)/4 + 1 = 55`. So the output volume is 55×55×96.

**C2.** Intersection area = `(10-5) × (10-5) = 25`. Union area = `100 + 100 - 25 = 175`. IoU = `25/175 ≈ 0.143`.

**C3.** Each filter has `3×3×64 = 576` weights plus 1 bias = 577 parameters. With 128 filters: `128 × 577 = 73,856` parameters.
