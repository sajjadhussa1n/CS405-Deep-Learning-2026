# Quiz 1 — Neural Network Foundations

*Covers: [Week 1](../lectures/week01-introduction-to-neural-networks.md) and [Week 2](../lectures/week02-training-neural-networks.md).*

Attempt every question before checking the answer key at the bottom. Where a question asks you to compute something, show your work — the numeric answer alone won't help you find your mistake if it's wrong.

## Section A — Short answer

**A1.** What is the key limitation of a single perceptron that motivated the move to multi-layer networks? Name the classic example problem used to illustrate it.

**A2.** Write out the four-step training loop (in words) that every neural network training run follows, from a fresh batch of data to an updated set of weights.

**A3.** A colleague initializes every weight in a network to exactly zero. Explain, in your own words, why this breaks training regardless of how many hidden units the network has.

**A4.** Define the vanishing gradient problem in one or two sentences, and name one architectural cause and one initialization-related cause.

**A5.** What does Batch Normalization normalize, and at what point in the layer's computation is it typically inserted?

## Section B — Multiple choice

**B1.** Which activation function is most associated with causing vanishing gradients in deep networks due to saturation on both ends?
(a) ReLU (b) Sigmoid (c) Leaky ReLU (d) Softmax

**B2.** He initialization is specifically designed to pair well with which activation function?
(a) Sigmoid (b) Tanh (c) ReLU (d) Linear

**B3.** Which regularization technique works by randomly zeroing out a fraction of neurons' activations during training?
(a) L2 regularization (b) Batch Normalization (c) Dropout (d) Early stopping

**B4.** Which optimizer maintains both a running average of past gradients (momentum-like term) *and* a running average of past squared gradients (adaptive learning rate)?
(a) SGD (b) SGD with Momentum (c) RMSProp (d) Adam

**B5.** L2 regularization adds which term to the loss function?
(a) The sum of absolute values of the weights
(b) The sum of squared weights
(c) The number of nonzero weights
(d) The maximum weight magnitude

## Section C — Calculation

**C1.** Consider the tiny network from [Handout 2](../handouts/handout02-backpropagation-derivation.md): `x=1`, `y=3`, `w1=1.0`, `b1=0`, `w2=0.5`, `b2=0`, sigmoid hidden activation, linear output, MSE loss. Compute the forward pass (`z1, a1, z2, ŷ`) and the loss `L`.

**C2.** Using your answer from C1, compute `δ2 = ŷ - y` and the gradient `∂L/∂w2`.

**C3.** A network's loss plateaus early during training and the gradients in the first hidden layer are on the order of `1e-8` while gradients in the last layer are order `1`. Name two concrete remedies discussed in Week 2 that directly target this symptom.

## Answer Key

**A1.** A single perceptron can only represent linearly separable functions — it cannot learn XOR, since no single straight line separates XOR's positive and negative examples in input space. Stacking layers (with nonlinear activations) is what allows a network to represent non-linear decision boundaries.

**A2.** (1) Forward pass — push the input through the network to produce a prediction. (2) Compute the loss — compare the prediction to the true label using a loss function. (3) Backward pass — use backpropagation (the chain rule) to compute the gradient of the loss with respect to every weight. (4) Update — adjust every weight a small step in the direction that decreases the loss (gradient descent).

**A3.** With all weights at zero, every neuron in a given layer computes the exact same output (zero, then the same activation), and by symmetry receives the exact same gradient during backpropagation. Every neuron in the layer updates identically forever, so the layer behaves as if it had only one effective neuron no matter how many units it has — this is the "symmetry breaking" problem, and it's why weights are initialized randomly.

**A4.** The vanishing gradient problem is when gradients shrink multiplicatively as they're backpropagated through many layers, becoming so small that early layers barely update. An architectural cause is stacking many layers with saturating activations (sigmoid/tanh), where each layer's local derivative is less than 1, so the product across layers shrinks toward zero. An initialization-related cause is weights initialized too small (or too large), which compounds the same multiplicative shrinkage (or explosion) layer over layer.

**A5.** Batch Normalization normalizes the pre-activation (or activation) values of a layer *across the current mini-batch* to zero mean and unit variance, then applies a learned scale (`γ`) and shift (`β`). It's typically inserted after the linear/convolutional transformation and before the nonlinear activation function.

**B1.** (b) Sigmoid.

**B2.** (c) ReLU.

**B3.** (c) Dropout.

**B4.** (d) Adam.

**B5.** (b) The sum of squared weights.

**C1.** `z1 = 1.0×1 + 0 = 1.0`; `a1 = σ(1.0) ≈ 0.7311`; `z2 = 0.5×0.7311 + 0 ≈ 0.3655`; `ŷ ≈ 0.3655`. `L = 0.5×(0.3655-3)² = 0.5×(−2.6345)² ≈ 3.4703`.

**C2.** `δ2 = ŷ - y = 0.3655 - 3 = -2.6345`. `∂L/∂w2 = δ2 × a1 = -2.6345 × 0.7311 ≈ -1.9264`.

**C3.** Any two of: switch to a non-saturating activation such as ReLU/Leaky ReLU; use He or Glorot/Xavier initialization matched to the activation function; add Batch Normalization, which keeps activations in a well-behaved range at every layer and directly stabilizes the gradient flow; use residual/skip connections (previewed in Week 2, developed fully in Week 4) so gradients have a direct path back to earlier layers.
