# Assignment 1 — Neural Networks from Scratch

**Covers:** Week 1 (Introduction to Neural Networks), Week 2 (Training Neural Networks: A Deep Dive)
**Deliverable:** Python code (NumPy only for Part A; any framework for Part B) + written report

## Learning objectives

By completing this assignment you should be able to: implement the forward pass and backpropagation of a multi-layer perceptron using only NumPy; explain, in your own words and with your own numbers, why non-linear activation functions are necessary; diagnose vanishing-gradient behavior empirically; and use standard training tricks (better initialization, batch normalization, regularization, modern optimizers) to fix a poorly-training network.

## Part A — Build an MLP from scratch (NumPy only)

Implement, without using any deep learning framework (no PyTorch/TensorFlow — NumPy only):

1. A fully connected layer with configurable input/output size, supporting a forward pass and a backward pass that returns gradients with respect to its weights, bias, and input.
2. At least three activation functions (sigmoid, tanh, ReLU) with their forward and backward (derivative) implementations.
3. A full forward pass through a network with at least one hidden layer, and a full backward pass using the chain rule (Week 1) to compute gradients for every weight and bias in the network.
4. A training loop using plain (vanilla) gradient descent that trains your network on a small binary classification dataset of your choice (a synthetic 2D dataset such as `make_moons` or `make_circles` from scikit-learn is a good choice because you can visualize the decision boundary).

**Required experiment:** Train your from-scratch network on the XOR problem (4 points, as in Week 1's slides). Show that a single-layer perceptron (no hidden layer) cannot solve it, but your two-layer network can. Include a plot of the learned decision boundary.

## Part B — Diagnose and fix training problems (Week 2 techniques)

Using PyTorch (or the framework of your choice) this time, build a deeper feedforward network (at least 6–8 hidden layers) trained on a real dataset (e.g., Fashion-MNIST or a tabular dataset of your choice).

1. **Reproduce vanishing gradients:** train the deep network with all-sigmoid activations and naive (e.g., all-zero or unit-Gaussian) weight initialization. Plot the average gradient magnitude per layer over the first several training steps and show that early layers receive much smaller gradients than later layers.
2. **Fix it:** apply the techniques from Week 2 one at a time (ReLU activations; then He initialization; then batch normalization) and show, with plots, how each change affects the gradient-magnitude-per-layer plot and the training/validation loss curves.
3. **Optimizers:** compare training curves (loss vs. iteration) for plain SGD, SGD with momentum, and Adam on the same network and dataset. Discuss which converges fastest and why, referencing the mechanics covered in the Week 2 notes.
4. **Regularization:** add dropout and/or L2 weight decay to your best-performing configuration from steps 2–3, and show its effect on the gap between training and validation accuracy (i.e., overfitting).

## Report requirements

Your report should include: your XOR decision-boundary plot and a short explanation of why a single-layer perceptron fails; your gradient-magnitude-per-layer plots for the "broken" and "fixed" configurations, with a written explanation connecting what you observe to the vanishing-gradient theory from Week 2; your optimizer comparison plot and a short discussion; and your final training/validation accuracy curves with and without regularization.

## Grading rubric

| Component | Weight |
|---|---|
| Part A: correct from-scratch forward/backward implementation | 30% |
| Part A: XOR experiment and decision boundary plot | 10% |
| Part B: vanishing-gradient reproduction and diagnosis | 20% |
| Part B: fixes (activation/init/BatchNorm) with supporting plots | 15% |
| Part B: optimizer and regularization comparisons | 15% |
| Report clarity and correctness of explanations | 10% |
