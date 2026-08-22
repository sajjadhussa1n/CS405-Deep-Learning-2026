# Week 1 — Introduction to Neural Networks

*Companion notes for [`slides/lecture_week01.pdf`](../slides/lecture_week01.pdf)*

## Why this week matters

Before we can talk about convolutional networks, transformers, or large language models, we need to understand the single building block that all of them are made of: the artificial neuron. This week traces the idea of a "neuron" from a rough sketch of biology, through a mathematical model that could compute logic gates, through the first real crisis in the field (the XOR problem and the first "AI winter"), and finally to the multi-layer networks and the learning algorithm — backpropagation — that make today's deep learning possible. If you understand this week deeply, everything that follows in the course is really just "more of the same idea, arranged differently."

## 1. From biological neurons to a mathematical model

A real neuron in your brain has **dendrites** that receive signals from other neurons, a **nucleus** that combines those signals, an **axon** that carries an output signal away from the cell, and **synapses** where it connects to the next neuron. The key idea that early researchers borrowed from biology was not the electrochemistry — it was the *pattern*: a neuron collects several inputs, combines them in a weighted way, and fires (or doesn't) depending on whether that combination crosses some threshold. Everything in this course is an elaboration of that one pattern: *weighted sum, then a decision*.

### The McCulloch-Pitts neuron

The very first mathematical model of a neuron, proposed by McCulloch and Pitts, turned that idea into an equation. Each input `x_i` is multiplied by a weight `w_i`, the products are summed, and the neuron "fires" (outputs 1) if that sum meets or exceeds a threshold `θ`, and stays silent (outputs 0) otherwise:

```
output = 1  if  Σ(w_i · x_i) ≥ θ
output = 0  otherwise
```

This model had no learning — the weights were fixed by hand — and it operated synchronously (all neurons updated in lock-step). Despite being so simple, it turns out you can hand-pick weights and thresholds to make a single McCulloch-Pitts neuron behave exactly like a logic gate:

- **AND gate:** with weights `w1 = w2 = 1` and threshold `θ = 2`, the neuron only fires when *both* inputs are 1, because 1 + 1 = 2 is the only combination that reaches the threshold of 2.
- **OR gate:** lowering the threshold to `θ = 1` means the neuron fires as soon as *either* input is 1, since any single "1" already reaches the threshold.
- **NOT gate:** using a single *inhibitory* (negative) weight `w = -1` and threshold `θ = 0` flips the input: when `x = 1`, the weighted sum is -1, which is below the threshold, so the neuron stays off; when `x = 0`, the sum is 0, which meets the threshold, so the neuron fires.

So a single artificial neuron, with the right weights, can implement basic Boolean logic. This was exciting in the 1940s and 50s because it suggested that networks of these simple units might be able to compute anything a computer could.

### The problem that logic gates could not solve: XOR

The excitement ran into a wall almost immediately. Consider the XOR ("exclusive or") function: it outputs 1 only when its two inputs *differ* (0,1 → 1 and 1,0 → 1) and 0 when they are the *same* (0,0 → 0 and 1,1 → 0). If you try to draw the four input points `(0,0)`, `(0,1)`, `(1,0)`, `(1,1)` on a 2D plane and color them by their XOR output, you'll find there is **no single straight line** that can separate the "1" points from the "0" points. Since a single McCulloch-Pitts neuron (or perceptron) can only draw one straight decision boundary, it is mathematically impossible for one neuron, with any choice of weights and threshold, to compute XOR. This is the central limitation of a *single-layer* network: it can only solve **linearly separable** problems.

The fix, even in the earliest days, was already understood conceptually: use *two layers* instead of one. XOR can be rewritten as `(x1 NAND x2) AND (x1 OR x2)`. The first layer computes two intermediate values (a NAND and an OR), and a second layer combines those two intermediate results with an AND gate to produce the final XOR output. This is the seed of the idea of a **hidden layer**: a layer of neurons that doesn't produce the final answer directly, but instead computes useful intermediate features that a later layer can combine.

## 2. The Perceptron and its learning rule

In 1958, Frank Rosenblatt introduced the **perceptron**, the first neuron model that could *learn* its own weights from data instead of having them hand-set. The perceptron generalizes the McCulloch-Pitts neuron by allowing real-valued inputs and weights and by adding an explicit **bias** term `b` (which plays the same role as the threshold, just moved to the other side of the equation):

```
z = Σ(w_i · x_i) + b
output = step(z)     (step(z) = 1 if z ≥ 0, else 0)
```

The learning algorithm is beautifully simple. For every training example `(x, y_true)`:

1. Compute the perceptron's current prediction, `y_pred = step(w · x + b)`.
2. Compute how wrong it was: `error = y_true - y_pred`.
3. Nudge every weight in the direction that would have reduced that error: `w_i ← w_i + η · error · x_i`.
4. Nudge the bias the same way: `b ← b + η · error`.

Here `η` (eta) is the **learning rate**, a small positive number (commonly 0.1 or 0.01) that controls how big a step we take on each update. If a prediction was already correct, `error = 0` and nothing changes. If the perceptron predicted 0 but the true label was 1, the weights are pushed up in the direction of the input, making the neuron more likely to fire on similar inputs in the future.

A single perceptron can learn to compute AND, OR, and NOT — anything that is linearly separable — directly from examples, without a human hand-computing the weights. Geometrically, the equation `w1·x1 + w2·x2 + b = 0` defines a line in 2D (or a plane in 3D, or a hyperplane in higher dimensions), and training simply nudges that line/plane/hyperplane until it separates the two classes. But this also means the perceptron inherits exactly the same fundamental weakness as the McCulloch-Pitts neuron: it **cannot** learn XOR, because no straight line can separate XOR's positive and negative examples, no matter how long you train it.

## 3. The first AI winter

In 1969, Marvin Minsky and Seymour Papert published a book called *Perceptrons*, which mathematically proved that single-layer perceptrons cannot solve non-linearly-separable problems like XOR. Although the fix (adding hidden layers) was already conceptually known, the book was read pessimistically, and it triggered a sharp drop in funding and interest in neural networks. Research largely shifted toward symbolic/rule-based AI for well over a decade. This period of stagnation is remembered as the **first AI winter** — a recurring theme in AI history where a genuine technical limitation gets over-generalized into "this whole approach doesn't work," until someone shows otherwise.

## 4. Multi-layer perceptrons and why non-linearity matters

The eventual fix was exactly the two-layer idea sketched above, generalized into the **multi-layer perceptron (MLP)**: an **input layer** that receives the raw data, one or more **hidden layers** that compute intermediate features, and an **output layer** that produces the final prediction. Stacking layers like this lets the network build up complex representations out of simple ones — but only if something *non-linear* happens between the layers.

To see why, look at what happens with a two-layer network that uses **no** activation function at all:

```
a[1] = W[1] x + b[1]
a[2] = W[2] a[1] + b[2] = W[2](W[1] x + b[1]) + b[2] = (W[2]W[1]) x + (W[2]b[1] + b[2])
```

Notice that `W[2]W[1]` is just another matrix, and `W[2]b[1] + b[2]` is just another vector. In other words, two stacked *linear* layers collapse mathematically into a single linear layer — you gain nothing by stacking them. This is the single most important reason activation functions exist: they are what stops a deep network from secretly being a shallow one. Once you insert a non-linear activation function between layers, the network genuinely gains representational power with every extra layer, and can build up a hierarchy of features — for example, in an image classifier, layer 1 might detect edges and corners, layer 2 might combine those into shapes, layer 3 into object parts like eyes or wheels, and layer 4 into entire objects like faces or cars.

### The forward pass, step by step

For a small example network (3 inputs → 4 hidden neurons → 1 output), the forward pass has three steps:

1. **Weighted sum into the hidden layer:** for every hidden neuron `j`, `z_j^[1] = Σ_i (w_ji^[1] · x_i) + b_j^[1]`. In matrix form this is simply `Z[1] = W[1] X + b[1]`.
2. **Apply a non-linear activation:** commonly ReLU, `a_j^[1] = ReLU(z_j^[1]) = max(0, z_j^[1])`. In matrix form, `A[1] = ReLU(Z[1])`.
3. **Weighted sum into the output layer**, followed by whatever activation the output needs: `Z[2] = W[2] A[1] + b[2]`, `A[2] = f(Z[2])`.

Writing the computation in matrix form (`Z = WX + b`) isn't just notational convenience — it is what allows the entire forward pass for a whole batch of examples to be computed as a handful of matrix multiplications, which is exactly the kind of operation GPUs are built to do extremely fast in parallel. This is *why* deep learning became practical once GPUs became widely available.

### Common activation functions

For the **hidden layers**, three activation functions come up again and again:

- **Sigmoid**, `f(z) = 1 / (1 + e^-z)`, squashes any real number into the range (0, 1). It is smooth, but saturates for very positive or very negative inputs, which (as we'll see in Week 2) causes gradients to vanish.
- **Tanh**, `f(z) = (e^z - e^-z) / (e^z + e^-z)`, is similar to sigmoid but zero-centered, squashing values into (-1, 1).
- **ReLU** ("Rectified Linear Unit"), `f(z) = max(0, z)`, simply zeroes out negative values and passes positive values through unchanged. It is extremely cheap to compute and, importantly, does not saturate for positive inputs, which is a big part of why it became the default choice in modern deep networks.

The **output layer's** activation function, by contrast, is chosen based on *what kind of answer the problem needs*, not on training convenience:

| Problem type | Desired output | Activation |
|---|---|---|
| Regression (any real number, e.g. house price) | any real number | Linear (no activation) |
| Regression restricted to positive values (e.g. age, salary) | non-negative number | ReLU |
| Binary classification (e.g. spam or not) | a probability | Sigmoid |
| Multi-class classification (e.g. digit 0–9) | a probability distribution over classes | Softmax |

Softmax deserves a special mention because it is used so often: `a_k = e^{z_k} / Σ_j e^{z_j}`. It exponentiates every output score and then normalizes by the sum, guaranteeing that all outputs are positive and sum to exactly 1 — literally a probability distribution over the possible classes. For example, raw scores of 2.0, 1.0, and 0.5 for three classes turn into probabilities of roughly 0.55, 0.33, and 0.12 after softmax.

## 5. How a network learns: backpropagation

Once a network can make predictions (the forward pass), we need a way to make it *improve*. The training loop always follows the same four steps:

1. **Forward pass:** run the input through the network to get a prediction `ŷ`.
2. **Loss calculation:** measure how wrong the prediction was. For regression, a common choice is the **mean squared error**, `L = ½(ŷ - y)²` (the factor of ½ is just there to make the derivative come out cleanly).
3. **Backward pass:** figure out how much each individual weight contributed to that error.
4. **Weight update:** nudge every weight a little in the direction that would reduce the error.

The hard part is step 3, and this is where **backpropagation** comes in. The central question backprop answers is: *"if I wiggle this one weight buried deep inside the network, how much does the final loss change?"* The tool for answering that is the **chain rule** from calculus, which lets you break a complicated derivative into a chain of simple, local derivatives multiplied together.

Concretely, for a weight `w11^[1]` in the first hidden layer of a small 2-2-2-1 network, the chain rule says:

```
∂L/∂w11^[1] = (∂L/∂ŷ) · (∂ŷ/∂z[3]) · (∂z[3]/∂a1^[2]) · (∂a1^[2]/∂z1^[2]) · (∂z1^[2]/∂a1^[1]) · (∂a1^[1]/∂z1^[1]) · (∂z1^[1]/∂w11^[1])
```

That looks intimidating, but every single factor in that chain is something easy to compute on its own: `∂L/∂ŷ = ŷ - y` (how far off the prediction was), `∂ŷ/∂z[3] = 1` (because the output activation was linear), each `∂z/∂a` term is just the weight connecting those two neurons, and each `∂a/∂z` term is the derivative of the activation function (for ReLU, this is simply 1 if the input was positive and 0 if it was zero or negative — ReLU's derivative acts like a gate that is either fully open or fully shut). Backpropagation's efficiency trick is to compute these terms **starting from the output and working backward**, reusing already-computed pieces (often called "delta" terms) instead of recomputing the whole chain from scratch for every single weight. That's what makes it possible to train networks with millions or billions of weights.

Once every weight's gradient `∂L/∂w` has been computed, **gradient descent** performs the actual update:

```
w_new = w_old - η · (∂L/∂w)
```

The gradient points in the direction of *steepest increase* of the loss, so subtracting it moves the weight in the direction that decreases the loss. All weights are updated simultaneously, and the whole four-step cycle — forward pass, loss, backward pass, update — is repeated over and over on the training data until the loss stops meaningfully decreasing. This is why backpropagation plus gradient descent is often described as the single algorithm that trains virtually every deep learning model in this course, from a two-layer toy network to a modern transformer with billions of parameters. It is efficient (all gradients are obtained in one backward sweep), modular (each layer's computation only needs local information), and completely general (it works for any network built out of differentiable pieces).

## 6. The second AI winter, and the modern deep learning revolution

Even after the XOR problem was solved conceptually with multi-layer networks, a second period of skepticism set in during the late 1990s and early 2000s. Computers were still too slow, datasets were still too small, deep networks suffered badly from the **vanishing gradient problem** (long chains of small derivatives multiplying together and shrinking toward zero, so early layers barely learned — the subject of next week), and competing methods like Support Vector Machines (SVMs) were winning benchmarks with far less computational cost. Funding and enthusiasm for neural networks dropped again.

What eventually broke this second winter was a genuine "perfect storm" of three simultaneous developments:

- **Hardware:** GPUs, originally built for rendering video game graphics, turned out to be extremely good at the massive parallel matrix multiplications that neural networks need.
- **Data:** the arrival of huge labeled datasets, most famously **ImageNet** with about 14 million labeled images, gave deep networks enough examples to actually make use of their capacity.
- **Algorithms:** better tools like the ReLU activation (which resists vanishing gradients), dropout regularization (Week 2), and improved optimizers like Adam (Week 2) made deep networks much easier to train reliably.

That combination launched the architectures that make up the rest of this course: **CNNs** (Weeks 3–5) for spatial data like images, exploiting local connectivity and spatial hierarchies; **RNNs and LSTMs** (Weeks 6–7) for sequential data like time series and language; and **Transformers** (Week 13), which use a self-attention mechanism and now underpin state-of-the-art natural language processing and large language models (Week 14).

## Key takeaways

Every neural network, no matter how large, still boils down to the same recipe you learned this week: take a weighted sum of inputs, add a bias, pass the result through a non-linear activation function, and stack many such units in layers so that later layers can build on the features earlier layers discovered. Training such a network is always the same four-step loop — forward pass, compute the loss, backpropagate the gradients with the chain rule, and update the weights with gradient descent. Historically, the field has swung between periods of overclaiming and periods of unfair pessimism (the two AI winters); today's deep learning boom is the result of finally having the data, hardware, and algorithmic tricks needed to make this decades-old idea work at scale. Keep this week's vocabulary — weights, bias, activation, forward pass, loss, backward pass, gradient descent, learning rate — close at hand, because every remaining week of the course reuses it.
