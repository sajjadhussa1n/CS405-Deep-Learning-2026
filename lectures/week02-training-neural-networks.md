# Week 2 — Training Neural Networks: A Deep Dive

*Companion notes for [`slides/lecture_week02.pdf`](../slides/lecture_week02.pdf)*

## Why this week matters

Week 1 gave us the recipe for building a network and the backpropagation algorithm for training it. In practice, however, simply writing down that recipe and hitting "train" often fails: gradients disappear before they reach the early layers, training is unstable, and the model overfits the training set. This week is a toolbox of the practical fixes that turn "a network that can theoretically learn" into "a network that actually trains well": smarter activation functions, smarter weight initialization, batch normalization, transfer learning, better optimizers, and regularization. Nearly every architecture in the rest of the course leans on this toolbox.

## 1. The vanishing gradient problem, precisely

Recall from Week 1 that a gradient for an early weight is a long product of terms, chained together by the chain rule:

```
∂L/∂w(1) = (∂L/∂a(L)) · (∂a(L)/∂a(L-1)) · ... · (∂a(2)/∂a(1)) · (∂a(1)/∂z(1)) · (∂z(1)/∂w(1))
```

Every one of the `∂a(l)/∂a(l-1)` terms depends on the derivative of the activation function, `g'(z)`. If that derivative is consistently less than 1, then multiplying many of them together (one per layer) makes the overall product shrink **exponentially** with depth — this is the vanishing gradient problem, and it means early layers receive a gradient signal so tiny that they barely update, effectively freezing them while later layers keep learning.

**Sigmoid is the classic culprit.** Its derivative is `σ'(x) = σ(x)(1 - σ(x))`, and this expression peaks at exactly 0.25 (when `σ(x) = 0.5`) and rapidly approaches 0 whenever `|x|` gets larger than about 4 (the sigmoid saturates, becoming flat near 0 or 1). So even in the *best* case, every layer using sigmoid multiplies the gradient by at most 0.25 — after just a handful of layers, the gradient has all but vanished.

**ReLU is the fix.** Its derivative is trivially 1 for any active neuron (`x > 0`) and 0 otherwise. For the neurons that are "on," the chain-rule product becomes `1 × 1 × ... × 1 = 1` instead of shrinking — the gradient passes through those layers essentially undiminished. This single property is a large part of why ReLU became the default hidden-layer activation once networks started getting deep.

## 2. Smart weight initialization

ReLU alone isn't enough — *how you initialize the weights before training even starts* also determines whether the signal explodes or vanishes as it flows through a deep network, purely from the forward pass. The intuition: we want the **variance** of a layer's outputs to match the variance of its inputs, layer after layer, so activations neither blow up nor shrink to nothing as depth increases; the same logic applies to gradients flowing backward.

For a simple linear layer `z = Wx + b` with zero-mean weights and inputs, it can be shown that `Var(z) = n_in · Var(W) · Var(x)`. For the output variance to equal the input variance, we need `Var(W) = 1/n_in`. Working out the analogous condition for the backward pass and finding a compromise between the two leads to two standard initialization schemes:

- **Glorot / Xavier initialization** (designed for **tanh/sigmoid**): samples weights from `N(0, 2/(n_in + n_out))` or a matching uniform distribution. It balances the variance requirement for both the forward and backward passes.
- **He initialization** (designed for **ReLU** and its variants): samples weights from `N(0, 2/n_in)`. It uses a larger variance than Glorot because ReLU zeroes out roughly half of its inputs, which would otherwise halve the effective variance at each layer.

The practical rule of thumb: match your initialization scheme to your activation function — He initialization for ReLU-family activations, Glorot/Xavier for tanh or sigmoid. Getting this wrong can make an otherwise-correct network fail to train at all.

## 3. Beyond plain ReLU

Plain ReLU has its own flaw, sometimes called the "dying ReLU" problem: if a neuron's weighted input `z` ends up negative for every training example, its gradient is *always* exactly 0, so that neuron can never update again — it's permanently "dead." Several variants patch this:

- **Leaky ReLU:** instead of a hard 0 for negative inputs, uses a small slope `f(x) = x if x > 0, else αx` (with a small fixed constant like `α = 0.01`), so a "dead" neuron still receives a tiny gradient and can recover.
- **Parametric ReLU (PReLU):** the same shape as Leaky ReLU, but `α` is no longer a fixed hyperparameter — it is *learned* during training, letting the network decide the best slope for negative inputs.
- **Exponential Linear Unit (ELU):** for negative inputs, uses a smooth exponential curve `α(e^x - 1)` instead of a straight line. This gives a smooth transition around zero (unlike the sharp corner of ReLU) and tends to push the mean activation of a layer closer to zero, which helps training.
- **Scaled Exponential Linear Unit (SELU):** a rescaled version of ELU with carefully chosen constants (`α ≈ 1.6733`, `λ ≈ 1.0507`) that gives it a remarkable **self-normalizing** property — under the right conditions (paired with LeCun initialization), the outputs of each layer automatically converge toward zero mean and unit variance as they pass through the network, which inherently resists both vanishing and exploding gradients without needing a separate normalization layer. It works particularly well in deep, purely feedforward networks.

## 4. Batch normalization

Even with a good activation function and initialization, the *distribution* of activations at each layer keeps shifting during training simply because the parameters of every earlier layer are also changing — this phenomenon is called **internal covariate shift**, and it slows training down because each layer is constantly having to re-adapt to a moving target.

**Batch Normalization (BatchNorm)** fixes this by explicitly re-normalizing the pre-activation values `z` of a layer, using the statistics of the current mini-batch, before the activation function is applied. Given a mini-batch of pre-activations `B = {z_1, ..., z_m}`, the algorithm is:

1. Compute the batch mean and variance: `μ_B = (1/m) Σ z_i` and `σ_B² = (1/m) Σ (z_i - μ_B)²`.
2. Normalize: `ẑ_i = (z_i - μ_B) / √(σ_B² + ε)` (the small constant `ε` just avoids dividing by zero).
3. **Scale and shift** with two *learnable* parameters: `BN(z_i) = γẑ_i + β`.

That third step is easy to overlook but essential: if the network simply forced every layer's output to be zero-mean and unit-variance, it could destroy useful information (some layers genuinely need larger or off-center activations). By making `γ` and `β` learnable, BatchNorm lets the network *choose* to undo the normalization if that turns out to be better, while still getting the training benefits of normalization by default.

BatchNorm helps in three concrete ways: it lets you use **higher learning rates** without the risk of the update blowing up, it makes training **less sensitive to the exact initialization** chosen, and it has a mild **regularizing effect**, because each training example's normalized value depends on the other examples in its mini-batch, injecting a bit of useful noise that reduces overfitting. One subtlety worth remembering: at *test time* there usually is no "mini-batch," so BatchNorm instead uses running averages of the mean and variance that were accumulated during training.

## 5. Transfer learning

Training a large network from scratch requires huge datasets and compute. **Transfer learning** asks: *why train from scratch when someone has already done the hard work on a similar problem?* You take a network pre-trained on a large source dataset (e.g., ImageNet) and adapt it to your own, usually smaller, target dataset. The right strategy depends on how much target data you have and how similar it is to the source domain:

1. **Small target dataset, similar to source:** freeze all the convolutional (feature-extracting) layers and train only a new classifier head on top. The pre-trained features are reused as-is.
2. **Medium target dataset, similar to source:** freeze the early layers (which tend to learn generic features like edges and textures) but fine-tune the later layers along with the classifier head.
3. **Large target dataset, similar to source:** fine-tune the *entire* network. Here the pre-trained weights are essentially just being used as a very good starting point (a smart initialization) rather than being frozen.
4. **Target dataset not similar to the source domain:** transfer learning is less likely to help, since the pre-trained features may not be relevant; training from scratch may be the better option.

We will see this technique used constantly with the CNN architectures introduced in Weeks 3–5.

## 6. Optimizers — smarter ways to take the gradient-descent step

Plain (vanilla) stochastic gradient descent updates parameters using only the current gradient: `θ_{t+1} = θ_t - η·g_t`. This is myopic — it only ever looks at the single gradient at this exact moment — and it has a very practical downside: as training approaches a minimum, the gradient naturally shrinks, so the step size shrinks too, and training can crawl to a halt long before actually reaching the minimum. A whole family of optimizers exists to fix this.

- **SGD with Momentum:** imagine a heavy ball rolling downhill — it doesn't just follow the instantaneous slope, it also carries momentum from where it's already been. We track a velocity `v_t = β·v_{t-1} + η·g_t` and update `θ_{t+1} = θ_t - v_t`, where `β` (commonly 0.9) controls how much past velocity is retained. Unrolling this recursion shows that `v_t` is really a weighted sum of *all* past gradients, with more recent gradients weighted more heavily. This gives three benefits: faster convergence in directions where the gradient consistently points the same way (the contributions add up), reduced oscillation in narrow ravines (steep walls, gentle floor) because opposing side-to-side gradients cancel out while the consistent downhill gradient keeps accumulating, and the ability to keep moving through flat regions where the instantaneous gradient is nearly zero.
- **Nesterov Accelerated Gradient (NAG):** a "look-ahead" refinement of momentum. Instead of computing the gradient at the current position, NAG first takes a tentative jump using the existing velocity, `θ_look-ahead = θ_t - β·v_{t-1}`, then computes the gradient *at that look-ahead point*, and finally updates velocity and parameters using that look-ahead gradient. This gives the optimizer a "preview" of where the momentum is about to carry it — if that jump is about to overshoot and increase the loss, the look-ahead gradient will already reflect that and correct the step before it's fully taken.
- **RMSProp:** rather than using one global learning rate for every parameter, RMSProp adapts the learning rate **per parameter**, based on how large that parameter's gradients have recently been. It keeps an exponentially decaying average of squared gradients, `E[g²]_t = β·E[g²]_{t-1} + (1-β)·g_t²`, and divides the learning rate by its square root: `θ_{t+1} = θ_t - (η / √(E[g²]_t + ε)) · g_t`. The effect: parameters that have been receiving large, frequent gradients get their effective learning rate shrunk (to avoid overshooting), while parameters with small, infrequent gradients get a relatively larger effective learning rate (so they aren't left behind).
- **Adam (Adaptive Moment Estimation):** the "best of both worlds," combining momentum's idea (tracking a first moment / running average of the gradient itself) with RMSProp's idea (tracking a second moment / running average of the squared gradient). Adam maintains both `m_t = β1·m_{t-1} + (1-β1)·g_t` and `v_t = β2·v_{t-1} + (1-β2)·g_t²`, then applies **bias correction** — dividing by `(1 - β1^t)` and `(1 - β2^t)` respectively — because `m_0` and `v_0` both start at zero, which would otherwise bias the early estimates toward zero. The final update is `θ_{t+1} = θ_t - (η / (√v̂_t + ε)) · m̂_t`. With typical defaults `β1 = 0.9`, `β2 = 0.999`, `ε = 10⁻⁸`, Adam is usually a strong, low-effort default optimizer, and you will see it used throughout this course.

## 7. Regularization — fighting overfitting

A model that fits its training data extremely well but generalizes poorly to new data is **overfitting**. Regularization techniques deliberately constrain the model to prevent this.

- **L2 regularization (weight decay / Ridge):** adds the *squared* magnitude of the weights to the loss function, `J_reg(θ) = J(θ) + (α/2)‖θ‖₂²`. Its gradient contribution is simply `αθ`, meaning every weight update also shrinks that weight proportionally toward zero. This discourages any single weight from growing very large, favoring smoother, simpler functions.
- **L1 regularization (Lasso):** adds the *absolute* magnitude of the weights instead, `J_reg(θ) = J(θ) + α‖θ‖₁`. Unlike L2, L1 tends to push many weights all the way to exactly zero, effectively performing feature selection by producing sparse weight vectors.
- **Dropout:** during training, for each forward pass, randomly "turn off" (set to zero) a fraction `p` of the neurons in a layer. Every mini-batch therefore trains a different random sub-network, which prevents neurons from co-adapting too tightly to specific other neurons ("complex co-adaptations"), forcing the network to learn more redundant, robust representations. At test time, the *full* network is used, but to keep the expected magnitude of activations consistent with training, the weights are scaled down by `(1 - p)` (equivalently, activations can instead be scaled up by `1/(1-p)` during training — "inverted dropout"). Conceptually, dropout behaves like training a huge ensemble of different sub-networks and then averaging their predictions at test time, which is a large part of why it is such an effective and widely used regularizer.

## Key takeaways

Training a deep network reliably is not automatic — it requires actively fighting the vanishing gradient problem (with ReLU-family activations and matching initialization schemes like He or Glorot), stabilizing the distribution of activations layer to layer (BatchNorm), reusing knowledge from related problems instead of starting from zero (transfer learning), taking smarter optimization steps than plain gradient descent (momentum, NAG, RMSProp, and especially Adam), and explicitly discouraging overfitting (L1/L2 weight penalties and dropout). Treat this week's techniques as your default toolbox: whenever a network in a later week "isn't training well," the first things to check are almost always initialization, activation choice, normalization, optimizer choice, and regularization — exactly the five ingredients covered here.
