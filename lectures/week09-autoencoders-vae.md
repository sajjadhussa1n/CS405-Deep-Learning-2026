# Week 9 — Autoencoders and Variational Autoencoders (VAEs)

*Companion notes for [`slides/lecture_week09.pdf`](../slides/lecture_week09.pdf)*

## Why this week matters

Everything so far in the course has been supervised: given labeled inputs and outputs, learn the mapping between them. This week starts a shift toward **unsupervised** and **generative** modeling — networks that learn structure from unlabeled data, and eventually networks that can *create* new data resembling what they were trained on. Autoencoders are the entry point, and Variational Autoencoders (VAEs) are the first genuinely generative model in the course, setting up the comparison with GANs next week.

## 1. What is an autoencoder?

An autoencoder is a neural network trained to learn efficient, compressed representations of data in a purely unsupervised way, by learning to reconstruct its own input. It has three parts: an **encoder** that compresses the input `x` down into a smaller representation `z` (the **latent code**), a **bottleneck** — the point where `z` lives, deliberately narrower than the input — and a **decoder** that tries to reconstruct the original input as `x̂` from that compressed code alone. Because the bottleneck is narrower than the input, the network is *forced* to discard unimportant details and keep only the information that matters most for reconstruction — this pressure is what makes the learned code `z` a genuinely useful, compressed representation rather than a trivial copy.

Concretely, a simple (one hidden layer) autoencoder computes:

```
z  = f_enc(x) = σ(W_e x + b_e)     # encoder
x̂ = f_dec(z) = σ(W_d z + b_d)     # decoder
```

and is trained purely by minimizing the **reconstruction error** — how different the output is from the original input, most commonly measured with mean squared error:

```
L(x, x̂) = ‖x - x̂‖²
min_θ  Σ_i ‖x_i - x̂_i‖²
```

Notice there are no labels anywhere in this objective — the "label" for each training example is simply the input itself. This is why autoencoders are described as **unsupervised**.

## 2. Understanding the latent space

The **latent space** is the space that the compressed code `z` lives in — a lower-dimensional, more structured representation of the original high-dimensional, complex data space, much like how a ZIP file compresses a large document while preserving its essential content. A well-trained latent space tends to exhibit useful properties: **compactness** (data represented in far fewer dimensions than the original), **continuity** (similar inputs map to nearby points in latent space), and, ideally, **disentanglement** (different latent dimensions capture different, semantically meaningful factors of variation — for face images, one dimension might control smile, another age, another whether the person is wearing glasses). Visualizing the latent space of a digit-recognition autoencoder trained on MNIST typically shows digit classes clustering together in distinct regions — even though the autoencoder was never told which images belonged to which digit class, similarity in *pixel space* naturally translates into proximity in *latent space*.

### Stacked and convolutional variants

A **stacked (deep) autoencoder** simply adds multiple hidden layers to both the encoder and decoder, giving the network more capacity to learn hierarchical features and generally producing higher-quality reconstructions, at the cost of more parameters to train. For image data specifically, a **convolutional autoencoder** replaces the fully connected layers with convolutional layers (Week 3) in the encoder — progressively reducing spatial dimensions while increasing depth/channels, down to a flattened latent vector `z` — and with transposed convolutions (Week 5) in the decoder, upsampling back to the original spatial dimensions. This lets the network exploit the same local-connectivity and parameter-sharing advantages that made CNNs effective for classification, now in service of learning efficient spatial representations without discarding structural information the way a naive flatten-and-reconstruct approach would.

## 3. What autoencoders are used for

Two classic applications stand out. **Dimensionality reduction**: unlike PCA, which can only learn a *linear* projection of the data, an autoencoder can learn a *non-linear* manifold, giving it more flexibility to represent complex data structures — a 100×100 pixel image (10,000 dimensions) might be compressed into a 100-dimensional latent code, a 100× compression, while capturing more of the image's true structure than a linear method could. **Unsupervised pre-training**: historically, before modern initialization and regularization techniques (Week 2) made training deep networks from scratch reliable, a common strategy was to first train an autoencoder unsupervised on large amounts of unlabeled data, discard its decoder, keep its encoder weights as a starting point, and then attach a classifier on top and fine-tune the whole thing with a smaller labeled dataset. The intuition is the same as transfer learning (Week 2): the encoder has already learned generally useful features from the data distribution, giving the supervised model a "head start" rather than starting from random weights — analogous to already knowing how to recognize edges, shapes, and textures before learning to recognize specific objects.

## 4. The problem: autoencoders cannot generate new data

A natural question: if the decoder can turn a latent code `z` into a realistic-looking `x̂`, could we generate *brand-new* data by feeding the decoder some random point in latent space instead of an encoded real image? In practice, this fails badly. A plain autoencoder's encoder maps each real training input to a single, specific point in latent space, with **no constraint at all** on how those points are distributed. The result is a latent space that is sparse and discontinuous — most of the space corresponds to no real training example, with gaps between the regions that *do* correspond to real data. A randomly sampled point is very likely to land in one of those empty gaps, and the decoder, having never seen anything like it during training, produces meaningless, nonsensical output. In short: **for generation to work, the entire latent space needs to be meaningful, not just the isolated points that happen to correspond to real training examples.**

## 5. Variational Autoencoders: making the encoder probabilistic

The Variational Autoencoder's (VAE) core fix is elegant: instead of the encoder mapping each input to a single deterministic point `z = f(x)`, it maps each input to a **probability distribution** over latent codes, typically a Gaussian: `z ~ N(μ(x), σ²(x))`. In other words, the encoder now outputs two vectors — a mean `μ` and a standard deviation `σ` — describing a small "cloud" of plausible latent codes for that input, rather than one exact point.

### The reparameterization trick

There's an immediate technical obstacle: if `z` is obtained by randomly *sampling* from `N(μ, σ²)`, that sampling step is not differentiable, so gradients cannot flow back through it during backpropagation — training would be stuck. The **reparameterization trick** sidesteps this neatly: instead of sampling `z` directly, sample a fixed, external source of randomness `ε ~ N(0, I)` (which doesn't depend on the network's parameters at all), and compute:

```
z = μ + σ ⊙ ε
```

Now `z` is a differentiable function of `μ` and `σ` (both produced by the encoder, and both trainable), with all of the actual randomness pushed into `ε`, which requires no gradient. Gradients can flow cleanly through `μ` and `σ` back into the encoder's weights, while the network still produces a genuinely stochastic latent code at every forward pass.

### The VAE loss function (ELBO)

VAEs are trained by maximizing the **Evidence Lower Bound (ELBO)**, equivalently minimizing:

```
L_VAE = -E_{z~q(z|x)}[log p_θ(x|z)]  +  D_KL(q_φ(z|x) ‖ p(z))
        \_____________________/        \_____________________/
           reconstruction term                regularization term
```

In practice, for image data, this becomes:

```
L_VAE = MSE(x, x̂)  +  D_KL(q(z|x) ‖ N(0, 1))
```

The **reconstruction loss** term (MSE for continuous data, binary cross-entropy for binary data) does the same job as in a plain autoencoder — pushing `x̂` to resemble `x`, preserving information. The **KL divergence** term is new, and it is what makes the crucial difference: it measures how much the encoder's distribution `q(z|x)` differs from a simple, fixed prior distribution `N(0, 1)`, and penalizes that difference — pushing every input's latent distribution to stay close to a shared, standard Gaussian, which is what "organizes" the latent space into something usable for generation.

For two Gaussians, `q(z|x) = N(μ, σ²)` and `p(z) = N(0, 1)`, the KL divergence has a clean closed form:

```
D_KL = -½ Σ_j [1 + log(σ_j²) - μ_j² - σ_j²]
```

Each term has an interpretable role: the `μ²` term penalizes the mean drifting away from 0, the `σ²` term penalizes the variance drifting away from 1, and `log(σ²)` provides balance between the two so the penalty is well-behaved for both very small and very large variances. In practice, the encoder is designed to output `log(σ²)` directly rather than `σ` or `σ²`: since a network's raw output is an unconstrained real number, and `σ²` must be strictly non-negative, having the network predict `log(σ²)` (which can be *any* real number, positive or negative) sidesteps that constraint entirely — you simply recover `σ² = exp(log(σ²))` afterward. This gives better numerical stability and lets standard, unconstrained gradient descent optimize it directly.

### How the KL term reshapes the latent space

Without any KL regularization (a plain autoencoder), each input's latent code sits at an isolated point, leaving the space sparse and disconnected with large empty gaps. With the KL term, training simultaneously pushes every `μ` toward 0 and every `σ²` toward 1 — the latent distributions from *different* inputs are pulled close enough together that they start to **overlap**. This overlap is precisely what produces a smooth, continuous, densely-covered latent space: because nearby (and even randomly sampled) points in latent space are now likely to fall within the "cloud" that some real training example was mapped to, the decoder has actually seen similar codes during training and can produce a sensible output for them. This is what finally enables both **interpolation** (smoothly morphing between two data points by walking through latent space) and true **generation** (sampling `z ~ N(0,1)` from scratch, with no input image at all, and decoding it into a new, plausible output).

### Training vs. generation

During **training**, both the encoder and decoder are used together: encode `x` into a distribution, sample `z`, decode back into `x̂`, and optimize the combined reconstruction + KL loss. During **generation**, the encoder is discarded entirely — you simply sample `z ~ N(0, 1)` directly from the prior and pass it through the trained decoder alone to produce a brand-new sample.

## 6. Autoencoder vs. VAE — key differences

| Property | Autoencoder | VAE |
|---|---|---|
| Encoder output | single point `z` | distribution `N(μ, σ)` |
| Latent space | sparse, disconnected | continuous, dense |
| Can generate new data? | no | yes |
| Interpolation | not smooth | smooth |
| Regularization | none | KL divergence |

A useful analogy from the slides: a plain autoencoder is like memorizing the exact addresses of your friends — useful for finding *them*, but useless for finding a new place you've never been. A VAE is like understanding the layout of the city itself — because you understand the structure, you can find your way to entirely new places you've never visited before.

## 7. Applications of VAEs

VAEs are used for **image generation** (producing new, plausible images from a trained model), **data augmentation** (generating additional synthetic training examples to supplement a limited dataset), **anomaly detection** (an input that reconstructs poorly, or whose encoding lands in a low-probability region of latent space, is likely anomalous relative to the training distribution), and **drug discovery** (generating new candidate molecular structures by sampling and decoding from a latent space trained on known molecules).

## Key takeaways

An autoencoder learns a compressed latent representation of its input purely by being trained to reconstruct that input through a narrow bottleneck, with applications in dimensionality reduction and unsupervised pre-training — but because nothing constrains where the latent codes end up, its latent space is sparse and full of gaps, so it cannot be used to generate new, realistic data by sampling randomly. The Variational Autoencoder fixes this by making the encoder output a *distribution* over latent codes rather than a single point, using the reparameterization trick (`z = μ + σ⊙ε`) to keep the sampling step differentiable, and adding a KL-divergence regularization term that pulls every input's latent distribution toward a shared standard Gaussian prior. That regularization is what turns a sparse, disconnected latent space into a smooth, continuous one where every point decodes to something plausible — the property that finally makes generation, not just reconstruction, possible. Next week's GANs offer a very different (adversarial, rather than probabilistic) route to the same generative goal, and we'll compare the trade-offs directly.
