# Week 10 — Generative Adversarial Networks (GANs)

*Companion notes for [`slides/lecture_week10.pdf`](../slides/lecture_week10.pdf)*

## Why this week matters

Last week's VAEs generate new data by sampling from a regularized latent space and decoding — but VAEs have a known weakness: they tend to produce blurry outputs, because a pixel-by-pixel reconstruction loss (like MSE) doesn't capture *global* realism, and averaging over the possible outputs that fit an input tends to wash out sharp detail. GANs take a completely different approach: instead of *defining* mathematically what "realistic" means (as MSE tries to), they let a second network *learn* what realistic means, by pitting two networks against each other in direct competition.

## 1. The core idea: learning by competing

The canonical analogy is an art forger versus an art expert. The **forger (Generator, G)** tries to create fake paintings realistic enough to pass as genuine. The **expert (Discriminator, D)** tries to tell real paintings from fakes. Both improve purely through this competition: the forger gets better at fooling the expert, which forces the expert to get better at detecting fakes, which in turn forces the forger to improve further — and so on, until, ideally, the forger's fakes become indistinguishable from the real thing even to a skilled expert.

## 2. Basic GAN architecture

The **Generator (G)** takes random noise `z` (typically drawn from a simple distribution like a Gaussian) as input and outputs a fake sample, `x̂ = G(z)`; its goal is to fool the discriminator. The **Discriminator (D)** takes either a real image `x` or a generated fake `x̂` as input and outputs a single probability, `D(x) ∈ [0, 1]`, representing how confident it is that the input is real; its goal is to correctly tell real from fake. During training, gradients flow backward through both networks: through the discriminator to improve its ability to detect fakes, and — critically — *through the discriminator and back into the generator* to tell the generator how it should change its output to be more convincing.

## 3. The minimax training objective

GAN training is framed as a two-player minimax game with the value function:

```
min_G max_D V(D, G) = E_{x~p_data}[log D(x)] + E_{z~p_z}[log(1 - D(G(z)))]
```

Reading the notation: `x ~ p_data` means real images sampled from the true data distribution, `z ~ p_z` means noise sampled from a simple prior (e.g., Gaussian), `G(z)` is the generator's fake image, `D(·)` is the discriminator's real-vs-fake probability, and `E[·]` denotes an expectation (an average over many samples). The **discriminator** tries to *maximize* `V(D,G)`: it wants `D(x) ≈ 1` for real images (so `log D(x) ≈ log(1) = 0`, its maximum possible value) and `D(G(z)) ≈ 0` for fake images (so `log(1 - D(G(z))) ≈ log(1) = 0` as well). The **generator** tries to *minimize* `V(D,G)` — since it has no control over the real-image term, this reduces to minimizing `log(1 - D(G(z)))`, which is achieved by pushing `D(G(z)) → 1`, i.e., successfully fooling the discriminator into calling its fakes "real."

This is a **zero-sum game**: what the generator gains, the discriminator loses. The theoretical resting point of this competition is a **Nash equilibrium**, where the generator produces samples so realistic that the discriminator can do no better than random guessing — `D(x) = 0.5` for every input, real or fake.

### A practical training subtlety: the vanishing-gradient version of the generator loss

Early in training, the discriminator is usually much better than the generator, so `D(G(z)) ≈ 0` for the generator's (still poor) fakes. Plugging this into the generator's loss, `log(1 - D(G(z))) ≈ log(1 - 0) = log(1) = 0` — the loss is already near its *minimum* possible value, meaning its gradient is nearly flat, and the generator gets almost no useful signal about how to improve, even though its outputs are clearly bad. (In practice, this is usually worked around by training the generator to instead maximize `log D(G(z)))` directly, which provides a much stronger gradient early in training, though the slides present the vanishing-gradient issue as a first motivation for the more principled Wasserstein fix described in Section 5.)

### Practical training loop: alternating updates

GANs are trained by alternating between two update steps at each iteration. **Update the discriminator:** sample a batch of real images and a batch of noise, generate fakes from the noise, and update `D`'s weights (only) to *maximize* `(1/m) Σ [log D(x⁽ⁱ⁾) + log(1 - D(G(z⁽ⁱ⁾)))]`. **Update the generator:** sample a fresh batch of noise, and update `G`'s weights (only) to *minimize* `(1/m) Σ log(1 - D(G(z⁽ⁱ⁾)))`. The two networks are never trained simultaneously on the exact same objective at the exact same moment — they take turns, each treating the other's current weights as fixed.

## 4. DCGAN — bringing convolutions to GANs

A plain, fully-connected (MLP-based) GAN struggles badly on image data for a familiar reason from Week 3: MLPs ignore spatial structure entirely, treating every pixel as an independent feature, which leads to unstable training and poor image quality. **DCGAN (Deep Convolutional GAN)**, introduced by Radford et al. in 2015, fixes this by replacing the MLP layers with CNN-style layers throughout, following five key architectural guidelines: replace pooling with **strided convolutions** in the discriminator (downsampling) and **transposed convolutions** (Week 5) in the generator (upsampling); use **batch normalization** (Week 2) in both networks to stabilize training; remove fully connected hidden layers entirely; and use specific, carefully chosen activation functions.

Concretely, the **DCGAN generator** starts from a 100-dimensional noise vector, projects it (via a dense layer) into a small spatial feature map (e.g., `4×4×1024`), and then repeatedly applies transposed convolutions that each double the spatial size while halving the channel depth (`4×4×1024 → 8×8×512 → 16×16×256 → 32×32×128 → 64×64×3`), using ReLU activations throughout except for a `tanh` output activation (matching pixel values normalized to `[-1, 1]`), with batch normalization after every layer and no pooling anywhere. The **DCGAN discriminator** mirrors this in reverse: starting from a `64×64×3` image, repeated strided convolutions each halve the spatial size while doubling the channel depth, ending in a flattened single probability output, using LeakyReLU (Week 2, `α=0.2`) throughout and a sigmoid on the final output. DCGAN's results — photorealistic 64×64 faces, coherent bedroom and building images, and the ability to do meaningful vector arithmetic in latent space (more on this in Section 7) — made it the de facto standard GAN architecture, paving the way for later models like StyleGAN and BigGAN.

## 5. Two major training problems, and the Wasserstein fix

Standard GAN training is notoriously unstable, primarily due to two failure modes.

**Vanishing gradients:** as described above, once the discriminator becomes very good, `D(G(z)) ≈ 0` for the generator's outputs, and the generator's loss saturates near its minimum, providing essentially zero useful gradient — the generator simply stops learning, and training stagnates.

**Mode collapse:** the generator discovers a single "trick" — one type of output — that reliably fools the current discriminator, and then keeps producing variations of essentially that *same* output over and over, regardless of the input noise `z`, rather than covering the full diversity of the real data distribution. This happens because the standard GAN objective only ever asks the generator to fool the discriminator on the samples it currently produces — nothing in the loss directly rewards the generator for covering the *entire* distribution. The slides describe this with a "lazy student" analogy: a student who finds one trick that reliably passes the teacher's test never has to learn the full subject material, and gets stuck at that local optimum.

**Wasserstein GAN (WGAN)** addresses both problems at once by replacing the log-based loss with the **Earth Mover's (Wasserstein-1) distance** — intuitively, the minimum "work" (amount of probability mass moved, times the distance moved) required to transform the generated distribution into the real data distribution, like the minimum effort needed to reshape one pile of dirt into another. Unlike the log loss, which can saturate (flatten out) when the real and fake distributions don't overlap at all, the Wasserstein distance provides a smooth, informative gradient *everywhere*, even when the two distributions are completely disjoint — it always tells you "how far apart" they still are.

In WGAN, the discriminator is renamed the **critic (C)** and, crucially, no longer outputs a bounded probability — it outputs an unbounded real-valued score, `C(x) ∈ ℝ`, with no sigmoid activation and no logarithms anywhere in the loss:

```
min_G max_C  E_{x~p_data}[C(x)] - E_{z~p_z}[C(G(z))]
```

The critic tries to **maximize the gap** between its scores on real data and on fake data (equivalently, minimize `L_C = -[E[C(x)] - E[C(G(z))]]`), while the generator tries to make its fakes score as highly as possible under the critic, minimizing `L_G = -E_z[C(G(z))]` — effectively trying to close that gap. Because Wasserstein distance shrinks whenever the generated distribution covers the real data's modes *more completely*, this loss directly penalizes mode collapse (a generator collapsed onto one mode has a large, persistent Wasserstein distance to the full real distribution), and because the score `C(x)` is unbounded, it never saturates to a flat gradient the way a bounded sigmoid probability can.

There is one important catch: for the Wasserstein-distance theory to hold, the critic function must satisfy a **1-Lipschitz constraint** — informally, its output can't change *too* quickly as its input changes (`|f(x_1) - f(x_2)| ≤ ‖x_1 - x_2‖`). The original WGAN paper enforced this crudely via **weight clipping** (forcibly clamping the critic's weights to a small range after each update); the improved **WGAN-GP** enforces it more gracefully with a **gradient penalty** term added to the loss, `λ · E_x̂[(‖∇_x̂ D(x̂)‖ - 1)²]`, which directly penalizes the critic's gradient norm for straying from 1.

| Property | Standard GAN | WGAN |
|---|---|---|
| Loss function | log loss | Wasserstein distance |
| Discriminator/critic output | probability `[0,1]` | unbounded score `(-∞, ∞)` |
| Output activation | sigmoid | linear (none) |
| Gradient saturation | yes | no |
| Mode collapse | common | rare |
| Training stability | often unstable, oscillating | smoother, more stable |
| Loss correlates with sample quality | no | yes |

That last row matters practically: in a standard GAN, the discriminator and generator losses oscillate and don't reliably tell you whether training is actually progressing, whereas WGAN's loss tends to decrease smoothly and correlates with actual sample quality — giving practitioners a genuinely useful training signal to monitor, rather than just eyeballing generated samples.

## 6. Conditional GANs — controlling what gets generated

A plain GAN gives you *no control* over what it generates — feed in random noise `z`, get a random image out, with no way to ask specifically for "a dog" instead of "a cat." The **Conditional GAN (cGAN)** fix: give both the generator and the discriminator access to extra **condition information `y`** (e.g., a class label), so the generator learns to produce outputs matching whatever condition it's given, and the discriminator learns to check not just "is this real?" but "does this real-or-fake image actually match its stated condition?"

On the **generator side**, the condition `y` (e.g., a one-hot class label) is simply **concatenated** with the noise vector `z` before being fed into the network — for example, a 100-dimensional noise vector plus a 10-dimensional one-hot MNIST digit label becomes a 110-dimensional combined input, and the generator learns "given this noise and this label, produce an image of that specific digit." On the **discriminator side**, the condition is typically added as **extra channels** to the input image — for MNIST, a `28×28×1` image is concatenated with a `28×28×10` condition map (the one-hot label broadcast across every spatial position) to give a `28×28×11` input, letting the discriminator directly check whether the image content is consistent with the stated label, and penalize the generator if it isn't.

An important subtlety: even with a fixed condition `y`, varying the noise `z` still produces different outputs — the intuition being that `y` controls the *class/category/content* of the output, while `z` controls remaining stylistic variation (pose, expression, exact handwriting style, and so on). Together, `y` and `z` give fine-grained control over both *what* is generated and *how* it looks.

## 7. Exploring and manipulating the latent space

Even in an unconditional GAN, the latent space `z` is not entirely opaque — it can be explored and, to a useful degree, controlled after the fact.

**Latent space entanglement:** ideally, each dimension of `z` would independently control exactly one interpretable feature (pose, gender, hair, age). In practice, standard GAN latent spaces are usually **entangled** — individual dimensions tend to affect *multiple* visual attributes simultaneously (changing one dimension might shift both smile and hair color at once), which makes precise, independent control of specific features difficult using raw coordinate axes alone. Despite this, specific *directions* in latent space (not necessarily aligned with any single coordinate axis) can still be found that reliably control one particular attribute, even if the axes themselves are entangled.

**Finding a `z` for a desired feature:** given a trained (and now frozen) generator, and a separately trained feature classifier (e.g., a "glasses" or "smile" detector), you can find a noise vector that produces a desired feature *without retraining the generator at all* — by treating `z` itself as the trainable parameter. Starting from a random `z`, generate `x = G(z)`, classify it to get `p = Classifier(x)`, compute a loss `L = -log(p)` (pushing `p` toward 1), and backpropagate that loss *through the frozen classifier and the frozen generator* to update only `z` via gradient descent — repeating until the resulting image reliably exhibits the desired feature: `z* = argmin_z L_BCE(Classifier(G(z)), target=1)`. This is a striking example of the same backpropagation machinery from Week 1 being reused for an entirely different purpose — optimizing an *input* rather than a network's weights.

**Latent space arithmetic:** perhaps the most famous GAN party trick is that meaningful semantic directions in latent space behave almost like linear vectors that can be added and subtracted. By interpolating between the latent codes of images with and without some attribute, you can extract a "feature direction" — for instance, `z_smile = z_smiling - z_neutral` — and then *add* that direction to any other face's latent code to add a smile to it: `z_smiling_woman = z_woman + z_smile`. The classic reported example is a "gender vector," `z_man - z_woman`, which can be added to essentially any face's latent code to shift its apparent gender — a result that works surprisingly well in practice and hints at real underlying structure in the learned latent space, despite the entanglement discussed above.

## 8. Applications of conditional GANs

Conditional GANs power a wide range of practical image-generation tasks: **image-to-image translation** (sketch→photo, day→night, black-and-white→color, satellite imagery→map), **text-to-image generation** (conditioning on a text description like "a red bird on a branch"), **attribute editing** (adding or removing glasses, changing hair color, aging a face up or down, changing expression), and **data augmentation** (generating additional synthetic examples of rare classes to help balance an imbalanced training dataset).

## Key takeaways

GANs reframe generative modeling as a competitive game between a generator that creates fakes and a discriminator that tries to catch them, trained via the minimax objective `min_G max_D E[log D(x)] + E[log(1-D(G(z)))]`, with both networks alternately updated until (ideally) the generator produces samples the discriminator can no longer distinguish from real data. DCGAN made this practical for images by replacing fully connected layers with strided/transposed convolutions and batch normalization. Standard GAN training is notoriously unstable, prone to vanishing gradients (when the discriminator gets too strong) and mode collapse (when the generator settles for fooling the discriminator with limited diversity); WGAN fixes both by replacing the log loss with the Wasserstein (Earth Mover's) distance, computed via an unbounded critic subject to a 1-Lipschitz constraint, giving smoother gradients and a loss that actually tracks sample quality. Conditional GANs regain control over generation by feeding a condition `y` into both networks, and, whether conditional or not, a trained GAN's latent space can be probed and manipulated after training — via gradient-based search or surprisingly linear vector arithmetic — to control specific attributes of the generated output. Compared to last week's VAEs, GANs generally produce sharper, more realistic samples, at the cost of noticeably less stable training — a trade-off you should be able to articulate clearly for the midterm and beyond.
