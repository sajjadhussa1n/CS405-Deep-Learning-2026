# Quiz 4 — Generative Models

*Covers: [Week 9](../lectures/week09-autoencoders-vae.md) and [Week 10](../lectures/week10-gans.md).*

## Section A — Short answer

**A1.** What is the difference between a plain (vanilla) autoencoder and a Variational Autoencoder (VAE), in terms of what the encoder outputs and why that difference matters for generation?

**A2.** A plain autoencoder trained purely to minimize reconstruction error is generally a poor generative model on its own. Why — what goes wrong if you sample a random point in its latent space and decode it?

**A3.** Describe the VAE loss function at a high level: what are its two terms, and what does each one encourage?

**A4.** Explain the two-player game a GAN sets up between the Generator and the Discriminator, including what each network is trying to optimize.

**A5.** What specific training instability is Wasserstein GAN (WGAN) designed to address, and what does it change about the loss function compared to a standard GAN?

**A6.** What extra information does a Conditional GAN (cGAN) provide to both the Generator and Discriminator that a vanilla GAN does not, and what capability does this add?

## Section B — Multiple choice

**B1.** The "reparameterization trick" in VAEs exists to:
(a) Speed up the decoder's forward pass
(b) Allow gradients to flow through the random sampling step during backpropagation
(c) Reduce the number of parameters in the encoder
(d) Replace the KL divergence term entirely

**B2.** In the standard (non-saturating) GAN objective, the Discriminator is trained to:
(a) Minimize its ability to distinguish real from fake
(b) Maximize its accuracy at distinguishing real data from Generator output
(c) Generate new samples directly
(d) Compress images into a latent code

**B3.** DCGAN's main contribution was:
(a) A new loss function
(b) A set of architectural guidelines (strided/transposed convolutions, batchnorm, specific activations) for stable convolutional GAN training
(c) Removing the Discriminator entirely
(d) Introducing the reparameterization trick

**B4.** Mode collapse in GAN training refers to:
(a) The Discriminator's loss going to zero
(b) The Generator producing only a limited variety of outputs, ignoring much of the diversity in the real data distribution
(c) The training process crashing due to a numerical error
(d) The latent space becoming discontinuous

**B5.** WGAN replaces the standard GAN's Jensen-Shannon-divergence-based objective with an approximation of:
(a) KL divergence (b) Earth Mover's (Wasserstein) distance (c) Cosine similarity (d) Cross-entropy loss

## Section C — Applied reasoning

**C1.** You train a VAE and notice that when you set the KL divergence term's weight to zero (i.e., drop it from the loss entirely), the model gets excellent reconstructions but generates garbage when you sample random latent vectors and decode them. Explain why, referencing what the KL term is normally doing.

**C2.** A GAN's discriminator loss quickly drops to near zero early in training and stays there, while the generator's loss keeps climbing. Diagnose what is likely happening and suggest one concrete fix discussed in Week 10.

**C3.** You want to build a model that generates a photo-realistic face conditioned on a text description (e.g., "a smiling person with glasses"). Which of the two architectures from this week (VAE vs. GAN family) is more naturally suited to producing sharp, high-fidelity images, and which concept from this week would you use to make the generation *conditional* on the text?

## Answer Key

**A1.** A plain autoencoder's encoder outputs a single deterministic latent vector `z` for a given input. A VAE's encoder instead outputs the *parameters of a probability distribution* over the latent space for that input — typically a mean vector `μ` and a variance/log-variance vector `σ²` — and `z` is then sampled from that distribution. This matters for generation because it forces the latent space to be smooth and continuous (nearby points decode to similar, plausible outputs), which lets you sample a random `z` (e.g., from a standard normal prior) and get a coherent decoded output — something a plain autoencoder's latent space, with no such structure enforced, generally cannot support.

**A2.** A plain autoencoder's latent space has no constraint forcing it to be smooth, densely packed, or centered around any particular distribution — it only has to be *useful for reconstruction* of the exact training points it saw. This means there can be large "holes" or discontinuous regions in the latent space that don't correspond to any real training example, and decoding a randomly sampled point (rather than one produced by encoding a real input) very likely lands in one of those regions, producing an incoherent, unrealistic output.

**A3.** The VAE loss has a reconstruction term (typically MSE or binary cross-entropy between the input and the decoded output) that encourages the decoder to accurately reconstruct the input from its latent code, and a KL divergence term that pulls the encoder's output distribution `q(z|x)` toward a fixed prior (typically a standard normal, `N(0,I)`), which encourages the latent space to be smooth, continuous, and well-organized (no gaps, roughly centered at the origin) so that sampling from the prior at generation time produces realistic outputs.

**A4.** The Generator takes random noise as input and tries to produce fake samples realistic enough to fool the Discriminator; the Discriminator is a binary classifier that takes either a real training example or a Generator-produced fake and tries to correctly label which is which. The Generator is trained to *maximize* the Discriminator's error rate (or equivalently minimize `log(1-D(G(z)))`, or in the non-saturating form, maximize `log(D(G(z)))`), while the Discriminator is trained to *minimize* its own error rate at telling real from fake — a minimax game where each network's improvement pushes the other to improve in response.

**A5.** WGAN addresses training instability and mode collapse caused by the standard GAN's Jensen-Shannon-divergence-based objective, which can provide vanishing or uninformative gradients to the Generator when the real and fake distributions have little overlap (common early in training). WGAN replaces the discriminator ("critic") with one that estimates the Earth Mover's/Wasserstein distance between the real and fake distributions instead of classifying real vs. fake, which provides a smoother, more meaningful gradient signal to the Generator throughout training, along with a weight-clipping (or later, gradient penalty in WGAN-GP) constraint to enforce the Lipschitz condition the Wasserstein estimate requires.

**A6.** A cGAN feeds an extra conditioning signal — such as a class label, text embedding, or another image — into *both* the Generator (concatenated with the noise vector, to steer what it generates) and the Discriminator (concatenated with the real/fake sample, so it judges not just "is this realistic" but "is this realistic *and consistent with the given condition*"). This adds the capability to control what the GAN generates (e.g., "generate a 7" rather than a random digit) rather than only sampling an arbitrary example from the learned distribution.

**B1.** (b) Allow gradients to flow through the random sampling step during backpropagation.

**B2.** (b) Maximize its accuracy at distinguishing real data from Generator output.

**B3.** (b) A set of architectural guidelines (strided/transposed convolutions, batchnorm, specific activations) for stable convolutional GAN training.

**B4.** (b) The Generator producing only a limited variety of outputs, ignoring much of the diversity in the real data distribution.

**B5.** (b) Earth Mover's (Wasserstein) distance.

**C1.** Without the KL term, nothing constrains the encoder's output distributions to align with a known prior or to be smooth/continuous across the latent space — the encoder is free to place each training example's latent code wherever is most convenient for reconstruction, potentially in tight, disconnected clusters with large unused gaps between them. Reconstructions still look good because the decoder only ever needs to work well at the exact latent codes the encoder actually produces for real inputs. But a randomly sampled latent vector (e.g., from `N(0,I)`) has no guarantee of landing near any of those clusters, so the decoder — which never learned to produce sensible outputs for that region — generates garbage. The KL term is precisely what normally prevents this, by pulling every input's latent distribution toward the shared prior so the whole space stays densely and smoothly covered.

**C2.** This is a classic sign the Discriminator has become too strong too fast relative to the Generator — once it can perfectly separate real from fake, the gradient it provides to the Generator (via `log(1-D(G(z)))` or the non-saturating alternative) vanishes or becomes uninformative, stalling Generator learning even as its loss climbs. Concrete fixes discussed in Week 10 include: using the non-saturating Generator loss formulation (maximize `log(D(G(z)))` instead of minimizing `log(1-D(G(z)))`); balancing the training schedule (e.g., updating the Generator more often, or the Discriminator less often, per step); label smoothing on the Discriminator's real labels; or switching to a WGAN-style critic, which is specifically designed to keep providing a useful gradient even when it easily distinguishes real from fake.

**C3.** The GAN family (not the VAE) is more naturally suited to sharp, high-fidelity image generation — VAEs tend to produce comparatively blurrier outputs, a well-known consequence of the pixel-wise reconstruction loss term averaging over plausible outputs. To make generation conditional on the text description, you'd apply the Conditional GAN (cGAN) idea: feed a text embedding of the description as the conditioning signal into both the Generator (alongside the noise vector) and the Discriminator (alongside the image), so the Generator learns to produce images consistent with the given text and the Discriminator learns to reject images that don't match their paired description, not just images that look unrealistic.
