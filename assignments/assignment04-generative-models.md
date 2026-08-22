# Assignment 4 — Generative Models: Autoencoders, VAEs, and GANs

**Covers:** Week 9 (Autoencoders and VAEs), Week 10 (GANs)
**Deliverable:** PyTorch code + generated samples + written report

## Learning objectives

Implement a plain autoencoder and demonstrate its inability to generate new data by random sampling; implement a Variational Autoencoder and show how the KL-divergence term reshapes the latent space; implement a (DC)GAN and directly compare its generations against the VAE's on the same dataset.

## Part A — Autoencoder and its generation failure

1. Implement a convolutional autoencoder (Week 9) and train it on a simple image dataset (e.g., MNIST or Fashion-MNIST).
2. Visualize reconstructions for a handful of test images.
3. **Demonstrate the failure mode from Week 9:** sample random points from the latent space (matching the empirical mean/variance of your encoder's outputs) and decode them. Show that a meaningful fraction of these decoded images look like noise or nonsensical blends, unlike genuine digits/garments. Include a grid of at least 25 such samples.

## Part B — Variational Autoencoder

1. Convert your Part A autoencoder into a VAE: modify the encoder to output `μ` and `log(σ²)`, implement the reparameterization trick (Week 9), and add the KL-divergence term to your loss.
2. Train it on the same dataset and again visualize a grid of samples generated purely from `z ~ N(0,1)`, with no input image. Compare this grid against Part A's random-sampling grid and discuss the difference.
3. Visualize the 2D latent space (if you use a 2D latent dimension, plot it directly; otherwise use PCA/t-SNE) colored by class label, and discuss whether classes are separated and whether nearby points in latent space correspond to visually similar outputs.
4. Perform latent-space **interpolation**: pick two real images, encode both, linearly interpolate between their latent codes, and decode a sequence of intermediate points. Include the resulting image sequence and comment on whether the interpolation looks smooth.
5. Run a small ablation: train a second VAE with a much smaller (or larger) weight on the KL term (relative to the reconstruction term) and show how it affects (a) reconstruction quality and (b) the quality/coverage of randomly sampled generations. Relate your observation to the trade-off discussed in Week 9.

## Part C — GAN

1. Implement a DCGAN (Week 10: strided/transposed convolutions, batch norm, LeakyReLU in the discriminator, ReLU/tanh in the generator) and train it on the same dataset used in Parts A–B.
2. Track and plot the generator and discriminator losses over training. Note any instability (oscillation, one network "winning" too early) and describe what you observed.
3. Generate a grid of samples from your trained GAN and place it side by side with the VAE's sample grid from Part B. Discuss the visual difference in sharpness, and connect this to the theoretical VAE-vs-GAN comparison in Week 10 (pixel-wise reconstruction loss vs. adversarial loss).
4. **Optional (bonus, up to 10% extra credit on this assignment):** implement the Wasserstein loss with gradient penalty (WGAN-GP, Week 10) instead of the standard GAN loss, and compare training stability (loss curves) and sample quality against your standard DCGAN.

## Report requirements

Include all requested visualizations (reconstructions, random-sample grids for AE/VAE/GAN, latent-space plot, interpolation sequence, loss curves) alongside written discussion connecting your observations back to the theoretical material in Weeks 9–10 — in particular, be explicit about *why* the plain autoencoder fails to generate, and *why* the GAN's samples look different from the VAE's.

## Grading rubric

| Component | Weight |
|---|---|
| Part A: autoencoder implementation and generation-failure demonstration | 15% |
| Part B: VAE implementation, sampling, interpolation, and KL ablation | 40% |
| Part C: GAN implementation, training stability discussion, and comparison to VAE | 35% |
| Report clarity | 10% |
