# Lab 8 — Autoencoders and VAEs

**Matches:** [Week 9 — Autoencoders and Variational Autoencoders](../lectures/week09-autoencoders-vae.md)
**Goal:** Build a plain autoencoder, watch it fail to generate, then fix it with the reparameterization trick and KL regularization.

## Setup

```bash
pip install torch torchvision matplotlib
```

## Step 1 — A convolutional autoencoder

```python
import torch
import torch.nn as nn

class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),   # 28->14
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),  # 14->7
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, latent_dim),
        )
        self.decoder_fc = nn.Linear(latent_dim, 32 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),  # 7->14
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid(),  # 14->28
        )

    def forward(self, x):
        z = self.encoder(x)
        h = self.decoder_fc(z).view(-1, 32, 7, 7)
        return self.decoder(h), z
```

Train it on MNIST with binary cross-entropy or MSE reconstruction loss. Plot reconstructions of a handful of test images next to the originals.

## Step 2 — Demonstrate the generation failure

Compute the empirical mean and standard deviation of your trained encoder's outputs across the training set. Sample 25 random latent vectors from `N(empirical_mean, empirical_std)`, decode them, and display the results in a 5×5 grid. Note how many of the 25 look like garbage compared to the reconstructions in Step 1.

## Step 3 — Convert to a VAE

```python
class VAE(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.enc_conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(32 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(32 * 7 * 7, latent_dim)
        self.decoder_fc = nn.Linear(latent_dim, 32 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.enc_conv(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std          # the reparameterization trick

    def decode(self, z):
        h = self.decoder_fc(z).view(-1, 32, 7, 7)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(recon, x, mu, logvar):
    recon_loss = nn.functional.binary_cross_entropy(recon, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl, recon_loss, kl
```

Train it on MNIST, logging the reconstruction and KL terms separately over training.

## Step 4 — Generate from pure noise

Sample `z ~ N(0, I)` directly (no encoder involved), decode, and display a 5×5 grid. Compare against Step 2's grid — the VAE's samples should look noticeably more like plausible digits, even where imperfect.

## Step 5 — Latent space visualization and interpolation

If `latent_dim=2`, plot the latent codes of a few hundred test images directly, colored by digit label (if `latent_dim>2`, project with PCA/t-SNE first). Then pick two test images of different digits, encode both to get `mu_A` and `mu_B`, linearly interpolate `z = (1-t)*mu_A + t*mu_B` for `t` in `[0, 0.25, 0.5, 0.75, 1]`, decode each, and display the resulting sequence.

## Step 6 — KL weight ablation

Retrain your VAE with the KL term scaled by a small factor (e.g., `0.1 * kl`) and again with a large factor (e.g., `10 * kl`) in the loss. For each, regenerate the Step 4 grid and note the trade-off between reconstruction sharpness and sample quality/coverage.

## Checkpoint questions

1. In Step 2 vs. Step 4, describe concretely what changed about the *look* of the failures, if any remain in the VAE's grid, compared to the plain autoencoder's.
2. In Step 5, does interpolating between two different digit classes pass through recognizable intermediate shapes, or does it produce a blurry, ambiguous blend partway through? What does this tell you about how "smooth" your learned latent space actually is?
3. In Step 6, what happened to reconstruction quality when you *increased* the KL weight? Does this match the reconstruction-vs-regularization trade-off described in the lecture notes?
