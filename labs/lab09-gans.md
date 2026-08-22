# Lab 9 — GANs

**Matches:** [Week 10 — Generative Adversarial Networks](../lectures/week10-gans.md)
**Goal:** Train a small DCGAN on MNIST, monitor its training dynamics, and directly compare the results with your VAE from Lab 8.

## Setup

```bash
pip install torch torchvision matplotlib
```

## Step 1 — Generator and Discriminator (DCGAN-style)

```python
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, noise_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(noise_dim, 128, 7, stride=1, padding=0), nn.BatchNorm2d(128), nn.ReLU(),   # 1->7
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),            # 7->14
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1), nn.Tanh(),                                  # 14->28
        )

    def forward(self, z):
        return self.net(z.view(z.size(0), -1, 1, 1))

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 4, stride=2, padding=1), nn.LeakyReLU(0.2),      # 28->14
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),  # 14->7
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
        )

    def forward(self, x):
        return self.net(x)  # raw logit; combine with BCEWithLogitsLoss
```

Note: MNIST images should be scaled to `[-1, 1]` (to match the generator's `Tanh` output) rather than `[0, 1]`.

## Step 2 — The training loop (alternating updates)

```python
import torch.optim as optim

g = Generator(); d = Discriminator()
opt_g = optim.Adam(g.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_d = optim.Adam(d.parameters(), lr=2e-4, betas=(0.5, 0.999))
criterion = nn.BCEWithLogitsLoss()

def train_step(real_images):
    batch_size = real_images.size(0)
    real_labels = torch.ones(batch_size, 1)
    fake_labels = torch.zeros(batch_size, 1)

    # --- Update Discriminator ---
    opt_d.zero_grad()
    d_loss_real = criterion(d(real_images), real_labels)
    z = torch.randn(batch_size, 100)
    fake_images = g(z).detach()
    d_loss_fake = criterion(d(fake_images), fake_labels)
    d_loss = d_loss_real + d_loss_fake
    d_loss.backward(); opt_d.step()

    # --- Update Generator ---
    opt_g.zero_grad()
    z = torch.randn(batch_size, 100)
    fake_images = g(z)
    g_loss = criterion(d(fake_images), real_labels)  # generator wants D to say "real"
    g_loss.backward(); opt_g.step()

    return d_loss.item(), g_loss.item()
```

## Step 3 — Train and log losses

Train for several epochs on MNIST, recording `d_loss` and `g_loss` at every step. Plot both curves. Note any oscillation or one network's loss collapsing toward zero (a sign the other network is "winning" too easily) — this is exactly the training instability discussed in the Week 10 lecture notes.

## Step 4 — Generate a sample grid, and compare against your VAE

Every few epochs, generate a fixed 5×5 grid of samples from a *fixed* noise batch (so you can watch the same set of samples improve over training) and save the images. At the end of training, place your GAN's final sample grid next to the VAE sample grid from [Lab 8, Step 4](lab08-autoencoders-and-vaes.md) and compare sharpness/detail.

## Step 5 (bonus) — Try the vanishing-gradient failure mode on purpose

Train the discriminator for many more steps than the generator per iteration (e.g., 5 discriminator updates per 1 generator update) for a short run, and observe what happens to the generator's loss and its sample quality. Relate this to the "vanishing gradients when D becomes too strong" discussion in the lecture notes.

## Checkpoint questions

1. In Step 3, did your losses oscillate, or did one of the networks' losses collapse toward zero at any point? What does each failure mode suggest is happening in the adversarial game?
2. In Step 4, describe the qualitative difference you observe between your GAN's and VAE's generated digits. Does it match the "GANs produce sharper but potentially less diverse samples; VAEs produce blurrier but more consistently plausible samples" framing from the lecture?
3. In Step 5, did you observe mode collapse (the generator producing very similar-looking digits regardless of the input noise)? If so, what specifically did you see in the sample grid that indicated it?
