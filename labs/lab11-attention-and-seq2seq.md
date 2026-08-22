# Lab 11 — Attention and Seq2Seq

**Matches:** [Week 12 — From Word Embeddings to Attention](../lectures/week12-rnn-attention-nlp.md)
**Goal:** Build an RNN encoder-decoder for a toy sequence-to-sequence task, add Bahdanau-style attention, and visualize the resulting attention matrix.

## Setup

```bash
pip install torch matplotlib
```

## Step 1 — A toy "translation" task

Rather than a full parallel corpus, use a synthetic task that's easy to verify: **sequence reversal**. Input: a random sequence of integers (e.g., `[3, 7, 1, 9]`). Output: the same sequence reversed (`[9, 1, 7, 3]`), plus an end-of-sequence token. This has the same "different positions matter, output built token by token" structure as translation, but you can check correctness by eye.

```python
import random

def make_example(min_len=3, max_len=8, vocab_size=10, eos=10):
    length = random.randint(min_len, max_len)
    seq = [random.randrange(vocab_size) for _ in range(length)]
    target = list(reversed(seq)) + [eos]
    return seq, target
```

## Step 2 — Plain encoder-decoder (no attention)

Build a GRU-based encoder that reads the input sequence and produces a final hidden state (the context vector `c`), and a GRU-based decoder that generates the output autoregressively, conditioned only on `c` and the previously generated token — exactly the Week 12 architecture, before attention is added.

```python
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        embedded = self.embed(x)
        outputs, hidden = self.gru(embedded)
        return outputs, hidden  # outputs: (B, T, H) needed later for attention

class DecoderNoAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, y_prev, hidden):
        embedded = self.embed(y_prev).unsqueeze(1)
        output, hidden = self.gru(embedded, hidden)
        logits = self.out(output.squeeze(1))
        return logits, hidden
```

Train with teacher forcing on batches of generated reversal examples, padding to a common length within each batch. Evaluate exact-match accuracy on held-out sequences of length 8–12 (longer than most training examples, to stress-test long-range copying).

## Step 3 — Add Bahdanau-style attention

Implement the alignment score, softmax, and weighted-sum context vector from the lecture notes:

```python
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: (B, 1, H); encoder_outputs: (B, T, H)
        scores = torch.bmm(self.W(decoder_hidden), encoder_outputs.transpose(1, 2))  # (B, 1, T)
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights, encoder_outputs)  # (B, 1, H)
        return context, weights.squeeze(1)
```

Wire this into a new `DecoderWithAttention` that, at each step, computes attention over the encoder's outputs using the current decoder hidden state, concatenates the resulting context vector with the decoder's own hidden state before the output projection, and returns both the logits and the attention weights (so you can visualize them later).

## Step 4 — Compare accuracy with and without attention, especially on long sequences

Train both decoders (with and without attention) on the same data, and evaluate exact-match accuracy at several test sequence lengths (e.g., 4, 8, 12, 16 — including lengths longer than anything seen in training, to test generalization). Plot accuracy vs. length for both.

## Step 5 — Visualize the attention matrix

For a handful of test examples, collect the attention weights at every decoder step (shape `T_decoder × T_encoder`) and display them as a heatmap (`plt.imshow`). For the reversal task, a well-trained attention mechanism should show a clear **anti-diagonal** pattern (decoder step `t` attends most strongly to encoder position `T_encoder - t`), directly mirroring the near-diagonal alignment pattern shown for the "Je suis étudiant" example in the lecture notes.

## Checkpoint questions

1. In Step 4, at what sequence length (if any) does the no-attention decoder's accuracy start to drop noticeably below the attention decoder's? Relate this to the "fixed context vector becomes a bottleneck for long sequences" argument in Week 12.
2. In Step 5, does the attention heatmap show the expected anti-diagonal pattern? If it's noisy or doesn't match, what might explain it (insufficient training, too small a hidden dimension, a bug in the attention computation)?
3. What would you expect the attention matrix to look like for a *different* toy task, such as "output every second token of the input"? Try constructing that task and check whether your attention visualization matches your prediction.
