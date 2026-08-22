# Lab 10 — Word Embeddings

**Matches:** [Week 11 — Word Embeddings: From One-hot to Dense Representation](../lectures/week11-word-embeddings.md)
**Goal:** Train small skip-gram embeddings with negative sampling from scratch, and explore what they learn with nearest-neighbor and analogy queries.

## Setup

```bash
pip install torch numpy matplotlib scikit-learn
```

## Step 1 — Build a tiny corpus and vocabulary

Use any medium-sized plain-text file (a public-domain book from Project Gutenberg works well). Tokenize (simple whitespace + lowercase + punctuation stripping is fine), build a vocabulary of the top N most frequent words (e.g., N=5000), and map every word to an integer index.

## Step 2 — Generate skip-gram (target, context) pairs

```python
def make_skipgram_pairs(token_ids, window=2):
    pairs = []
    for i, target in enumerate(token_ids):
        for offset in range(-window, window + 1):
            j = i + offset
            if offset == 0 or j < 0 or j >= len(token_ids):
                continue
            pairs.append((target, token_ids[j]))
    return pairs
```

## Step 3 — Negative sampling and the embedding model

```python
import torch
import torch.nn as nn
import numpy as np

class SkipGramNS(nn.Module):
    def __init__(self, vocab_size, embed_dim=100):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embed_dim)
        self.out_embed = nn.Embedding(vocab_size, embed_dim)

    def forward(self, target, context, negatives):
        v = self.in_embed(target)            # (B, D)
        u_pos = self.out_embed(context)       # (B, D)
        u_neg = self.out_embed(negatives)     # (B, K, D)
        pos_score = torch.sigmoid((v * u_pos).sum(-1))
        neg_score = torch.sigmoid(-(u_neg * v.unsqueeze(1)).sum(-1))
        loss = -(torch.log(pos_score + 1e-10) + torch.log(neg_score + 1e-10).sum(-1))
        return loss.mean()
```

Build the negative-sampling distribution `P(w) ∝ freq(w)^0.75` (Week 11) and sample `K=5` negatives per training pair using `torch.multinomial` or `np.random.choice` with those weights.

## Step 4 — Train and extract embeddings

Train for several epochs over your (target, context, negatives) triples. Extract the final embedding matrix as `model.in_embed.weight.detach()`.

## Step 5 — Nearest neighbors

```python
import torch.nn.functional as F

def nearest_neighbors(word, embeddings, word2idx, idx2word, k=5):
    idx = word2idx[word]
    sims = F.cosine_similarity(embeddings[idx].unsqueeze(0), embeddings)
    top = sims.topk(k + 1).indices.tolist()
    return [idx2word[i] for i in top if i != idx][:k]
```

Query at least 5 words of your choice and report the nearest neighbors. Comment on whether they look semantically sensible.

## Step 6 — Analogies

```python
def analogy(a, b, c, embeddings, word2idx, idx2word, k=5):
    vec = embeddings[word2idx[b]] - embeddings[word2idx[a]] + embeddings[word2idx[c]]
    sims = F.cosine_similarity(vec.unsqueeze(0), embeddings)
    top = sims.topk(k + 5).indices.tolist()
    exclude = {word2idx[a], word2idx[b], word2idx[c]}
    return [idx2word[i] for i in top if i not in exclude][:k]
```

Try `analogy("man", "king", "woman")` (expecting something close to "queen") and at least 4 other analogies drawn from your own corpus's vocabulary. Report your hit rate and discuss any failures, connecting them to the "why some analogies fail" discussion in Week 11.

## Step 7 — 2D visualization

Pick 30–50 words spanning a few semantic categories present in your corpus (e.g., a handful of character names, places, common verbs). Run PCA or t-SNE (`sklearn.manifold.TSNE`) on their embeddings and scatter-plot them with word labels. Do you see any visible clustering by category?

## Checkpoint questions

1. How does your nearest-neighbor quality change if you shrink `embed_dim` from 100 down to, say, 10? What does this suggest about the trade-off between embedding dimensionality and representational capacity?
2. In Step 3, what would go wrong (in terms of training signal) if you sampled negatives *uniformly at random* over the vocabulary instead of using the `freq^0.75` distribution? Try it and compare nearest-neighbor quality.
3. Did any of your Step 6 analogies fail? Inspect the corpus for how often the relevant words actually appear — is data sparsity a likely explanation, per the lecture notes' discussion of rare-word failures?
