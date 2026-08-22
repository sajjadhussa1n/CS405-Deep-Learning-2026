# Week 11 — Word Embeddings: From One-Hot to Dense Representation

*Companion notes for [`slides/lecture_week11.pdf`](../slides/lecture_week11.pdf)*

## Why this week matters

Weeks 6–7 built RNNs and LSTMs that can process sequences of vectors, but we never addressed a basic question: how do you turn a *word* into a vector in the first place? This week answers that question, moving from naive representations (indices, one-hot vectors) to learned, dense **word embeddings** that capture meaning geometrically — the representation that every NLP model in Weeks 12–14 (RNN taggers, attention, transformers, LLMs) is built on top of.

## 1. The core problem: computers only understand numbers

Text needs to be converted into numerical form before any neural network can process it, and we want that numerical representation to have three properties: similar words should get similar representations, the representation should capture actual meaning, and mathematical operations on the representation should reveal real semantic relationships. We need a mapping from **word → vector**.

### Naive attempt 1: word indices

Assign every word in the vocabulary a unique integer (king=0, queen=1, man=2, ...). This fails immediately: there's no notion of similarity (king=0 and queen=1 are just as "different," numerically, as king=0 and pizza=999), no meaning is captured, and worse, the model would implicitly treat index comparisons like "1 > 0" as meaningful, which is nonsensical for word identities.

### Naive attempt 2: one-hot vectors

Represent each word as a binary vector with a single 1 at that word's unique position and 0s everywhere else. For a vocabulary of size `|V|=5`: `king=[1,0,0,0,0]`, `queen=[0,1,0,0,0]`, and so on. This avoids the false-ordering problem, but introduces two new ones. First, **every pair of one-hot vectors is orthogonal** — `king · queen = 0`, exactly the same dot product as `king · pizza` — so the representation still encodes *zero* similarity information between any two words. Second, the **curse of dimensionality**: with real vocabularies of 50,000 to over 1,000,000 words, each vector has that many dimensions, is extremely sparse (a single 1 among tens of thousands of 0s), and carries almost no information per dimension, forcing models to need huge amounts of data just to compensate.

## 2. The distributional hypothesis

The theoretical foundation for a better representation goes back to linguist J.R. Firth's famous 1957 observation: **"you shall know a word by the company it keeps."** Words that tend to appear in similar contexts tend to have similar meanings — a word's meaning is, in a very real operational sense, defined by the words that typically surround it. "King" tends to appear near throne, crown, royal, queen, palace; "queen" tends to appear near almost exactly the same set of words — that pattern of shared context is itself evidence that "king" and "queen" are semantically related. This is the intuition that every word-embedding method in this lecture — and, ultimately, every language model in this course — is built to exploit.

## 3. Word embeddings: dense, low-dimensional, and meaningful

Instead of a sparse `|V|`-dimensional one-hot vector, a **word embedding** represents each word as a dense vector of only `d ≈ 50` to `300` real-valued dimensions. Conceptually, each dimension of a good embedding ends up encoding some latent semantic feature — even if no human explicitly designed it to. A toy example: a "gender" dimension might be strongly positive for "king" and "man" and strongly negative for "queen" and "woman," while a separate "royalty" dimension is strongly positive for both "king" and "queen" but negative for "man" and "woman." Crucially, **the model discovers these latent features automatically purely from raw text**, with no hand-labeling of "this dimension means gender" required.

Visualized in two dimensions (e.g., via t-SNE, a dimensionality-reduction technique used for visualization), embeddings trained on real text tend to show semantically related words clustering together — royalty words (king, queen, prince, princess) form one cluster, professions (doctor, nurse, teacher, engineer) form another, animals (cat, dog, horse, tiger) form a third — purely as an emergent property of training, not because any cluster label was ever provided.

### Vector arithmetic: the famous analogy trick

If embeddings genuinely capture meaning, semantic *relationships* should correspond to consistent *vector operations*. The canonical example: "king is to queen as man is to ___?" turns out to be answerable almost exactly with simple vector arithmetic: `v_king - v_man + v_woman ≈ v_queen`. Intuitively, `v_king - v_man` isolates something like "the direction that separates royal-male from royal-neutral" (roughly, removes "maleness"), and adding `v_woman` reintroduces "femaleness" — landing very close to the actual embedding for "queen." A concrete worked example with hypothetical 3D embeddings (Gender, Royalty, Status dimensions) confirms this numerically: computing `[0.8-0.7-0.8, 0.9+0.8-0.7, 0.7+0.6-0.6] = [-0.7, 1.0, 0.7]` lands almost exactly on queen's actual embedding of `[-0.7, 0.9, 0.7]`.

This isn't a one-off trick — it generalizes across many kinds of relationships: capital-country (`Paris - France + Germany ≈ Berlin`), pluralization (`apples - apple + car ≈ cars`), verb tense (`running - run + eat ≈ eating`), comparatives (`better - good + bad ≈ worse`), opposites (`hot - cold + good ≈ bad`), and professions (`actor - man + woman ≈ actress`). The underlying explanation is that the embedding space appears to be approximately **linear** along semantically meaningful directions — the "gender direction" `v_man - v_woman` turns out to be roughly parallel no matter which specific pair of gendered words you compute it from, so analogies reduce to simple vector addition: `v_king + (v_woman - v_man) ≈ v_queen`.

This linearity is not perfect, however. Analogy accuracy varies (some, like king:queen, work very reliably; others, like good:best, work less consistently), and failures tend to have identifiable causes: multi-step relationships (e.g., "grandfather" requires two separate gender-related steps), rare words with insufficient training data, genuinely non-linear semantic relationships, and — importantly — **dataset bias**: an infamous failure of early word embeddings was analogies like "doctor : man :: nurse : woman," which reflects and amplifies societal bias present in the training corpus rather than any inherent linguistic fact. Vector arithmetic reveals real structure in embeddings, but it also faithfully reveals the biases baked into whatever text the embeddings were trained on.

| Property | One-hot | Embedding |
|---|---|---|
| Dimension size | `\|V\|` (50k–1M) | `d` (50–300) |
| Sparsity | very sparse | dense |
| Similarity | none (orthogonal) | cosine similarity |
| Captures meaning | no | yes |
| Trainable | no | yes |
| Interpretability | high | low (latent features) |

## 4. Learning embeddings: predict a word from its context

The core training trick behind nearly all classic word-embedding methods: train a neural network to **predict a word from its surrounding context** (or vice versa), and then, rather than actually using the network's predictions afterward, throw away the prediction task and keep the network's *hidden-layer weights* — those weights are the embeddings. This works because a network that gets good at predicting shared contexts must, along the way, learn similar internal representations for words that tend to appear in similar contexts.

The architecture is a shallow feedforward network: a `|V|×1` one-hot input, a hidden layer of size `N` (the embedding dimension, 50–300), and a `|V|×1` softmax output layer giving a probability distribution over the vocabulary. There are two weight matrices — `W1` (`N×|V|`, input→hidden) and `W2` (`|V|×N`, hidden→output) — and **after training, each column of `W1` is the learned embedding vector for the corresponding word.**

Three ways to build the (context, target) training pairs are common: **next-word prediction** (autoregressive — predict the following word from the preceding `k` words, summing or averaging their one-hot vectors as input), **CBOW (Continuous Bag-of-Words)** (predict the *center* word of a window from the surrounding context words on both sides, again summed/averaged), and **Skip-gram** (the reverse of CBOW — predict each individual surrounding context word from a single target/center word; each (target, context-word) pair becomes its own separate training example).

### Forward pass, loss, and training

The forward pass is: `h = ReLU(W1·x + b1)` (project the one-hot input into embedding space), `z = W2·h + b2` (project back out to vocabulary-sized logits), and `ŷ = softmax(z)` (convert logits to a probability distribution over the vocabulary). Since the true target `y` is one-hot with a 1 only at the correct word's index `c`, the cross-entropy loss collapses to a simple expression: `L = -Σ y_i log(ŷ_i) = -log(ŷ_c)` — literally just the negative log-probability the model assigned to the correct word. A confident, correct prediction (`ŷ_c` close to 1) gives a loss close to 0; a poor prediction (`ŷ_c` close to 0) gives a large loss (e.g., `-log(0.01) ≈ 4.6`). Training proceeds exactly as in Week 1: forward pass, compute loss, backpropagate (via the chain rule through `W2` then `W1`), update weights with gradient descent, repeat over many epochs — and, at the end, extract the embeddings as the columns of the trained `W1`.

## 5. The softmax bottleneck, and the fix: negative sampling

There's a serious practical problem with the setup above: computing the softmax's denominator requires summing over **every single word in the vocabulary** at every training step. With `|V| = 1,000,000`, that's a million exponentials computed per training example — far too slow to be practical at real-world scale.

**Negative sampling** reframes the learning problem to sidestep this. Instead of the hard question "which one word, out of a million, is the correct context?", it asks the much cheaper question "is *this specific* word a valid context for the target, yes or no?" — repeated for the one real (positive) context word plus a handful (`K`, typically 5–20) of randomly chosen (negative) words that are *not* the actual context. The restaurant analogy from the slides: instead of a waiter computing your preference score for all 100 dishes on the menu (full softmax), a smart waiter just asks "do you want *this* specific dish?" for the one you actually ordered plus a handful of random alternatives — comparing only `K+1` options instead of the entire menu.

Architecturally, this collapses the `|V|`-way softmax into `K+1` independent binary classifiers: one positive branch (the real context word) and `K` negative branches (randomly sampled words), each scored by a dot product `h · W2[word_idx]` and passed through a sigmoid to produce a probability. This is an `O((K+1)·D)` operation per step instead of `O(|V|·D)` — a massive speedup, since only `K+1` rows of the output embedding matrix `W2` are touched (and updated) per training step, even though all `|V|` rows are still stored in memory for later use.

The loss for one training example, given the positive probability `p_pos` and the `K` negative probabilities `p_neg_j`, is a straightforward binary cross-entropy sum:

```
L = -log(p_pos) - Σ_{j=1}^{K} log(1 - p_neg_j)
```

Negative words are **not** sampled uniformly at random — they're sampled with probability proportional to their raw frequency raised to the **3/4 power**, `P(w) ∝ f(w)^(3/4)`. This deliberately dampens the dominance of extremely common words (like "the," whose raw frequency would otherwise make it show up as a "negative" example constantly, teaching the model little) while boosting the relative sampling probability of rarer words, producing more informative negative examples overall. The number of negatives `K` trades speed for quality: small datasets typically use `K=5–10`, larger datasets `K=10–30`; more negatives generally improve the model's ability to discriminate real contexts from noise, but with diminishing returns and slower training as `K` grows.

## 6. Word2Vec: the predictive family

**Word2Vec** is the umbrella name for exactly the predictive approach described above: slide a window over a corpus, form (target, context) pairs via CBOW or Skip-gram, and train the shallow prediction network (using negative sampling for efficiency) to obtain embeddings. Its strengths are fast training even on very large corpora, capturing local syntactic patterns well, and architectural simplicity. Its main limitation is that it only ever looks at a **local** window of `k` words around each target — it never directly uses global, corpus-wide statistics about how often word pairs co-occur across the entire dataset, potentially leaving useful information on the table.

## 7. GloVe: incorporating global co-occurrence statistics

**GloVe (Global Vectors for Word Representation)**, introduced by Pennington, Socher, and Manning (2014), takes a different, count-based route, aiming to combine the best of Word2Vec's predictive local-window approach with matrix-factorization methods' use of global corpus statistics.

The starting point is the **co-occurrence matrix** `X`, where `X_ij` counts how many times word `j` appears within a context window of word `i`, computed by scanning the entire corpus once. This matrix is `|V|×|V|` (potentially huge) and very sparse (most word pairs never co-occur), but it captures genuinely global statistics rather than just what happened to appear in a handful of sampled local windows.

GloVe's core modeling idea is that the dot product of two word vectors should relate to how often those words co-occur: `w_i · w̃_j ∝ (something related to X_ij)`. Directly setting `w_i · w̃_j = X_ij`, however, doesn't work well, because raw co-occurrence counts follow a power-law (Zipf's law) distribution — extremely common pairs like ("the," "of") might co-occur a million times, while meaningful but less frequent pairs like ("king," "queen") might co-occur only a few hundred times. Training directly on these raw counts would let the huge, mostly uninformative counts for common word pairs dominate the loss, drowning out the more meaningful differences between rarer pairs. The fix is to work with `log(X_ij)` instead of the raw count — the logarithm compresses the enormous dynamic range (a million becomes about 13.8, one becomes 0) and makes relationships between counts much closer to linear.

### The GloVe objective

```
J = Σ_{i,j} f(X_ij) · (w_i · w̃_j + b_i + b̃_j - log(X_ij))²
```

This is a **weighted least-squares regression**: it directly pushes `w_i · w̃_j + b_i + b̃_j` (a dot product plus two bias terms) to be close to `log(X_ij)`, weighted by a function `f(X_ij)`. This is structurally quite different from Word2Vec's binary cross-entropy classification loss.

The weighting function down-weights both extremes appropriately:

```
f(x) = (x / x_max)^α   if x < x_max
f(x) = 1                otherwise
```

with typical values `x_max = 100` and `α = 3/4`. This means very rare co-occurrences contribute little to the loss (since they're noisy and unreliable), while very common co-occurrences are *capped* rather than allowed to dominate — a pair that co-occurs 1,000 times gets the same maximum weight as one that co-occurs 100 times, preventing the most frequent pairs from swamping everything else.

GloVe actually learns **two** sets of vectors per word — a "target" embedding `w_i` and a "context" embedding `w̃_i` — because the model treats the two roles somewhat differently during training. In practice, both turn out to be meaningful, and the final embedding used downstream is usually their sum or average, `(w_i + w̃_i)/2`, which tends to reduce noise relative to using either one alone.

### Word2Vec vs. GloVe

| Aspect | Word2Vec | GloVe |
|---|---|---|
| Context used | local (sliding window) | global (full co-occurrence matrix) |
| Underlying method | neural network (predictive) | matrix factorization (count-based) |
| Loss function | binary cross-entropy | weighted mean squared error |
| Output activation | sigmoid | linear |
| Negative sampling needed | yes | no |
| What it captures best | local syntactic patterns | global corpus statistics |

## Key takeaways

Raw word indices and one-hot vectors both fail to capture meaning or similarity between words, and one-hot vectors additionally suffer from extreme sparsity and dimensionality. Word embeddings solve this by representing each word as a dense, low-dimensional vector learned from data, following the distributional hypothesis that words appearing in similar contexts should end up with similar representations — a property that turns out to make the embedding space approximately linear along many semantically meaningful directions, enabling the famous vector-arithmetic analogies (with the important caveat that these analogies also faithfully reflect any bias present in the training corpus). Word2Vec learns embeddings by training a shallow network to predict context from a target word (or vice versa) using only local windows, made computationally tractable at scale via negative sampling, which turns an expensive `|V|`-way softmax into a handful of cheap binary classifications. GloVe instead directly factorizes a global word-word co-occurrence matrix, using a log-transformed, appropriately weighted least-squares objective. Both produce broadly similar, high-quality embeddings in practice and remain widely used as pre-trained building blocks — the representation that every architecture from next week's RNN-based language tasks through the transformers and LLMs of Weeks 13–14 is ultimately built on top of.
