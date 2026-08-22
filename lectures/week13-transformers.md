# Week 13 — Transformer: Attention Is All You Need

*Companion notes for [`slides/lecture_week13.pdf`](../slides/lecture_week13.pdf)*

## Why this week matters

Week 12 introduced attention as a patch on top of an RNN encoder-decoder — a way to let the decoder look back at any encoder position instead of relying on one fixed context vector. This week takes the radical next step, following Vaswani et al.'s landmark 2017 paper "Attention Is All You Need": remove the RNN entirely, and build the whole model out of attention. The resulting architecture, the **Transformer**, is the backbone of essentially every large-scale NLP system covered for the rest of the course, including the LLMs in Week 14.

## 1. The limitation that motivates self-attention

In an RNN-based encoder-decoder with attention (Week 12), attention can only be computed *after* the encoder has finished processing the entire input sequentially, one token at a time — the encoder's hidden states `h_1, ..., h_T` have to be produced one after another before any attention weights can be computed over them. This creates three problems: attention computation cannot be parallelized across the sequence (it's gated by the sequential RNN), long-range dependencies are still, to some extent, mediated through the RNN's own recurrent state, and every decoder step re-derives its attention weights from scratch over the same fixed encoder outputs. **Self-attention** removes the RNN dependency altogether: every word attends to every other word in the sequence *simultaneously*, with no sequential bottleneck at all.

## 2. Self-attention: the core intuition

The key idea: each word looks at *all* other words in the sentence to build a richer understanding of itself, in context. For the French sentence "Le garçon jouait dans le parc hier soir" ("The boy was playing in the park yesterday evening"), when processing the word "jouait" (was playing), the model implicitly wants to know: who is doing it? ("garçon"), where? ("parc"), when? ("hier soir"). Every word becomes a **query** searching for relevant information among all the other words, which act as **keys** offering to be matched, and **values** carrying the actual content to be retrieved.

### The Query/Key/Value analogy

The slides use a library-search analogy: you (the **query**) are looking for something specific ("I need books about History"); each book has a label describing its subject (the **key**, e.g., "History book," "Fiction book"), which is compared against your query to compute a match score; and each book's actual content is the **value** — the information you actually retrieve, weighted by how well its key matched your query. In self-attention, *every single word* plays all three roles at once: it issues its own query ("what am I interested in?"), offers its own key ("what do I represent?"), and carries its own value ("my actual content"), all computed via separate learned linear projections of the same input embedding.

## 3. Computing self-attention for one word, step by step

Walking through the computation for "parc" (park) in the example sentence: first, a **query** is formed for "parc" via a learned matrix, `q_6 = W^Q · embedding("parc")` — this query effectively asks "what words are related to 'park'?" Second, that query is compared against the **key** of every word in the sentence via a dot product: `score_{6,j} = q_6 · k_j` for every position `j`. Intuitively, these raw scores should come out highest for the most relevant words — in this example, "jouait" (playing) scores highest, since the action happening *in* the park is the most relevant piece of context for understanding "park" here. Third, the raw scores are converted into proper, normalized attention weights with a **softmax**: `α_{6,j} = exp(score_{6,j}) / Σ_m exp(score_{6,m})` — for example, "jouait" might end up with `α ≈ 0.45` (the largest weight), "garçon" with `α ≈ 0.12`, "parc" itself with `α ≈ 0.18`, and so on, all summing to 1 across the sentence. Finally, the new, context-enriched representation of "parc" is a **weighted sum of every word's value vector**, using these attention weights: `c_parc = Σ_j α_j · v_j ≈ 0.02v_1 + 0.12v_2 + 0.45v_3 + ...`. The resulting vector for "parc" now genuinely contains information pulled in from "jouait" (the action happening there), "garçon" (who is doing it), and "hier soir" (when it happened) — all blended into a single, richer representation, computed with no recurrence at all.

### The Q, K, V projection matrices

Every word's query, key, and value vectors are obtained from its input embedding `x_i` via three separate, *learned* weight matrices: `q_i = W^Q x_i`, `k_i = W^K x_i`, `v_i = W^V x_i`. These three matrices are trained (via ordinary backpropagation, just like every other weight in the network) rather than fixed — using separate matrices for each role lets the same input embedding be transformed differently depending on whether it's being used to *search* for relevant context (query), to *advertise* what it represents to others (key), or to *carry* the actual content that gets aggregated (value).

## 4. The complete self-attention formula

Putting the whole computation together, for the entire sequence at once, gives the single equation that defines scaled dot-product attention:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

Reading this left to right: `QK^T` computes every query's dot product with every key simultaneously (a `T×d_k` matrix times a `d_k×T` matrix gives a `T×T` matrix of raw scores — every word compared against every other word, all at once); dividing by `√d_k` **scales** these scores (explained below); `softmax` (applied row-wise) converts each row of scores into a proper probability distribution over the sequence; and multiplying by `V` computes the weighted sum of value vectors for every position simultaneously.

### Why divide by `√d_k`?

For larger key/query dimensions `d_k` (e.g., 512), the dot products `q_i · k_j` tend to have larger magnitude simply because they're summing over more terms — with `d_k=512`, individual dot products can land in a range like `[-50, 50]`. Feeding such large values into a softmax makes it extremely "peaked" — one value close to 1 and everything else close to 0 — because `exp(50)` is astronomically larger than `exp(0)`. An almost one-hot softmax like this produces very small gradients almost everywhere (since the softmax is essentially saturated), reintroducing a vanishing-gradient-like problem. Dividing every score by `√d_k` before the softmax keeps the *variance* of the scores roughly constant regardless of dimensionality, avoiding this over-peaking and keeping gradients well-behaved.

### Why self-attention is so parallelizable

Because the entire computation above is expressed as matrix multiplications — `QK^T` (`[T×d_k] × [d_k×T] → [T×T]`), softmax+scale (still `[T×T]`), and `× V` (`[T×T] × [T×d_v] → [T×d_v]`) — every attention score for every pair of positions in the sequence can be computed **simultaneously** on a GPU, with no step waiting on the output of a previous step, unlike an RNN's inherently sequential recurrence. This is the single biggest practical advantage of self-attention over recurrence, and it's what makes training on today's massive text corpora computationally feasible at all.

Compared to RNNs, self-attention offers: no sequential bottleneck (full parallelism), *direct* connections between any two positions regardless of how far apart they are in the sequence (no need to propagate information step-by-step, which is exactly what caused vanishing gradients in Weeks 6–7), and naturally interpretable attention weights that can be visualized to see what the model is "looking at."

## 5. Multi-head attention: attending to multiple relationships at once

A single attention computation has a real limitation: it must blend *every* type of relevant relationship into one weighted average. For "parc" in our example sentence, a single attention head has to simultaneously answer "who is doing the action?" (garçon), "what action?" (jouait), "what's its grammatical role?" (dans), and "when?" (hier soir) — and it can only produce *one* set of attention weights, forcing all of that information to be blended together in a single pass, potentially losing the ability to cleanly separate these different kinds of relationships.

**Multi-head attention** runs several attention computations ("heads") in parallel, each with its *own* separate, independently learned `W^Q, W^K, W^V` projection matrices, letting different heads specialize in capturing different kinds of relationships — one might learn to focus on subject-verb relationships, another on grammatical/prepositional structure, another on temporal information, and so on:

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
```

Each head's projection matrices map into a smaller dimension `d_k = d_v = d_model / h`, so that after all `h` heads are concatenated back together, the total dimensionality returns to `d_model`; a final learned output projection `W^O` then mixes information across all the heads. Illustrating with tiny numbers: if `parc`'s Head 1 (subject-focused) output is `[0.2, 0.5]`, Head 2 (action-focused) is `[0.7, 0.1]`, and Head 3 (time-focused) is `[0.3, 0.4]`, concatenating gives `[0.2, 0.5, 0.7, 0.1, 0.3, 0.4]` (length `h×d_v`), and projecting this through `W^O` produces the final, combined representation — one vector that has genuinely incorporated subject, action, and time information as *separate* signals before being merged, rather than forcing them into one blended average from the start.

Real models use many heads: BERT-base uses `h=12` heads with `d_model=768` (so `d_k=d_v=64` per head), BERT-large uses `h=16` heads with `d_model=1024`, and GPT-3 uses `h=96` heads with `d_model=12288` — the general pattern being that larger models use both more heads and a larger model dimension.

## 6. The Transformer architecture

The full Transformer is built by stacking many layers of self-attention and simple feedforward processing, organized into an **encoder** (which processes the input/source sentence) and a **decoder** (which generates the output/target sentence), connected together — closely following the encoder-decoder pattern from Week 12, but with self-attention replacing recurrence throughout.

A useful analogy from the slides: just as a CNN (Weeks 3–4) builds up a hierarchy from pixels to edges to shapes to objects, the Transformer encoder builds up increasingly rich contextual understanding from words to local context to global, whole-sentence understanding — with self-attention providing the "sees everything at once" capability that a CNN's local receptive fields don't have, and positional encoding (next) providing the sense of order that convolution's spatial structure provides for free.

### Positional encoding

Self-attention, as defined above, has **no inherent notion of word order** — it treats the input as an unordered set of (query, key, value) triples, so "cat sat on mat" and "mat sat on cat" would, absent any fix, produce identical self-attention computations. The fix: add a **positional encoding** vector to each word's embedding before it enters the network, `final input = word embedding + positional encoding`, giving the model a way to distinguish position 1 from position 2 from position 3, and so on. The original Transformer uses a fixed **sinusoidal** encoding:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Using sine and cosine functions of different frequencies across the embedding dimensions ensures every position gets a unique encoding pattern, while also giving the encoding some useful mathematical structure (e.g., the encoding for a fixed offset can be expressed as a linear function of the encoding at any position, which theoretically helps the model learn to attend based on relative position).

### The encoder: bidirectional, unmasked self-attention

Inside the encoder, self-attention is **unmasked** — every word is free to attend to *every* other word in the input sentence, both to its left and to its right, giving genuinely bidirectional context. Processing "jouait" (was playing) in the encoder, the model looks left at "Le," "garçon" (to know *who* is playing) and right at "dans," "le," "parc," "hier," "soir" (to know *where* and *when*) — nothing is hidden from the encoder, since it has access to the complete input sentence from the start. After passing through several stacked encoder layers, each word's final representation has absorbed rich contextual information from the entire sentence — for example, the final representation of "parc" encodes that it is the location of "jouait," modified by the preposition "dans." These final encoder representations are then handed to the decoder, where they serve as the **keys and values** for a mechanism called cross-attention (below).

### The decoder: autoregressive generation

The decoder's job is to generate the output sentence (e.g., the English translation) one word at a time, **autoregressively**: it starts with only a start-of-sequence token `<SOS>`, predicts the first output word, then uses `<SOS>` plus that predicted word to predict the second word, and so on, continuing until an end-of-sequence token `<EOS>` is generated. Each decoder step involves three sub-components:

1. **Masked self-attention:** the decoder attends over the output tokens generated *so far*, but with a **causal mask** that prevents any position from attending to future positions that haven't been generated yet — a word can only look at itself and everything before it, never ahead. (Contrast this directly with the encoder's fully unmasked, bidirectional self-attention.)
2. **Cross-attention:** the decoder's current query attends over the **encoder's** output keys and values (not the decoder's own) — this is structurally the exact same query/key/value mechanism as self-attention, just with the queries coming from the decoder and the keys/values coming from the encoder, playing precisely the role that attention played in the RNN-based encoder-decoder of Week 12: letting the decoder "look back" at the relevant parts of the source sentence for whatever it's currently generating.
3. **Feed-forward processing and softmax** (detailed in Section 7) to produce a probability distribution over the output vocabulary for the next token.

Walking through the example translation "Je suis étudiant"-style sentence step by step: at step 1, with input `[<SOS>]`, masked self-attention has nothing to mask yet, cross-attention focuses strongly on "Le" in the encoder output, and the model predicts "The." At step 2, with input `[<SOS>, "The"]`, masked self-attention lets "The" attend back to `<SOS>` (but not the reverse), cross-attention now focuses on "garçon," and the model predicts "boy." At step 3, with input `[<SOS>, "The", "boy"]`, cross-attention shifts focus to "jouait," predicting "was playing." This continues — predicting "in," "the," "park," "yesterday," "evening" — until, at step 9, with the full generated sentence as input, cross-attention has effectively "used up" the whole source sentence, and the model predicts `<EOS>`, ending generation with the complete translation: "The boy was playing in the park yesterday evening."

## 7. The remaining architectural pieces

Two further ingredients, borrowed and adapted from earlier weeks, appear around every attention and feed-forward block in the Transformer.

### Add & Norm: residual connections and layer normalization

**"Add"** is a **residual (skip) connection** — the exact same idea introduced for CNNs in Week 4's ResNet — where a sub-layer's input is added directly to its output: `Add(x) = x + Sublayer(x)`. This prevents vanishing gradients (gradients can flow straight through the skip connection, bypassing the sub-layer entirely if needed), makes training easier (the model can effectively learn an identity mapping if a particular sub-layer isn't useful for a given input), and is precisely what makes it feasible to stack **100+ Transformer layers** in the largest modern models.

**"Norm"** is **Layer Normalization**, `LayerNorm(x) = γ·(x-μ)/σ + β`, where `μ` and `σ` are computed across the *feature* dimension for each individual example (in contrast to BatchNorm from Week 2, which normalizes across the *batch* dimension) — this distinction matters for sequence models, since sequence lengths vary and "batch statistics" are less natural to define per-token than per-example feature statistics. Layer normalization stabilizes training and speeds convergence, and in the Transformer it's applied *after* the residual addition: `LayerNorm(x + Sublayer(x))`. Every attention block and every feed-forward block in the Transformer is wrapped in this same Add & Norm pattern.

### The position-wise feed-forward network (FFN)

After each attention sub-layer, every position independently passes through an identical small feedforward network: `FFN(x) = max(0, xW_1 + b_1)W_2 + b_2` — two linear layers with a ReLU in between, expanding the dimension (e.g., BERT-base expands from `d_model=768` up to `d_ff=3072`, four times larger) and then contracting back down to `d_model`. Crucially, this network is **applied identically and independently to every position** in the sequence — the same weights are reused at every token, exactly like the parameter sharing that made convolution efficient in Week 3, just applied here across sequence positions rather than spatial positions.

Why is the FFN needed at all, given that attention already mixes information across positions? Because self-attention itself is fundamentally a **linear** operation — computing weighted sums of value vectors. Stacking several purely linear operations (attention layers with no non-linearity in between) would, exactly as in Week 1's argument about deep linear networks, collapse mathematically into something no more expressive than a single linear layer. The FFN's ReLU non-linearity is what gives the Transformer genuine representational depth, letting each layer do real, non-linear "processing" of the contextual information that attention just finished "mixing" together. This identical FFN structure appears in both the encoder and the decoder.

## 8. From decoder output to predicted words

The final decoder representation for each position is converted into an actual word prediction through two more layers: a **linear layer** that projects the `d_model`-dimensional decoder output up to the size of the output vocabulary, `logits = h_decoder · W_out + b_out`, followed by a **softmax** that converts those logits into a proper probability distribution over the vocabulary, `P(y_t | y_{<t}, x) = exp(logits_t) / Σ_v exp(logits_v)` — exactly the same linear-then-softmax pattern used for word prediction back in Week 11's embedding-training network.

**During training**, the full target sequence is available (teacher forcing, exactly as in Week 12), so softmax probabilities are produced for *every* position simultaneously, and the loss is the sum of cross-entropy terms across all positions: `L = -Σ_t log P(y_t^true | y_{<t}, x)`. **During inference**, only the most recently generated token's prediction actually matters (there's no "true" future to compute a loss against) — the model takes the highest-probability next token (`argmax`, or samples from the distribution using more sophisticated decoding strategies), appends it to the growing output sequence, and repeats, exactly matching the autoregressive generation process walked through in Section 6.

## Key takeaways

The Transformer replaces the sequential recurrence of RNNs entirely with **self-attention**, letting every word attend to every other word in a sequence simultaneously via a learned query/key/value mechanism, `Attention(Q,K,V) = softmax(QK^T/√d_k)V`, computed as pure matrix multiplication and therefore massively parallelizable on modern hardware. **Multi-head attention** runs several such attention computations in parallel with independently learned projections, letting different heads specialize in different kinds of relationships before being concatenated and recombined. The full architecture stacks these attention blocks with position-wise feedforward networks, residual connections, and layer normalization into an **encoder** (bidirectional, unmasked self-attention over the full input) and a **decoder** (causally masked self-attention over the output-so-far, plus cross-attention back into the encoder's representations), generating output autoregressively one token at a time. Because word order carries no inherent signal in pure attention, a **positional encoding** is added to the input embeddings to restore it. This architecture — introduced in 2017 specifically for machine translation — turned out to generalize far beyond translation, and is the direct foundation for the large language models covered next week.
