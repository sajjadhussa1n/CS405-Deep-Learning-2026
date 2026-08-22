# Quiz 5 — Transformers and Large Language Models

*Covers: [Week 13](../lectures/week13-transformers.md) and [Week 14](../lectures/week14-large-language-models.md).*

## Section A — Short answer

**A1.** What key limitation of RNN-based sequence models does the Transformer architecture remove, and what enables it to do so?

**A2.** Write out the self-attention formula and briefly explain the role of each of the three matrices `Q`, `K`, and `V`.

**A3.** Why does the self-attention formula divide by `√d_k` before applying softmax? What would go wrong without this scaling?

**A4.** What is the purpose of multi-head attention, as opposed to using a single attention computation with the full model dimension?

**A5.** Since self-attention has no inherent sense of word order (it's permutation-invariant), how does the Transformer inject positional information?

**A6.** Explain the difference between an encoder-only, decoder-only, and encoder-decoder Transformer architecture, and name one model family representative of each.

**A7.** What is the difference between pre-training and fine-tuning a large language model, and why is this two-stage approach so effective?

**A8.** What problem does LoRA (Low-Rank Adaptation) solve, and how does it reduce the cost of fine-tuning compared to updating all of a model's weights?

**A9.** What is Retrieval-Augmented Generation (RAG), and what specific weakness of LLMs does it help mitigate?

## Section B — Multiple choice

**B1.** In the Transformer's self-attention mechanism, the attention weights are computed by:
(a) A fixed, hand-designed similarity function
(b) A learned linear layer applied once per sequence
(c) The scaled dot product between Query and Key vectors, passed through softmax
(d) A convolution over the sequence

**B2.** Masked (causal) self-attention, used in decoder-only Transformers like GPT, ensures that:
(a) Every token can attend to every other token, including future ones
(b) A token can only attend to itself and earlier tokens in the sequence, not future ones
(c) Only the first token can attend to anything
(d) Attention weights are always uniform

**B3.** BERT is best classified as:
(a) A decoder-only, autoregressive model
(b) An encoder-only model trained with a masked language modeling objective
(c) An encoder-decoder model for translation
(d) A convolutional model for text

**B4.** Fine-tuning an LLM with LoRA works by:
(a) Retraining the entire model from scratch on the new task
(b) Freezing the original weights and learning a small pair of low-rank matrices whose product is added to the original weight matrices
(c) Removing most of the model's layers
(d) Replacing the tokenizer

**B5.** In a RAG pipeline, the retrieved documents are typically:
(a) Used to replace the LLM entirely
(b) Prepended or otherwise inserted into the LLM's input context so it can ground its answer in them
(c) Used only to fine-tune the model offline, never at inference time
(d) Discarded after retrieval and not shown to the model

## Section C — Calculation / applied reasoning

**C1.** In scaled dot-product attention, `Q` and `K` have dimension `d_k = 64`. Two vectors `q` and `k` have a raw dot product of `512`. What is the scaled score before softmax?

**C2.** A Transformer uses 8 attention heads with model dimension `d_model = 512`. What is the dimension of each individual head's `Q`, `K`, and `V` projections (assuming the standard even split)?

**C3.** You're fine-tuning a 7-billion-parameter LLM for a narrow, low-resource use case (limited GPU memory, need fast iteration) and you also want to preserve as much of the model's general knowledge as possible. Would you choose full fine-tuning or LoRA, and justify your answer using at least two properties of LoRA discussed in Week 14.

**C4.** A chatbot built on a base LLM (no retrieval) confidently answers questions about events from *after* its training data cutoff, and gets them wrong. Explain, referencing RAG, how you would fix this without retraining the model.

## Answer Key

**A1.** The Transformer removes the *sequential* dependency of RNNs — an RNN must process timestep `t` before it can process timestep `t+1`, since `h_t` depends on `h_{t-1}`, which prevents parallelizing computation across the sequence during training. Self-attention computes relationships between *all* pairs of positions in a sequence simultaneously via matrix operations, with no step-by-step recurrence, which allows the entire sequence to be processed in parallel on modern hardware (a major reason Transformers scale so much better than RNNs to large datasets and models).

**A2.** `Attention(Q,K,V) = softmax(QKᵀ/√d_k)V`. `Q` (Query) represents "what this position is looking for"; `K` (Key) represents "what each position offers/can be matched against"; `V` (Value) is the actual content/information carried by each position that gets aggregated once the attention weights (computed from `Q` and `K`) determine how much to weight each position.

**A3.** As `d_k` (the dimension of the query/key vectors) grows, the raw dot products `QKᵀ` tend to grow large in magnitude (since they're a sum over `d_k` terms), which pushes the softmax function into regions with extremely small gradients (saturating near one-hot outputs). Dividing by `√d_k` rescales the dot products back down to a range where softmax has well-behaved, informative gradients, keeping training stable regardless of the chosen dimension.

**A4.** A single attention computation can only learn one "pattern" of relationships between positions at a time (e.g., all attending based on one notion of similarity). Multi-head attention runs several smaller attention computations in parallel, each with its own learned `Q`, `K`, `V` projections into a lower-dimensional subspace, letting different heads specialize in capturing different kinds of relationships (e.g., syntactic dependencies, coreference, local vs. long-range patterns) simultaneously; their outputs are concatenated and linearly projected back to the model dimension.

**A5.** The Transformer adds a positional encoding vector to each token's input embedding before it enters the self-attention layers — either a fixed, deterministic pattern (e.g., the original sinusoidal encoding, using sine/cosine functions of different frequencies per dimension) or a learned embedding indexed by position. This injects information about each token's position in the sequence directly into its representation, compensating for self-attention's inherent lack of order-sensitivity.

**A6.** Encoder-only architectures (e.g., BERT) process the whole input bidirectionally (every token can attend to every other token, including ones that come after it) and are well suited to understanding/representation tasks like classification or extracting embeddings. Decoder-only architectures (e.g., the GPT family) use causal/masked self-attention (a token can only attend to itself and earlier tokens) and are trained to predict the next token, making them naturally suited to open-ended text generation. Encoder-decoder architectures (e.g., T5, or the original Transformer for machine translation) use a bidirectional encoder to process the input plus a causal decoder that attends to both its own previous outputs and the encoder's output, suited to sequence-to-sequence tasks like translation or summarization where the output is a transformed version of the input.

**A7.** Pre-training is the (very expensive, typically unsupervised/self-supervised) initial training phase where a model learns general language understanding and world knowledge from a massive, broad text corpus, usually via a next-token or masked-token prediction objective. Fine-tuning is a subsequent, much cheaper phase where the pre-trained model's weights are further adjusted on a smaller, task-specific or domain-specific dataset to specialize its behavior. This two-stage approach is effective because pre-training lets the model absorb broad linguistic and world knowledge that would be far too expensive to learn from any single narrow, labeled dataset, while fine-tuning efficiently redirects that general capability toward the specific behavior needed, using far less task-specific data and compute than training from scratch would require.

**A8.** LoRA addresses the cost (in GPU memory and compute) of fine-tuning very large models by updating *all* of their parameters — for a model with billions of parameters, storing gradients and optimizer states for every weight during fine-tuning is often prohibitively expensive. LoRA instead freezes the original pre-trained weight matrices entirely and, for selected weight matrices, learns a much smaller pair of low-rank matrices (`A` and `B`, with a small rank `r`) whose product `BA` is added to the frozen original weight at inference time. Since `r` is chosen to be small, the number of trainable parameters is a tiny fraction of the full model's parameter count, dramatically reducing memory and compute needed for fine-tuning while keeping the frozen base weights (and thus most of the model's general knowledge) intact.

**A9.** RAG augments an LLM by retrieving relevant documents or passages from an external knowledge source (typically via a vector similarity search over embedded documents) and inserting them into the model's input context before generation, so the model's answer is grounded in retrieved, up-to-date, or domain-specific text rather than relying solely on what was memorized during pre-training. This directly mitigates the problem of LLMs producing outdated answers (anything after their training data cutoff) or hallucinating facts, since the model can now cite and draw from real, retrievable source material provided at inference time.

**B1.** (c) The scaled dot product between Query and Key vectors, passed through softmax.

**B2.** (b) A token can only attend to itself and earlier tokens in the sequence, not future ones.

**B3.** (b) An encoder-only model trained with a masked language modeling objective.

**B4.** (b) Freezing the original weights and learning a small pair of low-rank matrices whose product is added to the original weight matrices.

**B5.** (b) Prepended or otherwise inserted into the LLM's input context so it can ground its answer in them.

**C1.** `512 / √64 = 512 / 8 = 64`.

**C2.** `512 / 8 = 64` — each head's `Q`, `K`, `V` projections have dimension 64.

**C3.** LoRA is the better fit. Two supporting properties from Week 14: (1) LoRA freezes the base model's weights and only trains a small number of additional low-rank parameters, which drastically reduces GPU memory requirements (no need to store optimizer state for billions of frozen parameters) — directly addressing the limited-GPU-memory constraint and enabling much faster iteration than full fine-tuning. (2) Because the original pre-trained weights are never modified (only a small additive update is learned and can even be merged or swapped out per task), the model's general pre-trained knowledge is preserved far better than with full fine-tuning, where updating every weight risks "catastrophic forgetting" of general capabilities in favor of the narrow fine-tuning task.

**C4.** This is a case for RAG: rather than retraining or re-pre-training the model (expensive and only pushes the cutoff problem forward in time again), you'd connect the chatbot to a retrieval system indexing up-to-date documents (e.g., recent news, a regularly refreshed knowledge base), and at query time retrieve the most relevant passages about the recent events and insert them into the model's context window alongside the user's question. The model then generates its answer grounded in that retrieved, current text instead of relying on its frozen, outdated training-time knowledge — and the knowledge source can keep being updated indefinitely without ever retraining the underlying LLM.
