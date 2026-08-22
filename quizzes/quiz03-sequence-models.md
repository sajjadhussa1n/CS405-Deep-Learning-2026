# Quiz 3 — Sequence Models

*Covers: [Week 6](../lectures/week06-sequential-models-rnn-gru.md), [Week 7](../lectures/week07-lstm.md), [Week 11](../lectures/week11-word-embeddings.md), and [Week 12](../lectures/week12-rnn-attention-nlp.md).*

## Section A — Short answer

**A1.** What structural property of an RNN allows it to process sequences of arbitrary length, and what is the "hidden state" conceptually carrying from one timestep to the next?

**A2.** What is Backpropagation Through Time (BPTT), and why does it lead to vanishing/exploding gradients specifically for *long* sequences?

**A3.** LSTM introduces a separate "cell state" alongside the hidden state, and controls information flow with gates. Name the three gates and, in one phrase each, what each one controls.

**A4.** How does a GRU differ structurally from an LSTM (in terms of gates and states), and what's the practical tradeoff between the two?

**A5.** Explain, conceptually, what a word embedding is and why "king − man + woman ≈ queen" is considered a striking demonstration of what embeddings capture.

**A6.** In an attention mechanism, what problem does attention solve that a plain fixed-length encoder-decoder RNN (with no attention) struggles with, especially on long sentences?

## Section B — Multiple choice

**B1.** The vanishing gradient problem is generally worse in vanilla RNNs than in feedforward networks of similar depth because:
(a) RNNs have more parameters
(b) The same weight matrix is multiplied repeatedly at every timestep, so its effect compounds multiplicatively over the sequence length
(c) RNNs don't use backpropagation
(d) RNNs cannot use ReLU activations

**B2.** Which gate in an LSTM is responsible for deciding what old information to remove from the cell state?
(a) Input gate (b) Output gate (c) Forget gate (d) Update gate

**B3.** Word2Vec's Skip-gram objective trains embeddings to:
(a) Predict a word's part of speech
(b) Predict surrounding context words given a center word
(c) Translate a word into another language
(d) Classify documents by topic

**B4.** Compared to GloVe, Word2Vec is best described as:
(a) A count-based method using a global co-occurrence matrix
(b) A predictive, neural network-based method trained via a local context window objective
(c) A rule-based method with no learned parameters
(d) A method that only works for rare words

**B5.** In Bahdanau-style attention for sequence-to-sequence models, the attention weights at each decoding step are computed as a function of:
(a) Only the current decoder hidden state
(b) Only the encoder outputs
(c) A compatibility score between the current decoder state and every encoder hidden state
(d) The final loss value

## Section C — Calculation / short exercise

**C1.** A vanilla RNN has a recurrent weight with the largest eigenvalue equal to `0.5`. Roughly what happens to a gradient signal after being backpropagated through 20 timesteps, and why? (Order-of-magnitude reasoning is fine — you don't need an exact number.)

**C2.** Two words have embedding vectors `u = [1, 0, 1]` and `v = [1, 1, 0]`. Compute their cosine similarity (`u·v / (‖u‖‖v‖)`).

**C3.** Name one advantage attention-based models have over vanilla encoder-decoder RNNs for machine translation of long sentences, tying your answer to a specific bottleneck attention removes.

## Answer Key

**A1.** An RNN reuses the *same* weight matrices at every timestep, applying the identical recurrence relation (`h_t = f(W_hh·h_{t-1} + W_xh·x_t + b)`) regardless of how many timesteps there are — this parameter sharing across time is what lets one fixed set of weights handle sequences of any length. The hidden state `h_t` is conceptually a compressed summary of everything the network has seen in the sequence up to and including timestep `t`, that it carries forward to influence how it processes future inputs.

**A2.** BPTT is backpropagation applied to an RNN "unrolled" across time — the network is treated as a very deep feedforward network with one "layer" per timestep, all sharing the same weights, and gradients are computed by the chain rule back through every timestep to the start of the sequence. Because the *same* recurrent weight matrix is multiplied at every one of those timesteps, the gradient involves that matrix raised to a high power (once per timestep); if its dominant eigenvalue is less than 1 the gradient shrinks exponentially with sequence length (vanishing), and if greater than 1 it grows exponentially (exploding) — the longer the sequence, the more extreme the effect.

**A3.** Forget gate: controls what fraction of the existing cell state to keep vs. discard. Input gate: controls how much of the new candidate information to write into the cell state. Output gate: controls how much of the (updated) cell state to expose as the hidden state / output at this timestep.

**A4.** A GRU merges the LSTM's cell state and hidden state into a single hidden state, and replaces the LSTM's three gates with two: an update gate (which blends the roles of the forget and input gates) and a reset gate. The practical tradeoff: GRUs have fewer parameters and are cheaper/faster to train, often matching LSTM performance on many tasks, while LSTMs' extra cell state and gate give it more representational capacity and can win out on tasks requiring finer-grained control over long-range memory.

**A5.** A word embedding is a dense, learned vector representation of a word, positioned in a continuous vector space such that words with similar meanings or usage contexts end up close together. The "king − man + woman ≈ queen" result is striking because it shows the embedding space isn't just clustering similar words — it has learned linear directions that correspond to meaningful semantic relationships (here, roughly a "gender" direction), so vector arithmetic on the embeddings performs analogical reasoning, something not explicitly trained for.

**A6.** A plain encoder-decoder RNN compresses the *entire* input sequence into a single fixed-length context vector (the encoder's final hidden state), which the decoder must rely on for every output step. For long sentences, this is an information bottleneck — a single fixed-size vector cannot losslessly retain everything needed from a long input, and performance degrades noticeably as sentence length grows. Attention solves this by letting the decoder look back at *all* encoder hidden states at every decoding step, weighted by relevance, instead of relying on one compressed summary.

**B1.** (b) The same weight matrix is multiplied repeatedly at every timestep, so its effect compounds multiplicatively over the sequence length.

**B2.** (c) Forget gate.

**B3.** (b) Predict surrounding context words given a center word.

**B4.** (b) A predictive, neural network-based method trained via a local context window objective.

**B5.** (c) A compatibility score between the current decoder state and every encoder hidden state.

**C1.** The gradient shrinks roughly by a factor of `0.5^20 ≈ 0.00000095` (about `9.5 × 10⁻⁷`) — essentially vanishing to zero. This happens because the gradient backpropagated through the recurrence is (approximately) multiplied by the recurrent weight matrix once per timestep, so its magnitude scales with the largest eigenvalue raised to the power of the number of timesteps; with an eigenvalue below 1, that power shrinks exponentially fast as the sequence gets longer.

**C2.** `u·v = 1×1 + 0×1 + 1×0 = 1`. `‖u‖ = √(1+0+1) = √2`. `‖v‖ = √(1+1+0) = √2`. Cosine similarity = `1 / (√2 × √2) = 1/2 = 0.5`.

**C3.** Any reasonable answer tied to the fixed-length context vector bottleneck is acceptable, e.g.: attention removes the requirement to compress the entire source sentence into one fixed-size vector, which lets the model retain and directly access fine-grained information from every source position (including early words in a long sentence) at the exact moment it's needed during decoding, rather than having that information degraded or lost by the time decoding reaches later output words.
