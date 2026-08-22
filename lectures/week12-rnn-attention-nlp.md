# Week 12 — From Word Embeddings to Attention: RNNs for Classification, Tagging, and Neural Machine Translation

*Companion notes for [`slides/lecture_week12.pdf`](../slides/lecture_week12.pdf)*

## Why this week matters

Week 11 gave us a way to turn individual words into meaningful vectors, and Weeks 6–7 gave us RNNs/LSTMs that can process sequences of vectors while maintaining memory. This week combines them, applying RNNs directly to three concrete language tasks — sentence classification, sequence tagging, and machine translation — and then confronts the biggest weakness of the basic RNN-based translation approach: squeezing an entire sentence into one fixed-size vector. That weakness motivates the **attention mechanism**, which we introduce here and which becomes the single most important building block of the Transformer in Week 13.

## 1. Why word embeddings alone aren't enough

Word embeddings (Week 11) give us a meaningful vector per word, but sentences aren't just unordered bags of vectors — order matters, and meaning depends on sequence: "Dog bites man" and "Man bites dog" contain the exact same words but mean opposite things. The fix, as in Weeks 6–7, is to feed the sequence of word embeddings through an RNN, which processes words one at a time while maintaining a hidden state that accumulates context. For a sentence of embeddings `e_1, ..., e_T`, the RNN computes `h_t = f(W·h_{t-1} + U·e_t + b)` at each step, so `h_t` ends up containing a summary of everything seen up through position `t`. What varies across tasks is simply *what you do with the hidden states* — this week walks through three different answers.

## 2. Task 1: Sentiment classification (many-to-one)

**Goal:** assign a single label (Positive/Negative/Neutral) to an entire sentence. The natural approach: run the RNN across the whole sentence, and use *only the final hidden state* `h_T` — which, by construction, has already absorbed information from every word in the sentence — as input to a small classifier: `score = W_class · h_T + b_class`, followed by softmax. For "I love this product, it's amazing!", the hidden state evolves step by step, accumulating "I" → "I love" → "I love this" → ... → the complete sentence, and the final hidden state `h_6` is fed to the classifier to produce something like `P(Positive)=0.92, P(Negative)=0.05, P(Neutral)=0.03`. This is a classic **many-to-one** sequence task: many input tokens, one output label.

## 3. Task 2: Named Entity Recognition (many-to-many, synchronous)

**Goal:** label *every individual word* with an entity type, e.g., person (PER), organization (ORG), location (LOC), or "other" (O) — as in "Apple Inc. was founded by Steve Jobs in Cupertino" → `B-ORG I-ORG O O O B-PER I-PER O B-LOC` (the `B-`/`I-` prefixes mark the beginning and continuation of a multi-word entity span). This is structurally different from sentiment classification: instead of one prediction from the final hidden state, we need a prediction at **every single position**, so each hidden state `h_t` feeds its *own* local classifier (softmax) to produce a per-token label. This is a **many-to-many, synchronous** task — the output sequence has exactly the same length as the input sequence, with one label emitted per input token. The slides also note that context from *both directions* generally helps for tagging tasks like this — knowing the words that come *after* a token, not just before it, can disambiguate its role — which motivates using a **bidirectional RNN** (running one RNN left-to-right and another right-to-left, then combining their hidden states at each position) rather than a plain single-direction RNN.

## 4. Task 3: Machine translation (many-to-many, asynchronous)

Translation is harder still, because the input and output sequences can have **different lengths**, and, unlike NER, the output isn't produced in lockstep with the input — translation only starts *after* the entire source sentence has been read. Compare the three task shapes side by side: sentence classification is many-to-one (sequence in, single label out); NER/tagging is many-to-many *synchronous* (sequence in, same-length sequence out, aligned position-by-position); machine translation is many-to-many but *asynchronous* (sequence in, a possibly different-length sequence out, generated only once the whole input has been consumed).

### The encoder-decoder architecture

The standard solution is the **encoder-decoder** architecture. The **encoder** is an RNN that reads the source sentence (e.g., French) one token at a time, `h_t = RNN(h_{t-1}, e(x_t))`, and its *final* hidden state, after reading the whole sentence including an end-of-sentence marker, becomes a single **context vector `c`** — a fixed-size summary of the entire input sentence. The **decoder** is a *separate* RNN that generates the target sentence (e.g., English) one token at a time, conditioned on that context vector: `s_t = RNN(s_{t-1}, [e(y_{t-1}); c])`, followed by `P(y_t | y_{<t}, c) = softmax(W·s_t + b)` — at each step, the decoder looks at its own previous hidden state, the *embedding of the word it just generated*, and the fixed context vector `c`, to decide what to generate next.

Concretely, generating "I am a student" from context vector `c`: step 1 uses `c` and a special start token to predict "I"; step 2 uses the previous decoder state and the embedding of "I" (the word just generated) to predict "am"; step 3 similarly produces "a"; step 4 produces "student"; step 5 produces an end-of-sentence token, at which point generation stops.

Because source and target are different languages, translation requires **two separate embedding matrices** — a source embedding matrix `E_src` (used by the encoder to embed French words) and a target embedding matrix `E_tgt` (used by the decoder both to embed previously-generated English words as input and, transposed, to produce output probabilities over the English vocabulary) — both learned jointly during training, exactly like the embeddings from Week 11, just specialized to each language and to this specific translation task.

### Training vs. inference

During **training**, the model has access to a **parallel corpus** of (source sentence, correct target sentence) pairs. A technique called **teacher forcing** is used: the encoder processes the French sentence to produce `c`; the decoder is fed the *actual* correct English sentence (shifted by one position, preceded by a start token) as its input at every step, rather than its own (possibly wrong) previous predictions; the decoder's output at each position is compared against the true next word via cross-entropy loss; and the loss is backpropagated through both the decoder and the encoder. Using the ground-truth previous word (rather than the model's own guess) as input at each training step keeps training stable and prevents early mistakes from cascading and corrupting the rest of the training signal.

At **inference** time, there is no ground truth to feed in — the model must generate purely from its own predictions. The simplest strategy, **greedy decoding**, feeds the encoder's context vector `c` and a start token to the decoder, takes whatever word the decoder predicts as *most likely*, feeds that predicted word back in as the next input, and repeats until an end-of-sentence token is generated. Translating "Je mange une pomme," for example, might unfold as: `<SOS>` → predicts "I"; `<SOS> I` → predicts "eat"; `<SOS> I eat` → predicts "an"; `<SOS> I eat an` → predicts "apple"; `<SOS> I eat an apple` → predicts `<EOS>`, ending generation.

## 5. The problem with a single fixed context vector

The basic encoder-decoder design has a serious structural weakness: the *entire* source sentence, however long, gets compressed into **one single fixed-size vector** `c`. For short sentences this can work reasonably well, but as sentences get longer, information from early words tends to get diluted or lost by the time the encoder finishes reading (echoing the vanishing-memory issues from Weeks 6–7), and — just as importantly — the decoder has **no way to "look back"** at any specific part of the source sentence once generation has started; it only ever has access to the one compressed summary. This is often called the **information bottleneck** problem.

## 6. The attention mechanism: letting the decoder look back

Human translators don't work by memorizing an entire sentence and then producing a translation from memory alone — they read along and pay attention to the *relevant part* of the source sentence as they produce each word of the translation: when generating "the," attend to "Le"; when generating "black," attend to "noir"; when generating "cat," attend to "chat"; and so on. **Attention** gives the decoder exactly this capability.

Instead of a single fixed context vector shared across every decoder step, attention computes a **different, dynamically weighted context vector at every decoder step**, built from *all* of the encoder's hidden states, not just the last one:

```
c⟨t⟩ = Σ_j α⟨t,j⟩ · h_j          where  Σ_j α⟨t,j⟩ = 1,  α⟨t,j⟩ ≥ 0
```

Here `α⟨t,j⟩` is the **attention weight** — how much decoder step `t` should focus on encoder position `j` — and the context vector `c⟨t⟩` for that decoder step is simply a weighted average of *all* the encoder's hidden states, with more weight placed on the positions that matter most for generating the current output word.

### Computing the attention weights

Attention weights are computed in two steps. First, an **alignment score** `e⟨t,j⟩ = score(s_{t-1}, h_j)` measures how well encoder hidden state `h_j` "matches" the decoder's current state — a common choice is a bilinear score, `s_{t-1}^T · W_a · h_j`, though other scoring functions are possible (we'll see a closely related but distinct scoring approach — scaled dot-product attention — in Week 13). Second, these raw scores are normalized into proper weights with a **softmax** across all encoder positions: `α⟨t,j⟩ = exp(e⟨t,j⟩) / Σ_k exp(e⟨t,k⟩)` — guaranteeing the weights for a given decoder step are all non-negative and sum to exactly 1, exactly like the softmax attention weighting we'll formalize further next week.

### Putting it together

The complete attention-augmented NMT forward pass, layer by layer:

```
Encoder:    h⟨t⟩ = RNN_enc(h⟨t-1⟩, x⟨t⟩)
Attention:  e⟨t',t⟩ = score(s⟨t'-1⟩, h⟨t⟩)
            α⟨t',t⟩ = softmax_t(e⟨t',t⟩)
            c⟨t'⟩ = Σ_t α⟨t',t⟩ · h⟨t⟩
Decoder:    s⟨t'⟩ = RNN_dec(s⟨t'-1⟩, [y⟨t'-1⟩; c⟨t'⟩])
            ŷ⟨t'⟩ = softmax(W_y · s⟨t'⟩ + b_y)
```

Notice that a *fresh* context vector `c⟨t'⟩` is computed for every single decoder step `t'`, each one a different weighted blend of the encoder's hidden states, rather than the single, shared `c` used throughout the basic encoder-decoder.

Collecting all the attention weights across every (decoder step, encoder position) pair gives an **attention matrix** `A`, where each row corresponds to one decoder output and sums to 1 across the encoder positions. For "Je suis étudiant" → "I am a student," a well-trained attention matrix looks intuitively sensible: generating "I" attends mostly to "Je" (weight 0.70), generating "am" attends mostly to "suis" (0.80), and generating "student" attends mostly to "étudiant" (0.95) — the attention weights essentially recover the correct word-to-word alignment between the two languages, learned entirely from data with no explicit alignment supervision.

## 7. Why attention works, and why it matters going forward

Without attention, a single shared context vector is used for every decoder step, so early source words are easily diluted or forgotten and performance degrades noticeably on long sentences. With attention, each decoder step gets its own dynamically computed context vector, tailored to what's currently being generated, letting the decoder "look back" at any relevant source word regardless of how far away it is in the sentence — which substantially improves performance on long sentences specifically. Attention brings four concrete benefits: it directly solves the fixed-vector information bottleneck; it provides genuine **interpretability**, since you can literally visualize which source words the model attended to when producing each output word (as in the attention-matrix example above); it gives gradients a more direct path back to the *relevant* encoder states (rather than forcing every gradient to route through one compressed vector), which helps training; and — most importantly for the rest of this course — **attention is the essential building block that next week's Transformer architecture generalizes and scales up**, removing the RNN entirely and building a model out of attention alone.

## Key takeaways

RNNs can be adapted to a range of language tasks simply by choosing what to do with their hidden states: use only the final hidden state for whole-sequence classification (sentiment analysis), use every hidden state independently for synchronous per-token tagging (NER), or use an encoder's final hidden state as a compressed context vector fed into a separate decoder RNN for asynchronous sequence-to-sequence generation (machine translation), trained with teacher forcing and run at inference time with greedy (or more sophisticated) decoding. The basic encoder-decoder's single fixed context vector is a serious bottleneck for long sequences, and the attention mechanism fixes it by letting the decoder compute a fresh, dynamically weighted combination of *all* encoder hidden states at every generation step — weights derived from a simple score-then-softmax computation that, as we'll see next week, generalizes directly into the self-attention mechanism at the heart of the Transformer.
