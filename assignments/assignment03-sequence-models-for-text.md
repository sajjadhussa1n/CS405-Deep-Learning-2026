# Assignment 3 — Sequence Models for Text

**Covers:** Week 6 (RNNs to GRUs), Week 7 (LSTM), Week 11 (Word Embeddings), Week 12 (RNNs for Language Tasks)
**Deliverable:** PyTorch code + written report

## Learning objectives

Train word embeddings and evaluate what they capture; implement and compare plain RNN, GRU, and LSTM cells on a task where long-range dependencies matter; and build a per-token sequence tagger, connecting the RNN/embedding material to a concrete NLP application.

## Part A — Word embeddings

1. Train Word2vec-style (skip-gram with negative sampling, Week 11) embeddings on a medium-sized text corpus of your choice (e.g., a few novels from Project Gutenberg, or a Wikipedia dump subset). You may implement the training loop yourself in PyTorch, or use an existing implementation (e.g., `gensim`) — but if you use `gensim`, additionally implement the negative-sampling loss computation by hand for at least one batch to demonstrate you understand the mechanics from Week 11.
2. Evaluate your trained embeddings with: (a) nearest-neighbor queries for at least 5 chosen words (do the nearest neighbors make semantic sense?), and (b) at least 5 analogy tasks in the style of Week 11's `king - man + woman ≈ queen` examples, using your own corpus's vocabulary. Report your success rate and discuss any failures.
3. Visualize a sample of your embeddings in 2D using t-SNE or PCA, and discuss any clusters you observe.

## Part B — RNN vs. GRU vs. LSTM on a long-range-dependency task

Design (or use a standard) synthetic task where the correct output at the end of a sequence depends on information seen near the *beginning* of a long sequence (e.g., a "copy the first token" task with increasing sequence length, or a sentiment task built so that a negation early in the sentence flips the overall label).

1. Implement a plain RNN, a GRU, and an LSTM cell (Weeks 6–7) — implementing at least one of the three from scratch in PyTorch (i.e., writing out the gate equations yourself rather than calling `nn.LSTM`/`nn.GRU`) is required; you may use built-in layers for the other two.
2. Train all three on your chosen task at several sequence lengths (e.g., 10, 30, 60, 100 steps) and plot accuracy (or loss) vs. sequence length for each architecture.
3. Reproducing the spirit of the "vanishing gradient" visualizations in Weeks 6–7, log and plot the gradient norm reaching back to the first time step as a function of sequence length, for each architecture. Discuss whether your empirical results match the theoretical expectation (plain RNN degrades fastest, GRU and LSTM are more robust).

## Part C — Named Entity Recognition (sequence tagging)

Using a bidirectional RNN/GRU/LSTM (your choice of cell, informed by Part B's results) on top of pre-trained or your own trained embeddings from Part A, build a per-token tagger for a Named Entity Recognition dataset (e.g., CoNLL-2003, or any similarly-formatted labeled dataset approved by the instructor), following the many-to-many tagging pattern from Week 12.

Report token-level accuracy and (ideally) entity-level F1 score on a held-out test set, and show at least 3 example sentences with your model's predicted tags alongside the ground truth.

## Report requirements

Include your embedding evaluation results (nearest neighbors, analogies, visualization); your RNN/GRU/LSTM comparison plots (accuracy vs. sequence length, and gradient norm vs. sequence length) with a written discussion connecting the results to the theory in Weeks 6–7; and your NER results table and example predictions.

## Grading rubric

| Component | Weight |
|---|---|
| Part A: embedding training and evaluation | 25% |
| Part B: RNN/GRU/LSTM implementation and comparison experiments | 40% |
| Part C: NER tagger implementation and results | 25% |
| Report clarity | 10% |
