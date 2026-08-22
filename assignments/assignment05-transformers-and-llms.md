# Assignment 5 — Transformers and Large Language Models

**Covers:** Week 13 (Transformer), Week 14 (Large Language Models)
**Deliverable:** Python/PyTorch + Hugging Face `transformers` code + written report

## Learning objectives

Implement the core self-attention and multi-head attention mechanics from scratch; build and train a small Transformer encoder on a sequence task; and get practical, hands-on experience with the modern pre-train → fine-tune workflow using a real pre-trained LLM, including parameter-efficient fine-tuning and a minimal RAG pipeline.

## Part A — Self-attention and multi-head attention from scratch

Using only NumPy or raw PyTorch tensor operations (no `nn.MultiheadAttention` or `nn.TransformerEncoderLayer`), implement:

1. Scaled dot-product attention: `Attention(Q,K,V) = softmax(QK^T / √d_k) V` (Week 13), given `Q`, `K`, `V` matrices as input. Verify your implementation against a small hand-worked example (e.g., a toy 4-token sequence with hand-chosen `Q`, `K`, `V` values) where you compute the expected output by hand and confirm your code matches.
2. Multi-head attention built on top of your single-head implementation, including the per-head learned projections and the final output projection `W^O` (Week 13).
3. A demonstration that scaling by `√d_k` matters: compute attention weights with and without the scaling factor for a case with a large `d_k` (e.g., 256+), and show (with a plot or printed softmax outputs) that omitting the scale produces a much more "peaked" (near one-hot) attention distribution, as discussed in Week 13.

## Part B — A small Transformer encoder

Using your Part A attention implementation (or PyTorch's built-in layers, your choice for this part), assemble a small Transformer encoder: multi-head self-attention + Add & Norm + position-wise feedforward + Add & Norm (Week 13), stacked for at least 2 layers, with sinusoidal positional encoding.

Train it on a sequence classification task of your choice (e.g., sentiment classification on the same dataset style used in Assignment 3, so you can directly compare against your RNN/GRU/LSTM results). Report accuracy and training time, and briefly compare against your best Assignment 3 result on a comparable task — discuss any differences you observe in accuracy, training speed, or ease of training.

## Part C — Fine-tuning a pre-trained LLM with Hugging Face

1. Using the Hugging Face `transformers` library (Week 14), load a small pre-trained model appropriate for your task (e.g., `distilbert-base-uncased` for classification, or a small decoder-only model such as `gpt2` or `distilgpt2` for generation) and run it zero-shot on a task of your choice, following the `pipeline` pattern shown in Week 14. Report its out-of-the-box performance.
2. Fine-tune the model on a small labeled dataset for your chosen task using **LoRA** (Week 14, via the `peft` library) rather than full fine-tuning. Report the number of trainable parameters LoRA actually updates (as a percentage of the total model), and compare performance before and after fine-tuning.
3. Briefly compare LoRA fine-tuning against full fine-tuning on the same task and dataset (full fine-tuning may be limited to a very small model if you have limited compute) in terms of training time, GPU memory usage, and final performance. Discuss the trade-off in your own words, connecting it to the Week 14 material on parameter-efficient fine-tuning.

## Part D — A minimal Retrieval-Augmented Generation (RAG) pipeline

Build a small RAG pipeline (Week 14) over a document collection of your choice (e.g., a handful of Wikipedia articles, your own course's lecture notes, or any small text corpus): embed your documents (you may reuse the embedding techniques from Week 11, or use a pre-trained sentence-embedding model), implement (or use a lightweight library for) similarity-based retrieval given a query, and pass the retrieved context into your Part C LLM to generate a grounded answer.

Demonstrate your pipeline on at least 5 example questions, showing the retrieved passages and the final generated answer for each. Include at least one example where retrieval clearly improves the answer compared to asking the LLM the same question with no retrieved context (i.e., a case where the ungrounded model would likely hallucinate or lack the specific information).

## Report requirements

Include your hand-verified attention example and scaling demonstration from Part A; your Transformer-encoder training results and comparison to Assignment 3 from Part B; your zero-shot vs. LoRA-fine-tuned comparison table and parameter-count discussion from Part C; and your RAG pipeline's example queries/retrieved passages/answers from Part D, including the "RAG helps here" example.

## Grading rubric

| Component | Weight |
|---|---|
| Part A: from-scratch (multi-head) attention implementation and verification | 25% |
| Part B: Transformer encoder training and comparison to Assignment 3 | 20% |
| Part C: zero-shot baseline and LoRA fine-tuning | 25% |
| Part D: RAG pipeline and example queries | 20% |
| Report clarity | 10% |
