# Assignments

This folder contains the graded assignments for CS-405 Deep Learning. Each assignment builds directly on the concepts covered in the corresponding weeks of [`lectures/`](../lectures/) and is meant to be started only after you have read the relevant week's notes and worked through the matching [`labs/`](../labs/).

| # | Assignment | Covers weeks | Topics |
|---|---|---|---|
| 1 | [Neural Networks from Scratch](assignment01-neural-network-from-scratch.md) | 1–2 | Perceptrons, backpropagation, training tricks |
| 2 | [CNNs for Image Classification](assignment02-cnn-image-classification.md) | 3–5 | Convolution, modern architectures, transfer learning, detection/segmentation |
| 3 | [Sequence Models for Text](assignment03-sequence-models-for-text.md) | 6–7, 11–12 | RNN/GRU/LSTM, word embeddings, tagging |
| 4 | [Generative Models](assignment04-generative-models.md) | 9–10 | Autoencoders, VAEs, GANs |
| 5 | [Transformers and LLMs](assignment05-transformers-and-llms.md) | 13–14 | Self-attention, mini-Transformer, fine-tuning, RAG |
| — | [Final Project](final-project.md) | any | Open-ended capstone, may include Week 15 (RL) |

## General submission guidelines

Unless the instructor states otherwise for a specific assignment:

- Submit a single ZIP archive (or a link to your own GitHub repository/fork) containing your code, any trained model checkpoints requested, and a short PDF or Markdown report.
- Your report should describe your approach, any design decisions or hyperparameters you chose and why, the results you obtained (with plots/tables where relevant), and a brief discussion of what worked, what didn't, and why.
- All code must run end-to-end from a clean environment; include a `requirements.txt` or note any special dependencies.
- Cite any external code, tutorials, or papers you referenced. Using publicly available code is fine as a learning aid, but you must understand and be able to explain everything you submit — copying a solution without understanding it defeats the purpose and will be treated as an academic integrity violation.
- Late submissions: follow the course's standard late-day policy as announced in class (not encoded here — check with the instructor for this semester's policy).

## Grading

Each assignment is typically graded on: correctness of implementation (50%), quality and clarity of experiments/results (25%), and quality of the written report/analysis (25%), unless a per-assignment rubric overrides this.
