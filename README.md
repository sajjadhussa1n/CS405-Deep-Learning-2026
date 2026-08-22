# CS-405: Deep Learning

**Instructor:** Dr. Sajjad Hussain (sajjad.hussain2@seecs.edu.pk)
**Department:** Electrical and Computer Engineering, SEECS, NUST
**Offering:** Spring 2026

Unofficial course repository for CS-405 Deep Learning. It contains the complete set of lecture slides, week-by-week written explanations of every lecture, hands-on labs, assignments, handouts, and quizzes used throughout the semester. Students should clone this repository at the start of the semester and pull new material as it is released each week.

---

## Course Description

CS-405 Deep Learning is a graduate/senior-undergraduate level course that builds a deep, practical, and mathematically grounded understanding of modern neural network methods. The course starts from the biological inspiration behind artificial neurons and progresses through feedforward networks, convolutional neural networks for vision, recurrent and attention-based architectures for sequences, generative models, and modern large language models, before closing with an introduction to reinforcement learning.

The course balances theory (derivations, architectural reasoning, and the "why" behind each design choice) with hands-on implementation (Python, NumPy, and PyTorch) so that students leave the course able to both understand and build state-of-the-art deep learning systems.

## Course Contents

The course is organized into 15 weekly modules. Detailed, easy-to-follow written notes for every week are available in [`lectures/`](lectures/), and the original slide decks are in [`slides/`](slides/).

| Week | Topic | Slides | Notes |
|------|-------|--------|-------|
| 1 | Introduction to Neural Networks — biological inspiration, the perceptron, AI winters, multi-layer networks, and how networks learn | [slides](slides/lecture_week01.pdf) | [notes](lectures/week01-introduction-to-neural-networks.md) |
| 2 | Training Neural Networks in Depth — vanishing gradients, weight initialization, activation functions, batch normalization, transfer learning, optimizers, and regularization | [slides](slides/lecture_week02.pdf) | [notes](lectures/week02-training-neural-networks.md) |
| 3 | Introduction to Convolutional Neural Networks — convolution, pooling, and the visual cortex analogy | [slides](slides/lecture_week03.pdf) | [notes](lectures/week03-introduction-to-cnns.md) |
| 4 | Deep CNN Architectures — AlexNet, VGGNet, 1×1 convolutions, ResNet, Inception, depthwise-separable convolutions, MobileNet V1/V2 | [slides](slides/lecture_week04.pdf) | [notes](lectures/week04-deep-cnn-architectures.md) |
| 5 | Advanced Computer Vision — object detection, IoU and non-max suppression, semantic segmentation, transpose convolutions, U-Net | [slides](slides/lecture_week05.pdf) | [notes](lectures/week05-advanced-computer-vision.md) |
| 6 | Sequential Models: RNNs to GRUs — recurrent networks, backpropagation through time, gated recurrent units | [slides](slides/lecture_week06.pdf) | [notes](lectures/week06-sequential-models-rnn-gru.md) |
| 7 | Long Short-Term Memory (LSTM) | [slides](slides/lecture_week07.pdf) | [notes](lectures/week07-lstm.md) |
| 8 | Midterm Examination / Review Week | — | — |
| 9 | Autoencoders and Variational Autoencoders (VAEs) | [slides](slides/lecture_week09.pdf) | [notes](lectures/week09-autoencoders-vae.md) |
| 10 | Generative Adversarial Networks (GANs) — DCGAN, WGAN, Conditional GANs | [slides](slides/lecture_week10.pdf) | [notes](lectures/week10-gans.md) |
| 11 | Word Embeddings — one-hot vs. dense representations, Word2Vec, GloVe | [slides](slides/lecture_week11.pdf) | [notes](lectures/week11-word-embeddings.md) |
| 12 | RNNs for Language Tasks and Attention — classification, tagging, and neural machine translation | [slides](slides/lecture_week12.pdf) | [notes](lectures/week12-rnn-attention-nlp.md) |
| 13 | Transformers — self-attention, multi-head attention, encoder-decoder architecture | [slides](slides/lecture_week13.pdf) | [notes](lectures/week13-transformers.md) |
| 14 | Large Language Models (LLMs) | [slides](slides/lecture_week14.pdf) | [notes](lectures/week14-large-language-models.md) |
| 15 | Reinforcement Learning | [slides](slides/lecture_week15.pdf) | [notes](lectures/week15-reinforcement-learning.md) |

## Learning Outcomes

By the end of this course, students will be able to:

1. Explain the biological and mathematical foundations of artificial neural networks, including perceptrons, activation functions, and the backpropagation algorithm.
2. Diagnose and fix common training problems in deep networks, such as vanishing/exploding gradients, poor initialization, and overfitting, using techniques like batch normalization, dropout, and modern optimizers.
3. Design and implement convolutional neural networks for image classification, object detection, and semantic segmentation, and explain the architectural evolution from AlexNet to MobileNet.
4. Design and implement recurrent architectures (RNN, GRU, LSTM) for sequence modeling and understand the vanishing-gradient limitations that motivated gated units.
5. Explain and implement generative models, including autoencoders, variational autoencoders, and generative adversarial networks, and compare their trade-offs.
6. Represent text numerically using word embeddings (Word2Vec, GloVe) and apply RNN- and attention-based models to language tasks such as tagging and translation.
7. Explain the self-attention and multi-head attention mechanisms and describe the Transformer encoder-decoder architecture in detail.
8. Describe how large language models are built, scaled, and adapted, and discuss their capabilities and limitations.
9. Formulate problems as reinforcement learning tasks and describe core RL concepts such as reward, discounting, policies, and value functions.
10. Build, train, evaluate, and debug deep learning models end-to-end in Python using NumPy and PyTorch, following good experimental practice.

## Repository Structure

```
CS405-Deep-Learning-2026/
├── README.md            This file
├── LICENSE              Usage terms
├── slides/               Original lecture slide decks (PDF), one per week
├── lectures/             Week-by-week written explanations of every slide deck, in plain language
├── assignments/          Graded assignments with instructions and rubrics
├── labs/                 Hands-on, code-along lab handouts (PyTorch)
├── handouts/              Supplementary reference material (math primers, cheat sheets, reading lists)
└── quizzes/               Short practice/graded quizzes with answer keys
```

Each subfolder has its own `README.md` (or index) describing exactly what is inside it and how it maps to the weekly schedule.

## How to Download and Use This Repository

### First-time setup

Clone the repository to your machine:

```bash
git clone https://github.com/sajjadhussa1n/CS405-Deep-Learning-2026.git
cd CS405-Deep-Learning-2026
```

If you do not use git, you can also download the repository as a ZIP file from the green "Code" button on the GitHub page and extract it locally.

### Getting weekly updates

New slides, notes, labs, and assignments are added throughout the semester. To pull the latest material without losing any of your own local changes:

```bash
git pull origin main
```

If you have made local edits to a tracked file (for example, filled-in lab notebooks) and this causes a conflict, make a copy of your work under a different filename before pulling, or commit your changes to your own fork/branch.

### Working through the material each week

1. Read the corresponding entry in `lectures/` for a plain-language walkthrough of that week's concepts before or after attending the lecture.
2. Review the original `slides/` deck for the exact figures, equations, and definitions used in class.
3. Complete the matching `labs/` handout to get hands-on practice implementing the concepts.
4. Consult `handouts/` for supporting math background or tooling references as needed.
5. Attempt the relevant `quizzes/` to self-check understanding.
6. Submit `assignments/` by their posted deadlines.

## Technical Requirements

- Python 3.9+
- An editor/IDE: VS Code, PyCharm, Jupyter Notebook, or Google Colab (recommended for GPU access without local setup)
- Core libraries: NumPy, Pandas, Matplotlib, scikit-learn
- Deep learning framework: PyTorch (primary framework used in labs and assignments); familiarity with TensorFlow/Keras is a plus
- A GitHub account, if you intend to submit assignments via pull request or fork (check with the instructor for the exact submission workflow used this semester)

Recommended setup:

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install numpy pandas matplotlib scikit-learn torch torchvision jupyter
```

## Contributing / Reporting Issues

This repository is maintained by the course instructor. If you spot a typo, broken link, or an error in the notes, please open a GitHub issue or a pull request describing the fix.

## License

This repository is released for **educational and personal use**. You are free to use, copy, and adapt the material for learning purposes. Commercial use, redistribution for profit, or use in another institution's course without written permission from the instructor is prohibited. See [LICENSE](LICENSE) for full terms.

## Recommended Textbook and Reference

- *Deep Learning* by Ian Goodfellow, Yoshua Bengio, and Aaron Courville, 2016. (Recommended textbook)
- *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* by Aurélien Géron, 2022. (Reference)

## Acknowledgements

Course designed and delivered by **Dr. Sajjad Hussain**, Department of Electrical and Computer Engineering, SEECS, NUST. Companion repository for the Machine Learning course: [CS470-Machine-Learning-2025](https://github.com/sajjadhussa1n/CS470-Machine-Learning-2025).
