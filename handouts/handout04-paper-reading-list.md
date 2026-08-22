# Handout — Paper Reading List

The lecture notes in this repository explain every topic in accessible, plain language, but nothing beats reading the original research once you have the conceptual foundation. This handout collects the primary source for each week's major topic, roughly in the order they'd be useful to read, organized by week. None of these are required reading for the course unless your instructor says otherwise — treat this as a curated "where to go deeper" list.

## Weeks 1–2: Foundations

- Rumelhart, D., Hinton, G., & Williams, R. (1986). *Learning representations by back-propagating errors.* Nature. — The paper that popularized backpropagation as a practical training method for multi-layer networks.
- Minsky, M., & Papert, S. (1969). *Perceptrons.* MIT Press. — The book whose proof of the perceptron's limitations (Week 1) triggered the first AI winter; worth understanding historically even if you only read summaries of it.
- Glorot, X., & Bengio, Y. (2010). *Understanding the difficulty of training deep feedforward neural networks.* AISTATS. — The original Xavier/Glorot initialization paper (Week 2).
- He, K., Zhang, X., Ren, S., & Sun, J. (2015). *Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification.* ICCV. — The He initialization paper (Week 2), from the same team behind ResNet.
- Ioffe, S., & Szegedy, C. (2015). *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift.* ICML. — The original BatchNorm paper (Week 2).
- Srivastava, N., et al. (2014). *Dropout: A Simple Way to Prevent Neural Networks from Overfitting.* JMLR. — The original Dropout paper (Week 2).
- Kingma, D., & Ba, J. (2015). *Adam: A Method for Stochastic Optimization.* ICLR. — The Adam optimizer paper (Week 2).

## Weeks 3–5: Computer Vision

- LeCun, Y., et al. (1998). *Gradient-Based Learning Applied to Document Recognition.* Proceedings of the IEEE. — The LeNet-5 paper (Week 3), the founding CNN architecture.
- Krizhevsky, A., Sutskever, I., & Hinton, G. (2012). *ImageNet Classification with Deep Convolutional Neural Networks.* NeurIPS. — The AlexNet paper (Week 4) that kicked off the modern deep learning era.
- Simonyan, K., & Zisserman, A. (2015). *Very Deep Convolutional Networks for Large-Scale Image Recognition.* ICLR. — The VGGNet paper (Week 4).
- Szegedy, C., et al. (2015). *Going Deeper with Convolutions.* CVPR. — The GoogLeNet/Inception paper (Week 4).
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition.* CVPR. — The ResNet paper (Week 4).
- Howard, A., et al. (2017). *MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications.* arXiv. — MobileNetV1 (Week 4).
- Sandler, M., et al. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks.* CVPR. — MobileNetV2 (Week 4).
- Redmon, J., et al. (2016). *You Only Look Once: Unified, Real-Time Object Detection.* CVPR. — The original YOLO paper (Week 5).
- Ronneberger, O., Fischer, P., & Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI. — The U-Net paper (Week 5).

## Weeks 6–7: Sequence Models

- Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory.* Neural Computation. — The original LSTM paper (Week 7), and one of the most-cited papers in the field.
- Cho, K., et al. (2014). *Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation.* EMNLP. — The original GRU paper (Week 6).
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*, Chapter 10. MIT Press. — A thorough textbook treatment of recurrent networks, freely available online.
- Olah, C. (2015). *Understanding LSTM Networks.* colah's blog. — Not a formal paper, but a widely recommended, excellent visual explanation of LSTM (Week 7).

## Weeks 9–10: Generative Models

- Kingma, D., & Welling, M. (2013). *Auto-Encoding Variational Bayes.* ICLR. — The original VAE paper (Week 9).
- Doersch, C. (2016). *Tutorial on Variational Autoencoders.* arXiv:1606.05908. — A gentler, tutorial-style companion to the original VAE paper.
- Goodfellow, I., et al. (2014). *Generative Adversarial Networks.* NeurIPS. — The original GAN paper (Week 10).
- Radford, A., Metz, L., & Chintala, S. (2015). *Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks.* ICLR. — The DCGAN paper (Week 10).
- Arjovsky, M., Chintala, S., & Bottou, L. (2017). *Wasserstein GAN.* ICML. — The WGAN paper (Week 10).
- Gulrajani, I., et al. (2017). *Improved Training of Wasserstein GANs.* NeurIPS. — The WGAN-GP paper (Week 10).
- Karras, T., Laine, S., & Aila, T. (2019). *A Style-Based Generator Architecture for Generative Adversarial Networks.* CVPR. — StyleGAN, a major successor architecture worth knowing about even though it's beyond this course's scope.

## Week 11: Word Embeddings

- Mikolov, T., et al. (2013). *Efficient Estimation of Word Representations in Vector Space.* arXiv. — The original Word2Vec paper.
- Mikolov, T., et al. (2013). *Distributed Representations of Words and Phrases and their Compositionality.* NeurIPS. — Introduces negative sampling for Word2Vec (Week 11).
- Pennington, J., Socher, R., & Manning, C. (2014). *GloVe: Global Vectors for Word Representation.* EMNLP. — The GloVe paper (Week 11).

## Weeks 12–13: Attention and Transformers

- Bahdanau, D., Cho, K., & Bengio, Y. (2015). *Neural Machine Translation by Jointly Learning to Align and Translate.* ICLR. — The paper that introduced attention for RNN-based sequence-to-sequence models (Week 12).
- Sutskever, I., Vinyals, O., & Le, Q. (2014). *Sequence to Sequence Learning with Neural Networks.* NeurIPS. — The RNN encoder-decoder architecture that attention was originally added to (Week 12).
- Vaswani, A., et al. (2017). *Attention Is All You Need.* NeurIPS. — The Transformer paper (Week 13). This is arguably the single most important paper for the second half of this course — read it directly if you read only one paper from this list.

## Week 14: Large Language Models

- Devlin, J., et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL. — The encoder-only architecture family (Week 14).
- Radford, A., et al. (2018/2019). *Improving Language Understanding by Generative Pre-Training* (GPT) and *Language Models are Unsupervised Multitask Learners* (GPT-2). OpenAI. — The decoder-only architecture family (Week 14).
- Brown, T., et al. (2020). *Language Models are Few-Shot Learners.* NeurIPS. — The GPT-3 paper, introducing large-scale few-shot/in-context learning (Week 14).
- Raffel, C., et al. (2020). *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer.* JMLR. — The T5 paper, an encoder-decoder LLM (Week 14).
- Hu, E., et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv. — The LoRA paper (Week 14), directly relevant to Assignment 5 and Lab 13.
- Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS. — The original RAG paper (Week 14).
- Ouyang, L., et al. (2022). *Training Language Models to Follow Instructions with Human Feedback.* NeurIPS. — The InstructGPT paper, introducing RLHF at scale (Week 14).

## Week 15: Reinforcement Learning

- Sutton, R., & Barto, A. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press. — The standard RL textbook; freely available online from the authors. Covers everything in Week 15 and far beyond.
- Mnih, V., et al. (2015). *Human-level control through deep reinforcement learning.* Nature. — The original Deep Q-Network (DQN) paper (Week 15), combining Q-learning with a neural network function approximator.

## How to read a paper efficiently

If you're new to reading research papers, a good default strategy: read the abstract and conclusion first to understand the claimed contribution, skim the figures (they usually summarize the architecture and main results), then read the introduction for context and related work, and only then work through the method and experiments sections in detail. It's normal, especially early in the course, not to understand every mathematical detail on a first pass — come back to a paper after you've covered the relevant week's lecture material, and it will make substantially more sense.
