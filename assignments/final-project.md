# Final Project

**Deliverable:** Project proposal (short), final code repository, final written report, and a short presentation/demo (format to be confirmed by the instructor)

## Overview

The final project is an open-ended, semester-capstone opportunity to apply the material from across CS-405 to a problem of your own choosing, working individually or in small groups (group size to be confirmed by the instructor). It should go meaningfully beyond any single assignment — either by tackling a more substantial real-world dataset/problem, by combining ideas from multiple weeks (e.g., a CNN feature extractor feeding into a Transformer-based sequence model, or a generative model used for data augmentation ahead of a classification task), or by exploring a topic touched on lightly in lecture in more depth (e.g., Week 15's introduction to Reinforcement Learning, object detection frameworks like YOLO beyond what's in Assignment 2, or a topic in current deep learning research).

## Suggested project tracks

You are not limited to these — any project connecting clearly to the course material and approved by the instructor is acceptable — but if you'd like a starting point:

- **Applied vision:** train a strong image classifier, detector, or segmentation model on a real dataset relevant to a domain you care about (medical imaging, satellite imagery, wildlife monitoring, manufacturing defect detection, etc.), building on Weeks 3–5.
- **Applied NLP:** build a more substantial NLP application — a text summarizer, a chatbot fine-tuned for a specific domain, a document classification or information-extraction system — building on Weeks 6–7 and 11–14.
- **Generative modeling:** go deeper into VAEs or GANs from Weeks 9–10 — for example, implement a conditional GAN for a specific image-to-image translation task, or explore a more advanced VAE/GAN variant not covered in lecture.
- **Reinforcement learning:** extend Week 15's introduction into a working Deep Q-Network (or a policy-gradient method of your choosing) trained on a standard RL benchmark environment (e.g., an OpenAI Gym/Gymnasium environment such as CartPole or a simple grid world of your own design), including a from-scratch implementation of the Bellman-equation-based training update.
- **Reproduction/extension:** carefully reproduce the key results of a published deep learning paper relevant to the course (with instructor approval), and extend it in some small but genuine way (a new dataset, an ablation the original paper didn't run, an architectural variant).

## Milestones

1. **Proposal (1–2 pages):** problem statement, why it's interesting, the dataset(s) you plan to use (with a link/source and a note on licensing/availability), your planned approach and which course concepts it builds on, and a rough timeline.
2. **Progress check-in:** a brief update (format TBD by instructor — could be a short written update or an in-class check-in) showing initial results, obstacles encountered, and any changes to your plan.
3. **Final submission:** code repository, trained model artifacts (or clear instructions to reproduce training), and a final written report.
4. **Presentation/demo:** a short presentation (format and length TBD by the instructor) showing your problem, approach, and results, ideally with a live or recorded demo.

## Final report expectations

Your final report should read like a small technical paper or a well-documented project write-up, and should include: a clear problem statement and motivation; a description of your dataset(s), including any preprocessing; a description of your method/architecture, explicitly connecting it to the relevant course concepts (cite lecture weeks where relevant); your experimental setup (hyperparameters, training details, hardware used, training time); your results, with appropriate plots/tables and comparisons against at least one reasonable baseline; and a discussion of limitations, failure cases, and what you would try next with more time.

## Grading rubric (indicative — instructor may adjust)

| Component | Weight |
|---|---|
| Proposal quality and feasibility | 10% |
| Technical depth and correct application of course concepts | 30% |
| Experimental rigor (baselines, ablations, honest reporting of failures) | 25% |
| Final report quality and clarity | 20% |
| Presentation/demo | 15% |
