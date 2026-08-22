# Assignment 2 — CNNs for Image Classification, Detection, and Segmentation

**Covers:** Week 3 (Introduction to CNNs), Week 4 (Deep CNN Architectures), Week 5 (Advanced Computer Vision)
**Deliverable:** PyTorch code + trained model checkpoint(s) + written report

## Learning objectives

Implement and train convolutional neural networks for image classification; compare the design trade-offs of classic architectures (AlexNet-style, VGG-style, ResNet-style, MobileNet-style); apply transfer learning from a pre-trained backbone; and get hands-on experience with at least one non-classification vision task (object detection or semantic segmentation).

## Part A — CNNs from the ground up

1. Implement a small CNN (2–4 convolutional layers + pooling + a small fully connected head) in PyTorch and train it on CIFAR-10 (or a dataset of similar scale approved by the instructor). Report your final test accuracy.
2. Implement a residual block (Week 4) from scratch (do not use `torchvision`'s built-in ResNet block) and add it to your network. Compare training curves and final accuracy with and without the residual connection, at a depth where the plain network starts to show the degradation problem described in Week 4.
3. Visualize the learned filters of your first convolutional layer, and visualize the feature maps produced by that layer for a handful of sample images. Briefly discuss what kinds of patterns the filters appear to have learned (edges, colors, textures, etc.), connecting this back to the biological-inspiration discussion in Week 3.

## Part B — Transfer learning

Using a CNN backbone pre-trained on ImageNet (e.g., ResNet-18/34 or MobileNetV2 from `torchvision.models`), fine-tune it on a smaller image classification dataset of your choice (something clearly different from ImageNet's typical categories — e.g., a specific domain like flowers, food, or medical images, subject to instructor approval and appropriate data licensing).

1. Follow the four transfer-learning scenarios described in Week 2/Week 4 (freeze everything except the classifier head; freeze early layers only; fine-tune the whole network) and report validation accuracy for each, along with training time. Discuss which scenario made sense for your dataset size and why.
2. Compare your fine-tuned model's accuracy against a model of the same architecture trained **from scratch** on the same (small) dataset. Discuss the result in terms of the Week 2 transfer-learning guidance.

## Part C — Choose one: Object Detection or Semantic Segmentation

**Option 1 — Object Detection.** Using a pre-trained detector (e.g., a lightweight YOLO variant or `torchvision`'s Faster R-CNN) on a small custom or provided dataset, run inference and visualize predicted bounding boxes with confidence scores. Implement Non-Max Suppression (Week 5) yourself from scratch (do not use a library's built-in NMS) and verify it produces the same results as the library implementation on at least 5 example images. Report the IoU-based precision at a threshold of your choice.

**Option 2 — Semantic Segmentation.** Implement a small U-Net (Week 5) from scratch (encoder, bottleneck, decoder, and skip connections — you may use library layers for the convolutions themselves, but wire up the U-Net structure yourself) and train it on a small segmentation dataset (e.g., the Oxford-IIIT Pet dataset's trimap annotations, or a dataset approved by the instructor). Report pixel accuracy and mean IoU on a held-out validation set, and include example predicted masks next to their ground truth.

## Report requirements

Include your CIFAR-10 training curves and final accuracy; your residual-vs-plain comparison plot and discussion; filter/feature-map visualizations; your transfer-learning results table across the different fine-tuning strategies; and your Part C results (bounding box visualizations + IoU/precision, or segmentation masks + pixel accuracy/mIoU, depending on which option you chose).

## Grading rubric

| Component | Weight |
|---|---|
| Part A: CNN implementation, training, and residual block comparison | 30% |
| Part A: filter/feature map visualization and discussion | 10% |
| Part B: transfer learning experiments and comparison to from-scratch training | 25% |
| Part C: detection or segmentation implementation and results | 25% |
| Report clarity | 10% |
