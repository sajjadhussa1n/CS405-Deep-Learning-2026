# Lab 5 — Object Detection and Segmentation

**Matches:** [Week 5 — Advanced Computer Vision](../lectures/week05-advanced-computer-vision.md)
**Goal:** Implement IoU and Non-Max Suppression from scratch, run a pre-trained object detector, and build a minimal transpose-convolution decoder.

## Setup

```bash
pip install torch torchvision matplotlib pillow
```

## Step 1 — IoU from scratch

```python
def iou(box_a, box_b):
    # boxes as (x1, y1, x2, y2)
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1, inter_y1 = max(xa1, xb1), max(ya1, yb1)
    inter_x2, inter_y2 = min(xa2, xb2), min(ya2, yb2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0
```

Test it on a few hand-designed pairs of boxes with known overlap and confirm your IoU values are sensible (identical boxes → 1.0, non-overlapping boxes → 0.0).

## Step 2 — Non-Max Suppression from scratch

```python
def nms(boxes, scores, iou_threshold=0.5):
    order = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
    keep = []
    while order:
        current = order.pop(0)
        keep.append(current)
        order = [i for i in order if iou(boxes[current], boxes[i]) <= iou_threshold]
    return keep
```

Construct a synthetic example with 5–6 overlapping boxes (some near-duplicates of the same object, some genuinely separate objects) and confirm your `nms` keeps one box per real object.

## Step 3 — Run a pre-trained detector

```python
import torchvision

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
model.eval()
```

Run the model on a handful of your own images (resize to a reasonable size, convert to a `[0,1]` float tensor), and draw the returned bounding boxes and labels using `torchvision.utils.draw_bounding_boxes`. Compare the library's built-in NMS output against your own `nms()` applied to the *raw* (pre-NMS) boxes/scores if the model API exposes them — otherwise, apply your `nms()` to boxes you construct yourself from a lower-level detection head, and confirm the results match.

## Step 4 — Transpose convolution, by formula and by code

Confirm the transpose-convolution output-size formula `H_out = (H_in - 1) × stride + K` from the lecture by constructing `nn.ConvTranspose2d` layers with a few different `(stride, kernel_size)` combinations and checking the output shape against the formula's prediction, e.g.:

```python
import torch.nn as nn

up = nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=3, stride=2)
x = torch.randn(1, 8, 4, 4)
y = up(x)
print(y.shape)  # compare against (4-1)*2 + 3 = 9
```

## Step 5 — A minimal U-Net skeleton

Implement a tiny U-Net (2 downsampling stages, 1 bottleneck, 2 upsampling stages with skip connections) and confirm it runs on a batch of dummy images, producing an output the same spatial size as the input:

```python
class TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU())
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.ReLU())
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU())
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.ReLU())  # 64 = 32(skip) + 32(up)
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(32, 16, 3, padding=1), nn.ReLU())  # 32 = 16(skip) + 16(up)
        self.out = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)
```

Verify: `TinyUNet()(torch.randn(2, 3, 64, 64)).shape` should be `(2, 1, 64, 64)`.

## Checkpoint questions

1. In Step 2, what happens to the number of surviving boxes if you set `iou_threshold` very low (e.g., 0.1) versus very high (e.g., 0.9)? Which setting risks merging genuinely separate objects, and which risks keeping duplicate detections?
2. In Step 5, why does the skip connection concatenate the encoder feature map with the *upsampled* decoder feature map (rather than, say, replacing it)? What would you lose if you removed the skip connections entirely?
3. Trace through the channel counts in `TinyUNet.forward` and confirm the `torch.cat` dimensions in `dec2`/`dec1` are consistent with the layer definitions above — why must they match?
