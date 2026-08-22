# Week 5 — Advanced Computer Vision

*Companion notes for [`slides/lecture_week05.pdf`](../slides/lecture_week05.pdf)*

## Why this week matters

Weeks 3 and 4 built up CNNs as image *classifiers* — "what single label describes this whole image?" Real applications usually need more: find *every* object and *where* it is (object detection), or label *every single pixel* (semantic segmentation). This week builds both capabilities on top of the CNN backbones from Week 4, and along the way introduces two ideas — converting fully connected layers into convolutions, and transpose convolutions — that reappear throughout modern computer vision.

## 1. What is object detection?

Think of it as a "spot the object" game with two questions: **what is it?** (classification) and **where is it?** (localization). Formally, object detection = localization + classification: given an image, identify what classes of objects are present *and* where each one is, usually expressed as a bounding box `(x, y, w, h)`. The output for one detected object might look like "Dog: 0.95" with box coordinates `(120, 45, 200, 180)`. The key challenge that separates detection from plain classification is that an image can contain **multiple** objects of **different** classes at **different** locations, while a classifier is only built to produce one answer per image.

One natural way to adapt a classification network's output is to predict, for every candidate region, a vector:

```
Prediction = [ p_obj, c_1, c_2, ..., c_K, b_x, b_y, b_h, b_w ]
```

where `p_obj` is the probability that an object is present at all, `c_1...c_K` are class probabilities, and `b_x, b_y, b_h, b_w` describe the bounding box. For example, `[0.95, 0, 1, 0, 0.3, 0.4, 0.2, 0.1]` might decode to "a cat, located at (0.3, 0.4), with size (0.2, 0.1)."

## 2. Why you can't just reuse a plain classifier, and the sliding window fix

A network trained purely as an image classifier learns "one image → one prediction," assumes a fixed input size, and has no built-in way to localize or handle multiple objects — it will typically only report the single most dominant object and miss the rest. The **sliding window** approach converts the problem from "one image → one prediction" into "many crops → many predictions": take a small window, slide it systematically across the image, and at each position ask the classifier "is there an object here, and if so, what?" Since each individual window looks like the single-object images the classifier was trained on, this trick lets a plain classifier be reused for detection without retraining.

Two extra knobs matter here. **Window size and stride** control how finely the image is scanned. And because **objects come in different sizes**, a single window size cannot catch both small and large objects well — the fix is an **image pyramid**: resize the image to several different scales and run the sliding window at each scale, so that some scale will roughly match the size of any given object.

This works, but it is extremely expensive. For a 224×224 image with a small window and stride 2, you get roughly `(224×224)/(2×2) ≈ 12,544` window positions; multiplying by, say, 5 scales gives about **62,720 classifier forward passes for a single image** — far too slow for anything close to real time. This computational cost is exactly the motivation for the "FC-to-convolution" trick described next, and ultimately for architectures like YOLO.

## 3. Turning fully connected layers into convolutions

The naive sliding window approach re-runs the entire network from scratch at every window position, redoing a huge amount of duplicate computation on overlapping regions. The key realization that fixes this: a network's fully connected (FC) layers are the only thing forcing a **fixed input size** and a **single output** per forward pass — because a fully connected layer's weight matrix is tied to one specific flattened input length. If we rewrite those FC layers as convolutional layers instead, the network can process an image of *any* size in a single forward pass and naturally produce a **grid of predictions**, one per spatial region, instead of one prediction for the whole image.

Concretely, suppose a network's convolutional base produces a `7×7×512` feature map, which used to be flattened (`25,088` values) and fed into `FC1: 25,088 → 4096`, `FC2: 4096 → 4096`, and `FC3: 4096 → 1000`. Each of those FC layers can be re-expressed as a convolution that produces exactly the same output on a `7×7×512` input:

- **FC1 → Conv6:** a convolution with 4096 filters, each of size `7×7×512` (i.e., exactly as large as the current spatial map), stride 1, padding 0. On a `7×7` input this produces a `1×1×4096` output — mathematically identical to the original FC1.
- **FC2 → Conv7:** a `1×1×4096 → 4096` convolution (a pointwise/channel-mixing convolution, exactly the 1×1 convolution from Week 4), preserving the `1×1` spatial size.
- **FC3 → Conv8:** a `1×1×4096 → 1000` convolution, again preserving spatial size, giving `1×1×1000`.

On a `7×7` input, this reproduces the original single classification, as expected. But now feed in a **larger** image — say the same network's convolutional base produces a `14×14×512` feature map. Conv6 (with its `7×7` filter) now slides across that larger map and produces an `8×8×4096` output instead of `1×1×4096`; Conv7 and Conv8 preserve that `8×8` spatial grid, giving a final `8×8×1000` output — a spatial *heatmap* of class scores, where **each of the 64 spatial locations corresponds to one 7×7 window position in the original sliding-window scheme.** In other words, a single forward pass through the FC-to-Conv network computes *all* of the sliding-window classifications simultaneously, sharing computation across overlapping windows, running efficiently in parallel on a GPU, and being end-to-end differentiable with no special post-processing — this is the foundational trick behind modern, fast detectors like YOLO.

## 4. YOLO — You Only Look Once

The FC-to-Conv trick gives us a dense grid of class predictions but not bounding boxes, and it's still tied to one fixed window size. **YOLO's** key insight: why not have each grid cell directly predict bounding boxes *and* class probabilities together, in one shot?

YOLO divides the input image into an `S×S` grid (classically `S=7`). Each grid cell is made responsible for any object whose *center* falls inside it, and predicts `B` bounding boxes (typically `B=2`), each with 5 numbers — `(x, y, w, h, confidence)` — plus `C` class probabilities shared across the boxes in that cell (`C=20` for the PASCAL VOC dataset used in the original paper). The full output tensor has shape `S × S × (B×5 + C)`; with the classic settings, that's `7 × 7 × (2×5 + 20) = 7 × 7 × 30`.

Within one cell's prediction: `x, y` are the box center coordinates *relative to that grid cell* (values between 0 and 1), `w, h` are the box's width and height relative to the *whole image* (also 0 to 1), and `confidence = P(object) × IoU(prediction, ground truth)` — a single number that captures both "is there an object here" and "how good is this particular box." The final detection score for a class is `confidence × class probability`, and only predictions above some threshold are kept.

Training data is prepared to match this exact target shape: for each object in an image, find the grid cell containing its center, mark that cell's confidence as 1 and record the correct class and box; every other cell gets confidence 0. YOLO's loss function then combines three terms: a **coordinate loss** on `(x, y, w, h)` (using square roots on `w, h` so that errors on small boxes are penalized proportionally more than the same absolute error on large boxes), a **confidence loss** on the objectness score, and a **classification loss** on the class probabilities — combined into one end-to-end differentiable loss, with the coordinate term typically up-weighted (`λ_coord = 5`) to emphasize getting the boxes right.

## 5. Cleaning up predictions: IoU and Non-Max Suppression

**Intersection over Union (IoU)** is the standard way to measure how well a predicted box matches a ground-truth box: `IoU = (area of intersection) / (area of union)`. An `IoU > 0.5` is generally considered a good detection, `0.3–0.5` indicates poor localization, and below `0.3` is usually treated as a false positive.

Because a detector like YOLO produces many overlapping candidate boxes for the same real object, **Non-Max Suppression (NMS)** is used to keep only the best one. The algorithm: sort all detections by confidence score; select the box with the highest confidence and keep it; remove every remaining box whose IoU with the selected box exceeds some threshold (since a high IoU means it's almost certainly a duplicate detection of the same object); repeat with whatever boxes are left. It's analogous to picking the sharpest photo out of several near-duplicate shots and discarding the blurry ones.

## 6. Semantic segmentation

Object detection is *sparse* — only the pixels inside a bounding box are directly meaningful. **Semantic segmentation** goes further: it assigns a class label to **every single pixel** in the image (e.g., "cat," "dog," "background" for each pixel), producing a dense, pixel-accurate map rather than a handful of boxes.

The standard architecture pattern for this is an **encoder-decoder**, sometimes described as "zoom out, then zoom in." The **encoder** is a familiar CNN that gradually reduces spatial size while increasing feature depth — capturing the "what" (semantic context) at the cost of losing precise "where." The **decoder** does the reverse: it gradually increases spatial size back up while reducing feature depth, trying to recover the "where" — precise pixel locations — for the final segmentation map. This is analogous to reading a map: first take in the big picture, then zoom in for detail.

## 7. Transpose convolution — learnable upsampling

The decoder needs a way to *increase* spatial resolution, the opposite of what ordinary convolution and pooling do. Simple options like nearest-neighbor or bilinear interpolation are fixed, non-learnable upsampling rules. **Transpose convolution** (sometimes loosely called "deconvolution") instead makes upsampling a *learnable* operation.

The intuition: think of it as the reverse of a normal convolution. A normal convolution slides a kernel over a *large* input to produce a *smaller* output, where each output value is a weighted combination of several nearby input values. Transpose convolution instead takes each *individual* value from a small input, multiplies the entire kernel by that single scalar, and "stamps" that scaled kernel onto a larger output canvas at the corresponding position; when neighboring stamps overlap (because the stride is smaller than the kernel size), the overlapping values are simply **added together**. So one input pixel contributes to *multiple* output values — the network learns, through training, the best way to "paint" fine details back out from a compressed feature representation.

The output size follows a formula that mirrors, and inverts, ordinary convolution's output-size formula:

```
H_out = (H_in - 1) × stride + K
```

where `K` is the kernel size — each input pixel produces a `K`-sized patch, and consecutive patches are spaced apart by the stride, so the total output size grows with both the kernel size and how many "gaps" (`H_in - 1`) there are between input pixels.

## 8. U-Net — the breakthrough architecture for segmentation

A plain encoder-decoder has a real weakness: the encoder, by design, throws away precise spatial detail as it compresses the image down (that's exactly what makes it good at capturing "what" is in the image), which makes it hard for the decoder to later recover sharp, precise object boundaries purely from the compressed bottleneck representation.

**U-Net's** solution is to add **skip connections** directly between corresponding encoder and decoder stages (the same core idea as ResNet's skip connections, applied here for a different purpose): at each resolution level, the decoder's upsampled features are **concatenated** with the encoder's features from that same resolution, before the corresponding downsampling step discarded spatial precision. The intuition: the encoder knows *what* is in the image (high-level semantic features), while the decoder needs to know *where* things are (precise spatial detail) — skip connections directly carry that spatial "where" information from encoder to decoder, bypassing the lossy bottleneck.

Architecturally, U-Net has three parts. The **contracting path (encoder)** repeats blocks of two 3×3 convolutions + ReLU, followed by 2×2 max pooling, doubling the number of feature channels at each stage while shrinking spatial size — learning increasingly abstract features. The **bottleneck** applies further 3×3 convolutions at the most compressed resolution, capturing the most global context. The **expanding path (decoder)** repeats blocks of a 2×2 up-convolution (transpose convolution), concatenation with the corresponding encoder feature map via a skip connection, and then two more 3×3 convolutions + ReLU — progressively increasing spatial size and combining global context with the fine-grained spatial detail carried in from the encoder, giving precise localization in the final segmentation output.

U-Net was originally designed for medical image segmentation (tumor, organ, and cell segmentation), where it works remarkably well even with limited training data and produces sharp object boundaries — but the same architecture has since been applied far beyond medicine, including autonomous driving (road and lane segmentation, obstacle segmentation), satellite imagery (land cover classification, building/road extraction), and creative applications (background removal, image editing, style transfer).

## Key takeaways

Object detection extends classification with localization, and the practical breakthrough that made dense, real-time detection possible was recognizing that fully connected layers are just a special case of convolution with a fixed-size kernel — rewriting them as convolutions lets a network process images of any size in a single forward pass and naturally produce a spatial grid of predictions, the foundation YOLO builds on by having each grid cell directly predict boxes and class probabilities. IoU measures box quality, and Non-Max Suppression removes duplicate detections. Semantic segmentation asks for a label at every pixel rather than a handful of boxes, typically solved with an encoder-decoder architecture; transpose convolution provides the learnable upsampling the decoder needs, and U-Net's skip connections solve the "the encoder threw away precise location information" problem, becoming the standard architecture for dense pixel-level prediction across medicine, autonomous driving, remote sensing, and beyond.
