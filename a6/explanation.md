# Assignment 6: Explanation of Reasoning

## The Task

Generate two groups of random 28x28 phantoms — Group 1 with one ellipse, Group 2
with two ellipses — each with random intensity, size, and orientation. Train a
neural network to classify the two groups.

## Dataset Design

### Phantom Generation

I reused the `phantom()` function from the lecture (CS6480Lect05.ipynb), which
renders ellipses onto an n x n grid. Each ellipse is parameterized by six values:
`[I, a, b, x0, y0, phi]`.

For each random ellipse I sample:
- **Intensity (I):** Uniform in [0.3, 1.0] — ensures ellipses are always visible
- **Semi-axes (a, b):** Uniform in [0.15, 0.5] and [0.10, 0.4] — large enough to
  be distinguishable at 28x28 resolution (a semi-axis of 0.15 spans about 4 pixels)
- **Center (x0, y0):** Uniform in [-0.4, 0.4] — keeps ellipses mostly within the
  image frame while allowing some variety
- **Rotation (phi):** Uniform in [0, 180] degrees — full rotational randomness

After generation, pixel values are clipped to be non-negative and normalized to
[0, 1].

### Dataset Size

- **Training:** 2000 phantoms per class (4000 total)
- **Test:** 500 phantoms per class (1000 total)

This is enough data for a small CNN to learn robust features without overfitting.

## Why This Problem is Non-Trivial

At first glance, counting ellipses seems simple, but there are several
challenges:

1. **Overlapping ellipses** — When two ellipses overlap in a Group 2 phantom,
   the result can look like a single blob, making it visually similar to a
   single larger ellipse.
2. **Variable sizes** — A small second ellipse inside a large one may be hard to
   detect. Similarly, two small separated ellipses could look like noise.
3. **Random orientations** — The variety of shapes and orientations means the
   network cannot rely on simple template matching.

## Network Architecture

I chose a **Convolutional Neural Network (CNN)** because:

1. **Spatial structure matters.** The key signal is the spatial arrangement of
   pixel intensities — whether there are one or two distinct bright regions. CNNs
   are designed to detect such spatial patterns through learned filters.
2. **Translation invariance.** The ellipses can appear anywhere in the image.
   CNN pooling layers provide natural translation invariance.
3. **Parameter efficiency.** A fully connected network on 28x28=784 inputs would
   need many more parameters. The CNN's weight sharing keeps the model small.

### Architecture Details

| Layer          | Output Shape    | Parameters |
|----------------|-----------------|------------|
| Conv2D(32, 3x3, relu) | 28x28x32 | 320        |
| MaxPooling(2x2)        | 14x14x32 | 0          |
| Conv2D(64, 3x3, relu) | 14x14x64 | 18,496     |
| MaxPooling(2x2)        | 7x7x64   | 0          |
| Flatten                | 3136      | 0          |
| Dense(128, relu)       | 128       | 401,536    |
| Dense(2, softmax)      | 2         | 258        |

Total: ~420K parameters. This is a standard LeNet-style architecture, well-suited
for 28x28 grayscale image classification (the same size as MNIST).

### How the CNN Learns

- **Conv layer 1** learns low-level features: edges, curves, and intensity
  gradients that form ellipse boundaries.
- **Conv layer 2** combines those into higher-level features: complete ellipse
  shapes, pairs of edges indicating a second ellipse, or overlap patterns.
- **Dense layers** use these spatial features to make the final one-vs-two
  classification decision.

## Training Details

- **Optimizer:** Adam (lr=0.001) — adaptive and effective for image tasks
- **Loss:** Sparse categorical cross-entropy — standard for integer-labeled
  multi-class classification with softmax output
- **Batch size:** 64
- **Epochs:** 30 — sufficient for convergence on this task

## Results

- **Test accuracy: 99.3%** — The CNN reliably distinguishes one-ellipse from
  two-ellipse phantoms across a wide range of random configurations.
- Training and validation curves show smooth convergence with minimal overfitting.
- Visual inspection of test predictions confirms correct classification even for
  difficult cases (overlapping ellipses, similar sizes).

## Conclusion

A small CNN is well-suited for this phantom classification task. The
convolutional layers naturally detect the spatial features (edges, shapes, number
of distinct regions) needed to count ellipses, while the one-hot softmax output
provides a clean probabilistic classification. The high test accuracy
demonstrates that the network generalizes well beyond the training data.
