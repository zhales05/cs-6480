# Assignment 7: Explanation of Reasoning

## The Task

Perform feature engineering on the random phantom images from A6. Normalize and
find the histogram of each image so that the data size is significantly reduced.
Find a method **without using neural networks or machine learning** that achieves
similar performance to A6's CNN (99.3% accuracy).

## Feature Engineering Strategy

The key question is: what measurable properties differ between 1-ellipse and
2-ellipse phantom images? I identified two categories of features:

### Intensity Histogram (16 values)

A normalized 16-bin histogram of pixel intensities captures the pixel value
distribution. A single ellipse produces a concentrated histogram (most pixels
are either 0 or 1.0), while two ellipses produce a broader distribution with
mid-range values.

### Connected Components (2 values)

Using `scipy.ndimage.label` on thresholded images at two levels (0.1 and 0.2),
I count the number of distinct bright regions. When two ellipses don't overlap,
this directly detects two separate blobs.

### Bright Pixel Fraction (1 value)

The fraction of pixels above threshold 0.1. Two ellipses cover more total area.

### Mean of Non-Zero Pixels (1 value)

The mean intensity of pixels above 0.01. This turned out to be the single most
powerful feature.

**Total: 21 features** (down from 784 pixels = **37x reduction**).

## The Key Insight: Non-Zero Pixel Mean

The `generate_phantom` function normalizes each image by dividing by its maximum
pixel value: `P = P / P.max()`. This has a crucial consequence:

- **1 ellipse:** The ellipse has uniform intensity I everywhere inside it. After
  normalization by max (which is I), ALL non-zero pixels become exactly 1.0.
  Therefore, the mean of non-zero pixels is exactly **1.0**.

- **2 ellipses:** The two ellipses have different random intensities I1 and I2.
  After normalization by max(all pixels), the lower-intensity ellipse has values
  < 1.0. Therefore, the mean of non-zero pixels is strictly **< 1.0**.

This property holds regardless of ellipse size, position, or orientation. The
only edge case is two ellipses with near-identical intensities, which is
extremely rare with continuous random sampling.

## Rule-Based Classifier (No ML)

Instead of training any model, I use a simple if/else decision rule:

```
if connected_components >= 2:
    predict 2 ellipses          # clearly separated
elif mean_of_nonzero_pixels < 0.99:
    predict 2 ellipses          # different intensities detected
else:
    predict 1 ellipse
```

This is a hand-crafted rule — no training, no fitting, no optimization, no
machine learning of any kind. The thresholds (0.99) are derived from the
mathematical properties of the phantom generator, not from data.

## Data Reduction

- **Raw image:** 784 values per sample (28 x 28 pixels)
- **Features used by classifier:** effectively 3 values (connected components
  at two thresholds + non-zero mean), though the full feature vector is 21
- **Compression ratio:** 37x (or 261x if counting only the features the
  decision rule actually examines)

## Results

| Metric | A6 (CNN) | A7 (Rule-Based) |
|--------|----------|-----------------|
| Input size per sample | 784 pixels | 21 features |
| Data reduction | 1x | 37x |
| Method | CNN (Conv2D + Dense) | Hand-crafted if/else rules |
| Uses ML/training? | Yes (neural network) | No |
| Test accuracy | 99.3% | **99.8%** |

The rule-based classifier achieved **99.8% test accuracy** (998/1000 correct,
only 2 misclassifications), surpassing the CNN's 99.3%.

## Why This Works So Well

1. **The normalization step is the key.** By dividing by the maximum pixel
   value, the phantom generator inadvertently creates a nearly perfect
   discriminative feature: single-ellipse images have uniform non-zero pixels
   (all equal to 1.0), while multi-ellipse images do not.

2. **Connected components handle the easy cases.** When ellipses don't overlap,
   simple blob counting at a threshold directly solves the problem.

3. **The non-zero mean handles the hard cases.** Even when two ellipses
   completely overlap (appearing as a single blob), their different intensities
   cause the mean to drop below 1.0 after normalization.

4. **The 2 misclassifications** are likely cases where two ellipses happen to
   have nearly identical intensities AND overlap completely, making them
   indistinguishable from a single ellipse by any method.

## Conclusion

A purely rule-based approach using domain knowledge about the phantom generator
surpasses CNN performance on this task, with no machine learning, no training,
and 37x less data per sample. This demonstrates that understanding the
mathematical properties of your data can be more powerful than any model.
