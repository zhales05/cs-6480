# Assignment 5: Explanation of Reasoning

## The Hypothesis

The assignment asks us to demonstrate that a two-layer neural network (one large
fully connected hidden layer + a 2-neuron output layer using the one-hot
strategy) can classify any two-dimensional partition in a finite region, even
when the decision boundary is not a straight line.

## Why I Chose Two Spirals

I chose the **two interleaving Archimedean spirals** dataset because it is one
of the most challenging 2D classification problems in machine learning:

1. **No linear separator exists.** A straight line cannot separate the two
   classes at all — the spirals wrap around each other multiple times.
2. **No simple curve suffices.** Even a circle or ellipse cannot separate the
   classes. The boundary must wind around the origin in a complex spiral pattern.
3. **Classic benchmark.** The two-spirals problem has been a standard test for
   neural network capacity since the 1980s. It was historically difficult for
   shallow networks and was one of the motivating examples for deeper
   architectures.

### Data Generation

Each spiral is an Archimedean spiral (r = theta) with Gaussian noise added:
- Class 0: points along r*cos(theta), r*sin(theta)
- Class 1: points along r*cos(theta + pi), r*sin(theta + pi) (same spiral
  rotated 180 degrees)

The angle theta ranges from 0.5 to 3*pi (about 1.5 full rotations), creating
two tightly interleaving spirals. Each class has 1000 training points and 1000
test points. Noise (sigma=0.3) is added to make the problem realistic without
making the spirals overlap too much.

## Network Architecture

Following the assignment requirements:

- **Input:** 2 neurons (x, y coordinates)
- **Hidden layer:** 512 neurons with ReLU activation
- **Output layer:** 2 neurons with softmax activation (one-hot strategy)

### Why This Works

The **Universal Approximation Theorem** tells us that a single hidden layer with
enough neurons can approximate any continuous function on a compact domain. Here
is how it applies to our problem:

1. **ReLU neurons create piecewise-linear regions.** Each of the 512 ReLU
   neurons in the hidden layer defines a half-plane (a linear boundary). The
   neuron outputs zero on one side and a positive linear function on the other.

2. **Combining 512 half-planes.** When 512 such neurons are combined, they can
   partition the 2D input space into a very large number of distinct
   piecewise-linear regions. With 512 neurons in 2D, the network can create up
   to O(512^2) = O(262,144) regions — far more than enough to approximate the
   spiral boundary.

3. **The softmax output layer.** The two output neurons with softmax convert the
   hidden layer's representation into a probability distribution over two
   classes. This is equivalent to drawing a single linear decision boundary in
   the 512-dimensional hidden space. The key insight is that while the spiral is
   not linearly separable in 2D, the hidden layer transforms the data into a
   512-dimensional space where it *is* linearly separable.

### Why 512 Neurons?

The spiral boundary is long and winding. Each section of the boundary requires
several ReLU neurons to approximate. With ~1.5 full rotations and a fine
boundary, 512 neurons provides ample capacity. A smaller layer (e.g., 32
neurons) would struggle to trace the full spiral, while 512 gives enough
piecewise-linear segments to closely follow the curved boundary.

## Training Details

- **Optimizer:** Adam (adaptive learning rate, works well for complex problems)
- **Loss function:** Sparse categorical cross-entropy (standard for multi-class
  classification with integer labels and softmax output)
- **Batch size:** 64 (small enough for gradient noise to help exploration, large
  enough for stable training)
- **Epochs:** 200 (enough for the model to converge on this problem)

## Results

- **Test accuracy: 99.8%** — The network correctly classifies nearly all test
  points despite the highly non-linear boundary.
- **Training converges smoothly** — Both loss and accuracy curves show stable
  convergence without significant overfitting.
- **Decision boundary visualization** confirms that the network learned the
  spiral shape, with the decision boundary closely following the true spiral
  structure.

## Conclusion

The results strongly support the hypothesis. A two-layer network with a
sufficiently large hidden layer (512 ReLU neurons) can learn even the most
challenging 2D classification boundaries. The two-spirals dataset is a
worst-case scenario for 2D classification, and the network achieves near-perfect
accuracy, demonstrating that this architecture is capable of classifying
arbitrary two-dimensional partitions in a finite region.
