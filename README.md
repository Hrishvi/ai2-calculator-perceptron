# ai3-calculator-perceptron

A from-scratch Python calculator that trains separate single-neuron models for addition and subtraction using gradient descent. Includes training and testing scripts, along with saved model files for each operation.

---

## What the Model Is

This project uses a **single-layer perceptron** — the simplest possible neural network. Instead of hard-coding `a + b` or `a - b`, two separate models are trained from scratch to learn what addition and subtraction mean purely from examples.

Each model is just **three numbers**: `weight_1`, `weight_2`, and `bias`. The model learns these values through thousands of training examples.

---

## The Core Formula

Every prediction is computed as:

```
output = (input_1 × weight_1) + (input_2 × weight_2) + bias
```

This is a **single neuron**. Training finds the correct values for `weight_1`, `weight_2`, and `bias` so this formula produces the right answer.

---

## Normalization

Raw numbers (up to 100,000) are scaled down to the `[0, 1]` range before being fed to the model:

```python
normalized = x / 100000
```

This keeps the math stable during training. After the model predicts, the result is scaled back up.

---

## How Training Works

1. A random example is generated (e.g. `30000 + 45000 = 75000`)
2. The model makes a prediction using the current weights
3. The **error** is calculated: `error = target - prediction`
4. Weights are nudged in the direction that reduces the error:
```
weight_1 += learning_rate × error × input_1
weight_2 += learning_rate × error × input_2
bias     += learning_rate × error
```
5. This repeats for 1,000 examples per epoch until loss reaches near zero

This technique is called **gradient descent**.

---

## What the Model Learns

For **addition**, the weights converge toward:
- `weight_1 ≈ 1.0`
- `weight_2 ≈ 1.0`
- `bias ≈ 0.0`

Because `(x1 × 1.0) + (x2 × 1.0) + 0 = x1 + x2` — exactly addition.

For **subtraction**:
- `weight_1 ≈ 1.0`
- `weight_2 ≈ -1.0`
- `bias ≈ 0.0`

The model discovers this entirely on its own through training — it is never told what the weights should be.

---

## Requirements

- Python 3.x
- No external libraries — only the standard library (`random`, `pickle`, `os`, `sys`)
