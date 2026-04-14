<div align="center">

# ai2-calculator-perceptron

**A neural network that learns arithmetic from scratch.**
*No libraries. No shortcuts. Just math.*

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white)
![ML](https://img.shields.io/badge/Type-Single--Layer%20Perceptron-FF6B35?style=flat-square)
![Libraries](https://img.shields.io/badge/External%20Libraries-None-22C55E?style=flat-square)

</div>

---

## The Model

This project trains two **single-layer perceptrons** — one for addition, one for subtraction — purely from examples. No formula is hard-coded. The model figures out the math on its own.

Each model is exactly **three learned numbers:**

| Parameter | Role |
|-----------|------|
| `weight_1` | How much input 1 influences the output |
| `weight_2` | How much input 2 influences the output |
| `bias` | A constant offset to fine-tune predictions |

---

## The Core Formula

Every prediction is a single line of math:

```
output = (input_1 × weight_1) + (input_2 × weight_2) + bias
```

> This is one neuron. That's the entire model.

---

## Normalization

Numbers up to `100,000` are scaled to the `[0.0 → 1.0]` range before training:

```python
normalized = x / 100_000   # scale down  →  feed to model
output     = result * 100_000   # scale back up  →  real answer
```

This prevents large numbers from destabilizing the weight updates during training.

---

## How Training Works

```
┌─────────────────────────────────────────────────────┐
│                  TRAINING LOOP                      │
│                                                     │
│  1. Generate random example  →  e.g. 30000 + 45000  │
│  2. Model predicts           →  weighted sum        │
│  3. Calculate error          →  target − prediction │
│  4. Nudge weights            →  gradient descent    │
│  5. Repeat 1000×             →  one epoch           │
│  6. Stop when loss ≈ 0       →  converged ✓         │
└─────────────────────────────────────────────────────┘
```

The weight update rule (gradient descent):

```
weight_1 += learning_rate × error × input_1
weight_2 += learning_rate × error × input_2
bias     += learning_rate × error
```

---

## What the Model Discovers

The model starts with random weights and converges to these on its own:

**Addition model**
```
weight_1 ≈  1.0
weight_2 ≈  1.0
bias     ≈  0.0
```
> Because `(x1 × 1) + (x2 × 1) + 0 = x1 + x2` ✓

**Subtraction model**
```
weight_1 ≈  1.0
weight_2 ≈ -1.0
bias     ≈  0.0
```
> Because `(x1 × 1) + (x2 × -1) + 0 = x1 - x2` ✓

The model is **never told** what these values should be. It learns them entirely through training.

---

## Requirements

- Python 3.x
- Zero external libraries — only `random` `pickle` `os` `sys`
