import random
import sys
import pickle
import os

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

# The maximum value used for normalization.
# All inputs and outputs are scaled relative to this number
# so the model works with small decimal values (0.0 to 1.0) instead of large integers.
MAX_VAL = 100000

# Each input number is capped at half of MAX_VAL.
# For addition: ensures the sum never exceeds MAX_VAL (e.g. 50000 + 50000 = 100000).
# For subtraction: result stays within [-MAX_VAL/2, MAX_VAL/2].
MAX_INPUT = MAX_VAL // 2


# ─────────────────────────────────────────────
# NORMALIZATION HELPERS
# ─────────────────────────────────────────────

def normalize(x):
    """Scale a raw number down to the [0, 1] range for the model."""
    return x / MAX_VAL

def denormalize(x):
    """Convert a model output back to the original number range."""
    return x * MAX_VAL


# ─────────────────────────────────────────────
# DATA GENERATION
# ─────────────────────────────────────────────

def generate_batch(size):
    """
    Generate a batch of random training examples.
    
    Each example is [input_a, input_b, target_output].
    
    ⚠️  NOTE: The docstring says 'addition pairs' but the target is (a - b).
         Change to (a + b) when training the Addition model.
    
    Args:
        size (int): Number of examples to generate.
    
    Returns:
        list of [int, int, int]: Training pairs with their expected output.
    """
    batch = []
    for _ in range(size):
        a = random.randint(0, MAX_INPUT)
        b = random.randint(0, MAX_INPUT)
        batch.append([a, b, a - b])   # ← Change to (a + b) for addition training
    return batch


# ─────────────────────────────────────────────
# MODEL PARAMETERS (initialized randomly)
# ─────────────────────────────────────────────

# weight_1 and weight_2 are what the model learns.
# They represent how much each input number contributes to the output.
# Starting random helps avoid a biased starting point.
weight_1 = round(random.uniform(-1, 1), 2)
weight_2 = round(random.uniform(-1, 1), 2)

# Bias shifts the output up or down independent of the inputs.
# Starts at 0 — the model will adjust it during training.
bias = 0.0

# How fast the model updates its weights each step.
# Too high = overshoots; too low = trains slowly.
learning_rate = 0.01


# ─────────────────────────────────────────────
# SAVE MODEL
# ─────────────────────────────────────────────

def save_model(filename, w1, w2, b):
    """
    Save trained weights and bias to a file in the Trained_Models folder.
    
    Uses Python's pickle format to serialize the model dictionary.
    Creates the Trained_Models directory if it doesn't exist yet.
    
    Args:
        filename (str): Name of the file to save (e.g. "Addition", "Subtraction").
        w1 (float): Trained weight for the first input.
        w2 (float): Trained weight for the second input.
        b  (float): Trained bias value.
    """
    if not os.path.exists('Trained_Models'):
        os.makedirs('Trained_Models')
    filepath = os.path.join('Trained_Models', filename)
    with open(filepath, 'wb') as f:
        pickle.dump({'weight_1': w1, 'weight_2': w2, 'bias': b}, f)
    print(f"✓ Model saved to: {filepath}")


# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────

epoch = 0
print("Starting Training...")

while True:
    total_loss = 0

    # Generate 1000 fresh random examples each epoch.
    # Using new data every epoch helps the model generalize better.
    batch = generate_batch(1000)

    for data in batch:
        # Normalize inputs and target so all values are small decimals
        x1     = normalize(data[0])
        x2     = normalize(data[1])
        target = normalize(data[2])

        # FORWARD PASS: compute the model's prediction
        # This is a single neuron: output = (w1 * x1) + (w2 * x2) + bias
        weighted_sum = (x1 * weight_1) + (x2 * weight_2) + bias

        # Calculate how wrong the prediction was
        error = target - weighted_sum

        # Accumulate squared error for loss tracking
        total_loss += error ** 2

        # BACKWARD PASS (Gradient Descent):
        # Nudge each weight in the direction that reduces the error.
        # The update is proportional to the input (x1, x2) and the error.
        weight_1 += learning_rate * error * x1
        weight_2 += learning_rate * error * x2
        bias     += learning_rate * error   # Bias has no associated input, so just use error

    # Average loss across the batch — lower is better
    avg_loss = total_loss / len(batch)
    epoch += 1

    # Print progress every 500 epochs so you can track convergence
    if epoch % 500 == 0:
        print(f"Epoch {epoch} | Loss: {avg_loss:.6f}")

    # Stop training when loss is essentially zero (model has converged)
    if avg_loss < 0.0000000000000001:
        print(f"\nConverged at epoch {epoch}! Loss: {avg_loss:.6f}")
        break


# ─────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────

print(f"\nWeight 1: {weight_1:.4f}")
print(f"Weight 2: {weight_2:.4f}")
print(f"Bias:     {bias:.4f}")

what_to_do = input("\nTest[t], Save[s], or quit: ")

if what_to_do.lower() == "t":
    # Manually test the trained model with custom inputs
    while True:
        go = input("Test again? [y/n]: ")
        if go.lower() != "y":
            break
        input1 = int(input(f"First number (0-{MAX_INPUT}): "))
        input2 = int(input(f"Second number (0-{MAX_INPUT}): "))
        weighted_sum = (normalize(input1) * weight_1) + (normalize(input2) * weight_2) + bias
        output = round(denormalize(weighted_sum))
        print(f"Answer: {output}")
    sys.exit()

elif what_to_do.lower() == "s":
    # Save the trained weights so the testing script can load them later
    filename = input("Enter filename: ")
    save_model(filename, weight_1, weight_2, bias)
    sys.exit()

else:
    sys.exit()
