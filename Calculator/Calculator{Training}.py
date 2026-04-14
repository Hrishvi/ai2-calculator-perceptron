import random
import sys
import pickle
import os

MAX_VAL = 100000
MAX_INPUT = MAX_VAL // 2

def normalize(x):
    return x / MAX_VAL

def denormalize(x):
    return x * MAX_VAL

def generate_batch(size):
    """Generate random addition pairs up to 500 each (so sum stays ≤ 1000)"""
    batch = []
    for _ in range(size):
        a = random.randint(0, MAX_INPUT)
        b = random.randint(0, MAX_INPUT)
        batch.append([a, b, a - b])
    return batch

weight_1 = round(random.uniform(-1, 1), 2)
weight_2 = round(random.uniform(-1, 1), 2)
bias     = 0.0

learning_rate = 0.01

def save_model(filename, w1, w2, b):
    if not os.path.exists('Trained_Models'):
        os.makedirs('Trained_Models')
    filepath = os.path.join('Trained_Models', filename)
    with open(filepath, 'wb') as f:
        pickle.dump({'weight_1': w1, 'weight_2': w2, 'bias': b}, f)
    print(f"✓ Model saved to: {filepath}")

epoch = 0
print("Starting Training...")

while True:
    total_loss = 0
    batch = generate_batch(1000) 

    for data in batch:
        x1     = normalize(data[0])
        x2     = normalize(data[1])
        target = normalize(data[2])

        weighted_sum = (x1 * weight_1) + (x2 * weight_2) + bias

        error       = target - weighted_sum
        total_loss += error ** 2

        weight_1 += learning_rate * error * x1
        weight_2 += learning_rate * error * x2
        bias     += learning_rate * error

    avg_loss = total_loss / len(batch)
    epoch += 1

    if epoch % 500 == 0:
        print(f"Epoch {epoch} | Loss: {avg_loss:.6f}")

    if avg_loss < 0.0000000000000001 :
        print(f"\nConverged at epoch {epoch}! Loss: {avg_loss:.6f}")
        break

print(f"\nWeight 1: {weight_1:.4f}")
print(f"Weight 2: {weight_2:.4f}")
print(f"Bias:     {bias:.4f}")

what_to_do = input("\nTest[t], Save[s], or quit: ")

if what_to_do.lower() == "t":
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
    filename = input("Enter filename: ")
    save_model(filename, weight_1, weight_2, bias)
    sys.exit()

else:
    sys.exit()