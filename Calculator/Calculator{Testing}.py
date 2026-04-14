import pickle
import sys
import os


# ─────────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────────

def load_model(filename):
    """
    Load a trained model's weights and bias from the Trained_Models folder.
    
    The model is stored as a pickle file — a serialized Python dictionary
    containing weight_1, weight_2, and bias.
    
    Args:
        filename (str): Name of the saved model file (e.g. "Addition", "Subtraction").
    
    Returns:
        tuple: (weight_1, weight_2, bias) — the three values the perceptron uses to predict.
    
    Exits:
        If the file doesn't exist, prints an error and terminates the program.
    """
    filepath = os.path.join('Trained_Models', filename)

    try:
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        print(f"✓ Model loaded from: {filepath}")
        return model_data['weight_1'], model_data['weight_2'], model_data['bias']
    except FileNotFoundError:
        print(f"Error: Could not find '{filename}' in the Trained_Models folder.")
        sys.exit()


# ─────────────────────────────────────────────
# NORMALIZATION HELPERS
# ─────────────────────────────────────────────

# Must match the MAX_VAL used during training.
# Normalization scales inputs to small decimals so the model can process them correctly.
max_value = 100000

def normalize(x):
    """Scale a raw number down to the [0, 1] range."""
    return x / max_value

def denormalize(x):
    """Convert a model's normalized output back to a real number."""
    return x * max_value


# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────

while True:
    question = input("What is the Addition or Subtraction you want to do? [type anything else to quit] ")
    continue_program = True

    if "+" in question:
        # Load the Addition model and split the input on the '+' symbol
        weight_1, weight_2, bias = load_model("Addition")
        input_1, input_2 = question.split("+")

    elif "-" in question:
        # Load the Subtraction model and split the input on the '-' symbol
        weight_1, weight_2, bias = load_model("Subtraction")
        input_1, input_2 = question.split("-")

    else:
        # If no valid operator found, exit the loop
        print("Model Quitting")
        continue_program = False

    if continue_program:
        # Strip whitespace and convert strings to integers
        input_1 = int(input_1.strip())
        input_2 = int(input_2.strip())

        # FORWARD PASS:
        # Normalize both inputs, multiply by their learned weights, add bias.
        # This is the core computation of the perceptron.
        weighted_sum = (normalize(input_1) * weight_1) + (normalize(input_2) * weight_2) + bias

        # Denormalize to convert the output back to a real number, then round to nearest integer
        output = round(denormalize(weighted_sum))
        print(f"Output: {output}\n")

    else:
        break

sys.exit()
