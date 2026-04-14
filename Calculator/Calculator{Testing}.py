import pickle
import sys
import os


def load_model(filename):
    # Automatically look in the Trained_Models folder
    filepath = os.path.join('Trained_Models', filename)
    
    try:
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        print(f"✓ Model loaded from: {filepath}")
        return model_data['weight_1'], model_data['weight_2'], model_data['bias']
    except FileNotFoundError:
        print(f"Error: Could not find '{filename}' in the Trained_Models folder.")
        sys.exit()

max_value = 100000
def normalize(x):
    return x / max_value

def denormalize(x):
    return x * max_value
    
    

while True:
    question = input("What is the Addition or Subtraction you want to do?[type anything else to quit] ")
    continue_program = True

    if "+" in question :
        weight_1, weight_2, bias = load_model("Addition")
        input_1, input_2 = question.split("+")

            
    elif "-" in question:
        weight_1, weight_2, bias = load_model("Subtraction")
        input_1, input_2 = question.split("-")

    else:
            print("Model Quitting")
            continue_program = False
            
    if continue_program == True :
        input_1 = int(input_1.strip())
        input_2 = int(input_2.strip())
        weighted_sum = (normalize(input_1) * weight_1) + (normalize(input_2) * weight_2) + bias
        output = round(denormalize(weighted_sum))
        print(f"Output: {output}\n")
        
    else:
        break
sys.exit()

