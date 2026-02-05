
import sys
import os

# Ensure we can import from src
sys.path.append(os.getcwd())

try:
    from src.model import predict_risk, train_model
    print("Import successful")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

text = "I feel very sad and anxious all the time."
print(f"Testing with text: '{text}'")

try:
    result = predict_risk(text)
    print("Result:", result)
except Exception as e:
    print(f"Error during prediction: {e}")
    import traceback
    traceback.print_exc()
