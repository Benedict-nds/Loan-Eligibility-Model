import os
import joblib
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

# Load the model and scaler
model = None
scaler = None

try:
    # Paths to the saved model and scaler
    model_path = os.path.join("Instances", "loan_model.h5")
    scaler_path = os.path.join("Instances", "scalar.pkl")
    
    # Load the model
    model = load_model(model_path)
    print(f"Model loaded successfully from {model_path}")

    # Load the scaler
    scaler = joblib.load(scaler_path)
    print(f"Scaler loaded successfully from {scaler_path}")

except Exception as e:
    print(f"Failed to load model or scaler: {e}")

@app.route("/")
def main():
    """Render the main HTML page."""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Ensure the model and scaler are loaded
        if model is None or scaler is None:
            return jsonify({"error": "Model or scaler not loaded correctly. Check the server logs."}), 500

        # Parse input JSON data
        data = request.json
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        print(f"Received input data: {data}")

        # Extract features from the input data
        input_features = [
            data.get("loan_amount"),
            data.get("rate_of_interest"),
            data.get("Interest_rate_spread"),
            data.get("Upfront_charges"),
            data.get("term_in_months"),
            data.get("property_value"),
            data.get("income"),
            data.get("Credit_Score")
        ]

        # Ensure all features are provided
        if None in input_features:
            return jsonify({"error": "Missing one or more input features"}), 400

        # Convert input features to a NumPy array and reshape for a single sample
        input_array = np.array(input_features).reshape(1, -1)

        # Scale the input features using the loaded scaler
        scaled_input = scaler.transform(input_array)

        # Make a prediction using the loaded model
        prediction_prob = model.predict(scaled_input)[0][0]  # Single prediction
        prediction_class = int(prediction_prob > 0.5)  # Apply threshold for binary classification

        # Return the prediction as JSON
        return jsonify({
            "prediction_probability": float(prediction_prob),
            "prediction_class": prediction_class
        })

    except Exception as e:
        return jsonify({"error": f"An error occurred during prediction: {str(e)}"}), 500

if __name__ == "__main__":
    # Ensure Flask runs in the correct working directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    app.run(debug=True)
