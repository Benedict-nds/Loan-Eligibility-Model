import os
import joblib
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

# Load model and scaler
model = None
scaler = None

try:
    model_path = os.path.join("Instances", "loan_model.h5")
    scaler_path = os.path.join("Instances", "scalar.pkl")

    model = load_model(model_path)
    print(f"Model loaded successfully from {model_path}")

    with open(scaler_path, 'rb') as f:
        scaler = joblib.load(f)
    print(f"Scaler loaded successfully from {scaler_path}")

except Exception as e:
    print(f"Failed to load model or scaler: {e}")

@app.route("/")
def main():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Check if model and scaler are loaded
        if model is None or scaler is None:
            print("Model or scaler not loaded correctly. Please check the server logs.")	
            return jsonify({"error": "Model or scaler not loaded correctly. Please check the server logs."}), 500

        # Parse input data from the request
        data = request.json
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        print(f"Received data: {data}")
        # Ensure input is in the correct format
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

        if None in input_features:
            return jsonify({"error": "Missing one or more input features"}), 400

        # Convert input to NumPy array and reshape for a single sample
        input_array = np.array(input_features).reshape(1, -1)

        # Scale input features
        scaled_input = scaler.transform(input_array)

        # Make prediction
        prediction_prob = model.predict(scaled_input)[0][0]  # Adjusted for TensorFlow/Keras output format
        prediction_class = int(prediction_prob > 0.5)  # Binary classification threshold

        # Return prediction result
        return jsonify({
            "prediction_probability": float(prediction_prob),
            "prediction_class": prediction_class
        })

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
