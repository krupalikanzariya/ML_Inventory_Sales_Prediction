from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import holidays
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Load the trained model safely
MODEL_PATH = "xgboost_model.pkl"
try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route("/")
def home():
    return render_template("index.html", prediction=None, data={})

@app.route("/predict", methods=["POST", "GET", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return "", 204  # Handle preflight request for CORS

    if request.method == "GET":
        return jsonify({"message": "Send data via POST"}), 400

    try:
        # Debugging: Print received form data
        print("Received Form Data:", request.form)

        # Extract values
        date_str = request.form.get("date", "").strip()
        store = request.form.get("store", "").strip()
        item = request.form.get("item", "").strip()

        if not date_str or not store or not item:
            return render_template("index.html", prediction=None, data=request.form, error="All fields are required.")

        store = int(store)
        item = int(item)

        # Convert date
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        month, day, weekday = date_obj.month, date_obj.day, date_obj.weekday()

        # Compute cyclic features
        m1 = np.sin(2 * np.pi * (month - 1) / 12)
        m2 = np.cos(2 * np.pi * (month - 1) / 12)

        # Check for holiday
        india_holidays = holidays.country_holidays("IN")
        holiday_flag = 1 if date_obj in india_holidays else 0

        year = date_obj.year

        # Prepare input DataFrame
        input_data = pd.DataFrame(
            [[store, item, month, day, holiday_flag, m1, m2, weekday, year]],
            columns=["store", "item", "month", "day", "holidays", "m1", "m2", "weekday", "year"]
        )

        # Convert DataFrame to numeric types
        input_data = input_data.astype(float)

        # Ensure the model is loaded
        if model is None:
            return render_template("index.html", prediction=None, data=request.form, error="Model not loaded!")

        # Make prediction
        prediction = model.predict(input_data)[0]

        # Debugging: Print prediction in terminal
        print("Prediction:", prediction)

        # Return prediction to the frontend
        return render_template("index.html", prediction=prediction, data=request.form)

    except Exception as e:
        print(f"Error in prediction: {e}")
        return render_template("index.html", prediction=None, data=request.form, error=f"Error: {e}")

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
