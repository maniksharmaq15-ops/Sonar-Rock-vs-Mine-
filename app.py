from flask import Flask, render_template, request
import joblib
import numpy as np
import traceback
import os

app = Flask(__name__)

# Load model with detailed error logging
try:
    print("Current directory:", os.getcwd())
    print("Files present:", os.listdir("."))
    model = joblib.load("model.pkl")
    print("Model loaded successfully!")
except Exception as e:
    print("ERROR LOADING MODEL:")
    traceback.print_exc()
    model = None

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template("index.html", prediction_text="Error: Model not loaded.")
    try:
        values = request.form["values"]
        input_data = [float(x.strip()) for x in values.split(",")]
        if len(input_data) != 60:
            return render_template("index.html", prediction_text="Please enter exactly 60 values.")
        input_array = np.array(input_data).reshape(1, -1)
        prediction = model.predict(input_array)
        result = "Mine" if prediction[0] == "M" else "Rock"
        return render_template("index.html", prediction_text=f"Prediction: {result}")
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run()
