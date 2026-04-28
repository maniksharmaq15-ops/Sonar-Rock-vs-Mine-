from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
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
