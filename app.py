from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Train model on startup - no pkl file needed
df = pd.read_csv("Sonar.csv", header=None)
X = df.drop(columns=60, axis=1)
Y = df[60]
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.1, stratify=Y, random_state=1
)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)
print("Model trained successfully!")

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
