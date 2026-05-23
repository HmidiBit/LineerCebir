from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Model yükleme
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    features = []

    for i in range(30):
        value = float(request.form[f"feature{i}"])
        features.append(value)

    data = np.array(features).reshape(1, -1)

    # Normalize
    data_scaled = scaler.transform(data)

    # PCA
    data_pca = pca.transform(data_scaled)

    # Prediction
    prediction = model.predict(data_pca)[0]

    if prediction == 1:
        result = "Fraudulent Transaction Detected"
    else:
        result = "Normal Transaction"

    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)