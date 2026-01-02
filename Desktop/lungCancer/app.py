from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model/knn_model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    values = [
        int(request.form['gender']),
        int(request.form['age']),
        int(request.form['smoking']),
        int(request.form['yellow_fingers']),
        int(request.form['anxiety']),
        int(request.form['peer_pressure']),
        int(request.form['chronic_disease']),
        int(request.form['fatigue']),
        int(request.form['allergy']),
        int(request.form['wheezing']),
        int(request.form['alcohol']),
        int(request.form['coughing']),
        int(request.form['shortness_breath']),
        int(request.form['swallowing']),
        int(request.form['chest_pain'])
    ]

    values = np.array(values).reshape(1, -1)
    values = scaler.transform(values)

    result = model.predict(values)

    output = "Lung Cancer Detected 😟" if result[0] == 1 else "No Lung Cancer 😊"

    return render_template("result.html", prediction=output)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

