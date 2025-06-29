import os
from flask import Flask, render_template, request

app = Flask(__name__)  # ✅ this must be the very first 'app' assignment

template_dir = os.path.abspath('C:\\Users\\JAHNAVI\\New folder\\Traffic\\.vscode\\template')

import pickle
import pandas as pd
# Load model correctly
with open("traffic_model.pkl", "rb") as f:
    model = pickle.load(f)


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    temp = float(request.form["temp"])
    rain = float(request.form["rain"])
    snow = float(request.form["snow"])
    # Map these correctly:
    holiday = 1 if request.form["holiday"] == "Yes" else 0
    weather_map = {"Clear": 0, "Cloudy": 1, "Rain": 2, "Snow": 3}
    weather = weather_map.get(request.form["weather"], 0)

    input_df = pd.DataFrame([[temp, rain, snow, holiday, weather]],
                            columns=["temp", "rain", "snow", "holiday", "weather"])
    input_df = pd.get_dummies(input_df)
    for col in model.feature_names_in_:
        expected_cols = ['col1', 'col2', 'colX']
    input_df = input_df.reindex(columns=expected_cols, fill_value=0)

    input_df = input_df[model.feature_names_in_]

    prediction = model.predict(input_df)[0]
    return render_template("result.html", volume=int(prediction))

if __name__ == "__main__":
    app.run(debug=True)
