from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("traffic_model.pkl", "rb"))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    temp = float(request.form["temp"])
    rain = float(request.form["rain"])
    snow = float(request.form["snow"])
    holiday = request.form["holiday"]
    weather = request.form["weather"]

    # Build DataFrame for prediction
    input_df = pd.DataFrame([[temp, rain, snow, holiday, weather]],
                            columns=["temp", "rain", "snow", "holiday", "weather"])

    # One-hot encode to match model training
    input_df = pd.get_dummies(input_df)

    # Ensure input columns match training columns
    for col in model.feature_names_in_:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[model.feature_names_in_]

    prediction = model.predict(input_df)[0]
    return render_template("result.html", volume=int(prediction))

if __name__ == "__main__":
    app.run(debug=True)