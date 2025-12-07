#from flask import Flask, render_template, request
import os
import pandas as pd
import joblib
from pathlib import Path
from webapp.utils.mapper import map_inputs
from flask import Flask, render_template, request

app = Flask(
    __name__,
    template_folder="templates"  # explicitly set if your templates folder is not default
)

MODEL_PATH = Path("E:/Diabetes-indicators-disease/result/model.pkl")
model = joblib.load(MODEL_PATH)


FEATURES = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education",
    "Income"
]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        inputs = map_inputs(request.form)
        df = pd.DataFrame([inputs], columns=FEATURES)

        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[0][1]

        result_text = "Diabetes: YES" if pred == 1 else "Diabetes: NO"

        return render_template(
            "result.html",
            result=result_text,
            probability=f"{prob:.4f}"
        )
    except Exception as e:
        return f"Error in prediction: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
