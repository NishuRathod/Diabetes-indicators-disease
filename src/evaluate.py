import joblib
import pandas as pd
from sklearn.metrics import classification_report
from src.config__ import load_config
from src.data_loader import load_data
from src.preprocess import get_features_and_target


def evaluate_model():
    cfg = load_config()

    # Load dataset
    df = load_data()
    X, y = get_features_and_target(df)

    # Load trained model
    model = joblib.load(cfg["paths"]["model"])

    # Predictions
    preds = model.predict(X)

    print("\n📌 Evaluation on full dataset:")
    print(classification_report(y, preds))


if __name__ == "__main__":
    evaluate_model()
