import joblib
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
#from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import json

from src.config__ import load_config
from src.data_loader import load_data
from src.preprocess import build_preprocessor, get_features_and_target, compute_classification_metrics
from imblearn.over_sampling import SMOTE
def main():

    cfg = load_config()

    # Load dataset
    df = load_data()

    # Split X, y
    X, y = get_features_and_target(df)

    # Preprocessor (IMPORTANT: use X, not df)
    preprocessor = build_preprocessor(X)

    # Model
    model = RandomForestClassifier(
        n_estimators=cfg["model_params"]["n_estimators"],
        max_depth=cfg["model_params"]["max_depth"],
        min_samples_split=cfg["model_params"]["min_samples_split"],
        min_samples_leaf=cfg["model_params"]["min_samples_leaf"],
        random_state=cfg["training"]["random_state"],
        class_weight="balanced"
    )

    # Pipeline
    pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("smote", SMOTE()),
    ("classifier", model)
])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg["training"]["test_size"],
        random_state=cfg["training"]["random_state"]
    )

    # Fit
    pipeline.fit(X_train, y_train)

    # Predictions
    # y_pred = pipeline.predict(X_test)
    y_pred = (pipeline.predict_proba(X_test)[:, 1] > 0.3).astype(int)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # Metrics
    metrics = compute_classification_metrics(y_test, y_pred, y_proba)

    # Save model
    joblib.dump(pipeline, cfg["paths"]["model"])

    # Save metrics
    with open(cfg["paths"]["metrics"], "w") as f:
        json.dump(metrics, f, indent=4)

    print("\n Training complete!")
    print("Saved model →", cfg["paths"]["model"])
    print("Saved metrics →", cfg["paths"]["metrics"])
    print("\n Metrics:", metrics)


if __name__ == "__main__":
    main()


