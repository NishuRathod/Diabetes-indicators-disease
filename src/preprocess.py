from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
from src.config__ import load_config


def get_features_and_target(df: pd.DataFrame):
    config = load_config()
    target_col = config["features"]["target"]

    # Separate X and y
    y = df[target_col].copy()
    X = df.drop(columns=[target_col])

    return X, y


def build_preprocessor(df: pd.DataFrame):
    config = load_config()
    target_col = config["features"]["target"]

    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # Remove target column from numeric features (IMPORTANT FIX)
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols)
        ],
        remainder="passthrough"  # keep other columns if any
    )
    return preprocessor


def compute_classification_metrics(y_true, y_pred, y_proba):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_true, y_proba), 4)
    }
