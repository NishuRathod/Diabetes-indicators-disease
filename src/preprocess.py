from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score




def build_preprocessor(numerical_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
    ])


    cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
    ])


    preprocessor = ColumnTransformer([
    ("num", num_pipeline, numerical_features),
    ("cat", cat_pipeline, categorical_features)
     ])
    return preprocessor




def get_features_and_target(df: pd.DataFrame, target_col: str = "Diabetes") -> Tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
       raise KeyError(f"Target column '{target_col}' not in dataframe")
    X = df.drop(columns=[target_col])
    y = df[target_col].copy()
    return X, y




def compute_classification_metrics(y_true, y_pred, y_proba=None) -> dict:
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_proba is not None:
        try:
            metrics['roc_auc'] = float(roc_auc_score(y_true, y_proba))
        except Exception:
            metrics['roc_auc'] = None
    return metrics