"""import joblib
import os
from pathlib import Path
import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline


from src.config__ import load_config
from src.data_loader import load_data, basic_eda
from src.preprocess import build_preprocessor, get_features_and_target, compute_classification_metrics

def main(params_path: str = "params.yaml"):
    params = load_config(params_path)
    data_cfg = params['data']
    train_cfg = params['train']
    mlflow_cfg = params.get('mlflow', {})
    preprocess_cfg = params.get('preprocess', {})


    df = load_data(data_cfg['path'], data_cfg.get('pattern', '*.csv'))
    basic_eda(df)


    X, y = get_features_and_target(df, target_col='Diabetes_binary')


    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=train_cfg.get('test_size', 0.2), random_state=train_cfg.get('random_state', 42), stratify=y
    )


    preprocessor = build_preprocessor(preprocess_cfg['numerical_features'], preprocess_cfg['categorical_features'])


    clf = RandomForestClassifier(random_state=train_cfg.get('random_state', 42))


    pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("clf", clf)
    ])


    param_grid = {
    'clf__n_estimators': train_cfg['model'].get('n_estimators', [100]),
    'clf__max_depth': train_cfg['model'].get('max_depth', [None])
    }


    grid = GridSearchCV(pipeline, param_grid, cv=train_cfg.get('cv', 5), n_jobs=-1, verbose=1)
    # MLflow experiment
    mlflow.set_experiment(mlflow_cfg.get('experiment_name', 'Diabetes-Indicators'))
    with mlflow.start_run(run_name=mlflow_cfg.get('run_name', 'run')):
       grid.fit(X_train, y_train)


       best = grid.best_estimator_
       y_pred = best.predict(X_test)
       # try predict_proba
       try:
           y_proba = best.predict_proba(X_test)[:, 1]
       except Exception:
            y_proba = None


       metrics = compute_classification_metrics(y_test, y_pred, y_proba)
       # log params and metrics
       mlflow.log_params({
       'best_params': grid.best_params_
        })
       mlflow.log_metrics(metrics)


       # save model to models/
       models_dir = Path('models')
       models_dir.mkdir(parents=True, exist_ok=True)
       model_path = models_dir / 'model.joblib'
       joblib.dump(best, model_path)
       mlflow.sklearn.log_model(best, artifact_path='model')


       print("Best params:", grid.best_params_)
       print("Metrics:", metrics)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', default='params.yaml')
    args = parser.parse_args()
    main(args.params)"""
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

    print("\n🎯 Training complete!")
    print("Saved model →", cfg["paths"]["model"])
    print("Saved metrics →", cfg["paths"]["metrics"])
    print("\n📊 Metrics:", metrics)


if __name__ == "__main__":
    main()


