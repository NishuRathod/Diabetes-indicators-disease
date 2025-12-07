import pandas as pd
from pathlib import Path
from .config__ import load_config

def load_data():
    config = load_config()
    data_path = Path(config["paths"]["raw_data"])

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)

    # Convert Diabetes_012 → Diabetes_binary
    df["Diabetes_binary"] = df["Diabetes_012"].apply(lambda x: 1 if x >= 1 else 0)

    # Remove unnecessary columns
    drop_cols = ["Diabetes_012", "source"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df


