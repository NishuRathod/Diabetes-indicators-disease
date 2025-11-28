from pathlib import Path
import pandas as pd
from .config__ import load_config
from typing import List

def load_data():
    cfg = load_config()
    data_path = Path(cfg["data"]["raw_path"])

    files = list(data_path.glob("*.csv"))
    dfs = []

    for f in files:
        df_temp = pd.read_csv(f)
        df_temp["source"] = f.name
        dfs.append(df_temp)

    df = pd.concat(dfs, ignore_index=True)
    return df


def basic_eda(df: pd.DataFrame) -> None:
    print("rows, cols:", df.shape)
    print("missing per column:\n", df.isna().sum())
    print(df.describe(include='all').T)

