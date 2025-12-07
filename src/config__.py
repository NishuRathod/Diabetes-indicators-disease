import yaml
from pathlib import Path

def load_config(path="params.yaml"):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError("Config file is empty or invalid YAML.")

    return config
