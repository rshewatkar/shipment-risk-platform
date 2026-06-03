"""
Central configuration loader.
Reads configs/config.yaml and exposes a typed settings object.
"""

from pathlib import Path

import yaml


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """
    Load and return the project configuration.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        dict: Full configuration dictionary.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return config


# Module-level config object - import this anywhere in the project
# Usage: from src.utils.config import config
config = load_config()
