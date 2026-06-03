"""
Unit tests for the configuration loader.
These run on every CI push to verify the project config is valid.
"""

from src.utils.config import load_config


def test_config_loads_successfully():
    """Config file must load without errors."""
    config = load_config()
    assert config is not None


def test_config_has_required_sections():
    """All major config sections must exist."""
    config = load_config()
    required_sections = ["project", "data", "features", "model", "api", "mlflow"]
    for section in required_sections:
        assert section in config, f"Missing config section: {section}"


def test_config_project_name():
    """Project name must match expected value."""
    config = load_config()
    assert config["project"]["name"] == "shipment-risk-platform"


def test_config_data_paths_defined():
    """Data paths must be defined."""
    config = load_config()
    assert "raw_path" in config["data"]
    assert "processed_path" in config["data"]


def test_config_risk_levels_complete():
    """All four risk levels must be configured."""
    config = load_config()
    levels = config["risk"]["levels"]
    assert "LOW" in levels
    assert "MEDIUM" in levels
    assert "HIGH" in levels
    assert "CRITICAL" in levels


def test_config_thresholds_are_ordered():
    """Risk thresholds must be in ascending order."""
    config = load_config()
    thresholds = config["model"]["thresholds"]
    assert thresholds["low_risk"] < thresholds["medium_risk"]
    assert thresholds["medium_risk"] < thresholds["high_risk"]
