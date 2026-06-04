"""
Unit tests for dataset schema definitions and validation logic.
"""
import pandas as pd
import numpy as np
import pytest
from src.data.schema import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    COLUMN_NAMES,
    TARGET_COLUMN,
    ALL_FEATURE_COLUMNS,
)
from src.data.validation import validate_dataset


# -- Schema structure tests ---------------------------------

def test_numeric_features_not_empty():
    assert len(NUMERIC_FEATURES) > 0


def test_categorical_features_not_empty():
    assert len(CATEGORICAL_FEATURES) > 0


def test_no_overlap_between_numeric_and_categorical():
    overlap = set(NUMERIC_FEATURES) & set(CATEGORICAL_FEATURES)
    assert len(overlap) == 0, f"Features in both lists: {overlap}"


def test_target_column_is_binary_int():
    assert TARGET_COLUMN.dtype == "int"
    assert TARGET_COLUMN.min_value == 0
    assert TARGET_COLUMN.max_value == 1


def test_all_categorical_columns_have_allowed_values():
    for col in ALL_FEATURE_COLUMNS:
        if col.dtype == "string":
            assert col.allowed_values is not None, (
                f"Categorical column '{col.name}' missing allowed_values"
            )
            assert len(col.allowed_values) > 0


def test_all_numeric_columns_have_min_max():
    for col in ALL_FEATURE_COLUMNS:
        if col.dtype in ("float", "int") and col.name not in (
            "value_to_weight_ratio",
        ):
            assert col.min_value is not None, (
                f"Numeric column '{col.name}' missing min_value"
            )


# -- Validation logic tests ---------------------------------

@pytest.fixture
def minimal_valid_df():
    """Smallest valid DataFrame that passes all validation checks."""
    data = {col: [None] for col in COLUMN_NAMES}
    df = pd.DataFrame(data)

    # Fill required values
    df["shipment_id"] = "SHP-TEST-001"
    df["customer_id"] = "CUST-001"
    df["created_at"] = pd.Timestamp("2024-01-01")
    df["shipment_value"] = 100.0
    df["package_weight"] = 2.0
    df["shipment_type"] = "STANDARD"
    df["is_high_value"] = 0
    df["value_to_weight_ratio"] = 50.0
    df["carrier_name"] = "FedEx"
    df["carrier_reliability_score"] = 0.9
    df["delivery_zone"] = "URBAN"
    df["delivery_distance_km"] = 50.0
    df["days_to_delivery"] = 3
    df["address_type"] = "RESIDENTIAL"
    df["address_risk_score"] = 0.1
    df["is_forwarding_address"] = 0
    df["delivery_attempt_history"] = 0
    df["customer_order_count"] = 10
    df["customer_dispute_rate"] = 0.01
    df["customer_return_rate"] = 0.05
    df["customer_account_age_days"] = 365
    df["velocity_7d"] = 1
    df["velocity_30d"] = 4
    df["payment_method"] = "CREDIT_CARD"
    df["is_weekend_order"] = 0
    df["is_holiday_period"] = 0
    df["order_hour"] = 14
    df["risk_label"] = 0
    return df


def test_valid_dataframe_passes(minimal_valid_df):
    result = validate_dataset(minimal_valid_df)
    assert result["success"] is True
    assert len(result["errors"]) == 0


def test_empty_dataframe_fails():
    df = pd.DataFrame()
    result = validate_dataset(df)
    assert result["success"] is False
    assert any("empty" in e.lower() for e in result["errors"])


def test_missing_column_fails(minimal_valid_df):
    df = minimal_valid_df.drop(columns=["shipment_value"])
    result = validate_dataset(df)
    assert result["success"] is False
    assert any("shipment_value" in e for e in result["errors"])


def test_null_in_critical_column_fails(minimal_valid_df):
    minimal_valid_df.loc[0, "carrier_name"] = None
    result = validate_dataset(minimal_valid_df)
    assert result["success"] is False


def test_value_below_minimum_fails(minimal_valid_df):
    minimal_valid_df.loc[0, "shipment_value"] = -10.0
    result = validate_dataset(minimal_valid_df)
    assert result["success"] is False


def test_value_above_maximum_fails(minimal_valid_df):
    minimal_valid_df.loc[0, "shipment_value"] = 99999.0
    result = validate_dataset(minimal_valid_df)
    assert result["success"] is False


def test_invalid_categorical_value_fails(minimal_valid_df):
    minimal_valid_df.loc[0, "carrier_name"] = "AmazonLogistics"
    result = validate_dataset(minimal_valid_df)
    assert result["success"] is False


def test_invalid_risk_label_fails(minimal_valid_df):
    minimal_valid_df.loc[0, "risk_label"] = 5
    result = validate_dataset(minimal_valid_df)
    assert result["success"] is False


def test_stats_returned_on_success(minimal_valid_df):
    result = validate_dataset(minimal_valid_df)
    assert "stats" in result
    assert result["stats"]["n_rows"] == 1
