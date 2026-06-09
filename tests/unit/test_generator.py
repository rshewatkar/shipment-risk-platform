"""
Unit tests for the ShipmentDataGenerator.
Tests cover output shape, schema compliance,
risk rate bounds, and reproducibility.
"""
import pandas as pd
import pytest

from src.data.generator import ShipmentDataGenerator
from src.data.schema import COLUMN_NAMES, CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.data.validation import validate_dataset


@pytest.fixture(scope="module")
def small_df():
    """Generate a small dataset once for all tests in this module."""
    gen = ShipmentDataGenerator(n_samples=500, seed=42)
    return gen.generate()


# ── Output shape tests ─────────────────────────────────────

def test_generates_correct_number_of_rows(small_df):
    assert len(small_df) == 500


def test_generates_all_required_columns(small_df):
    for col in COLUMN_NAMES:
        assert col in small_df.columns, f"Missing column: {col}"


def test_no_null_values_in_output(small_df):
    null_counts = small_df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    assert len(cols_with_nulls) == 0, (
        f"Columns with nulls: {cols_with_nulls.to_dict()}"
    )


# ── Risk rate tests ────────────────────────────────────────

def test_risk_rate_within_expected_bounds(small_df):
    risk_rate = small_df["risk_label"].mean()
    assert 0.05 <= risk_rate <= 0.25, (
        f"Risk rate {risk_rate:.1%} outside expected 5-25% range"
    )


def test_risk_label_is_binary(small_df):
    unique_labels = set(small_df["risk_label"].unique())
    assert unique_labels.issubset({0, 1})


# ── Feature distribution tests ─────────────────────────────

def test_shipment_value_within_bounds(small_df):
    assert small_df["shipment_value"].min() >= 5.0
    assert small_df["shipment_value"].max() <= 5000.0


def test_package_weight_within_bounds(small_df):
    assert small_df["package_weight"].min() >= 0.1
    assert small_df["package_weight"].max() <= 70.0


def test_carrier_names_are_valid(small_df):
    valid = {"FedEx", "UPS", "USPS", "DHL", "OnTrac"}
    assert set(small_df["carrier_name"].unique()).issubset(valid)


def test_all_delivery_zones_present(small_df):
    zones = set(small_df["delivery_zone"].unique())
    assert len(zones) >= 3, "Expected at least 3 delivery zones in 500 records"


def test_payment_methods_are_valid(small_df):
    valid = {"CREDIT_CARD", "DEBIT_CARD", "PAYPAL", "CRYPTO", "COD"}
    assert set(small_df["payment_method"].unique()).issubset(valid)


def test_address_types_are_valid(small_df):
    valid = {"RESIDENTIAL", "APARTMENT", "BUSINESS", "PO_BOX"}
    assert set(small_df["address_type"].unique()).issubset(valid)


def test_reliability_score_within_bounds(small_df):
    assert small_df["carrier_reliability_score"].min() >= 0.0
    assert small_df["carrier_reliability_score"].max() <= 1.0


def test_velocity_30d_gte_velocity_7d(small_df):
    assert (small_df["velocity_30d"] >= small_df["velocity_7d"]).all()


def test_is_high_value_matches_shipment_value(small_df):
    high_value_rows = small_df[small_df["shipment_value"] > 500]
    assert (high_value_rows["is_high_value"] == 1).all()


# ── Reproducibility test ───────────────────────────────────

def test_same_seed_produces_identical_output():
    gen1 = ShipmentDataGenerator(n_samples=100, seed=99)
    gen2 = ShipmentDataGenerator(n_samples=100, seed=99)
    df1 = gen1.generate()
    df2 = gen2.generate()
    pd.testing.assert_frame_equal(df1, df2)


def test_different_seeds_produce_different_output():
    gen1 = ShipmentDataGenerator(n_samples=100, seed=1)
    gen2 = ShipmentDataGenerator(n_samples=100, seed=2)
    df1 = gen1.generate()
    df2 = gen2.generate()
    assert not df1["shipment_value"].equals(df2["shipment_value"])


# ── Full validation pipeline test ─────────────────────────

def test_generated_data_passes_full_validation(small_df):
    result = validate_dataset(small_df)
    assert result["success"] is True, (
        f"Validation failed:\n" + "\n".join(result["errors"])
    )