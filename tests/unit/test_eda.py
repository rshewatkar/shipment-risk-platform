"""
Tests to verify EDA assumptions about the generated dataset.
These validate the business logic embedded in the data generator.
"""
import pandas as pd
import pytest
import numpy as np


@pytest.fixture(scope="module")
def df():
    return pd.read_csv("data/raw/shipments_raw.csv")


# ── Dataset health checks ──────────────────────────────────

def test_dataset_has_expected_size(df):
    assert len(df) >= 10000, "Dataset too small for meaningful EDA"


def test_risk_rate_is_realistic(df):
    rate = df["risk_label"].mean()
    assert 0.08 <= rate <= 0.20, (
        f"Risk rate {rate:.1%} outside realistic 8-20% range"
    )


def test_no_nulls_in_dataset(df):
    assert df.isnull().sum().sum() == 0


# ── Business logic validation ──────────────────────────────

def test_forwarding_address_has_higher_risk(df):
    """Core business assumption: forwarding addresses are riskier."""
    fwd_risk = df[df["is_forwarding_address"] == 1]["risk_label"].mean()
    std_risk = df[df["is_forwarding_address"] == 0]["risk_label"].mean()
    assert fwd_risk > std_risk, (
        "Forwarding addresses should have higher risk than standard"
    )


def test_high_value_has_higher_risk(df):
    """High value shipments should attract more fraud and theft."""
    hv_risk = df[df["is_high_value"] == 1]["risk_label"].mean()
    lv_risk = df[df["is_high_value"] == 0]["risk_label"].mean()
    assert hv_risk > lv_risk, (
        "High value shipments should have higher risk rate"
    )


def test_crypto_payment_has_elevated_risk(df):
    """Crypto payment should be above average risk."""
    overall_risk = df["risk_label"].mean()
    crypto_risk = df[df["payment_method"] == "CRYPTO"]["risk_label"].mean()
    assert crypto_risk > overall_risk, (
        "CRYPTO payment should have above-average risk"
    )


def test_new_accounts_have_higher_risk(df):
    """New customer accounts should be higher risk."""
    new_acct = df[df["customer_account_age_days"] <= 30]["risk_label"].mean()
    old_acct = df[df["customer_account_age_days"] >= 365]["risk_label"].mean()
    assert new_acct > old_acct, (
        "New accounts should have higher risk than established ones"
    )


def test_high_dispute_rate_correlates_with_risk(df):
    """Customers with high dispute rates should be riskier."""
    high_dispute = df[df["customer_dispute_rate"] > 0.3]["risk_label"].mean()
    low_dispute = df[df["customer_dispute_rate"] < 0.05]["risk_label"].mean()
    assert high_dispute > low_dispute


def test_velocity_7d_higher_for_risky(df):
    """Fraudsters place more orders in burst patterns."""
    risky_vel = df[df["risk_label"] == 1]["velocity_7d"].mean()
    safe_vel = df[df["risk_label"] == 0]["velocity_7d"].mean()
    assert risky_vel > safe_vel


# ── Feature distribution checks ───────────────────────────

def test_all_carriers_represented(df):
    carriers = set(df["carrier_name"].unique())
    expected = {"FedEx", "UPS", "USPS", "DHL", "OnTrac"}
    assert carriers == expected


def test_all_payment_methods_represented(df):
    methods = set(df["payment_method"].unique())
    expected = {"CREDIT_CARD", "DEBIT_CARD", "PAYPAL", "CRYPTO", "COD"}
    assert methods == expected


def test_all_zones_represented(df):
    zones = set(df["delivery_zone"].unique())
    expected = {"URBAN", "SUBURBAN", "RURAL", "REMOTE"}
    assert zones == expected


def test_address_risk_score_range(df):
    assert df["address_risk_score"].min() >= 0.0
    assert df["address_risk_score"].max() <= 1.0


def test_top_correlations_make_business_sense(df):
    """
    The features most correlated with risk_label should be
    the ones with the strongest business signals.
    """
    from src.data.schema import NUMERIC_FEATURES

    numeric_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    corr = df[numeric_cols + ["risk_label"]].corr()["risk_label"].drop("risk_label")
    top_features = corr.abs().sort_values(ascending=False).head(5).index.tolist()

    # These should always be in the top correlators
    expected_top = {
        "customer_dispute_rate",
        "address_risk_score",
        "is_forwarding_address",
    }
    overlap = set(top_features) & expected_top
    assert len(overlap) >= 2, (
        f"Expected key risk features in top correlators. "
        f"Got: {top_features}"
    )