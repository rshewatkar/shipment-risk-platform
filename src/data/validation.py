"""
Data validation using Great Expectations.
Validates that any shipment dataset matches the defined schema
before it enters the feature engineering pipeline.

Usage:
    from src.data.validation import validate_dataset
    result = validate_dataset(df)
    if not result["success"]:
        raise ValueError(result["errors"])
"""
import pandas as pd
from typing import Dict, Any
from src.data.schema import (
    ALL_FEATURE_COLUMNS,
    TARGET_COLUMN,
    COLUMN_NAMES,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def validate_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate a shipment DataFrame against the defined schema.

    Checks performed:
    - All required columns are present
    - No unexpected null values in critical columns
    - Numeric columns are within defined min/max ranges
    - Categorical columns only contain allowed values
    - Target variable is binary (0 or 1 only)
    - Dataset is not empty

    Args:
        df: Raw or processed shipment DataFrame.

    Returns:
        dict with keys:
            success (bool): True if all checks passed.
            errors (list): List of validation error messages.
            warnings (list): Non-fatal issues found.
            stats (dict): Basic dataset statistics.
    """
    errors = []
    warnings = []

    logger.info("starting dataset validation", n_rows=len(df), n_cols=len(df.columns))

    # -- Check 1: Dataset not empty -------------------------
    if len(df) == 0:
        errors.append("Dataset is empty - no rows found")
        return _result(False, errors, warnings, df)

    # -- Check 2: Required columns present -----------------
    missing_cols = [col for col in COLUMN_NAMES if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    if errors:
        return _result(False, errors, warnings, df)

    # -- Check 3: No nulls in critical columns --------------
    critical_columns = (
        ["risk_label", "shipment_value", "carrier_name",
         "address_type", "payment_method", "customer_dispute_rate"]
    )
    for col in critical_columns:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            errors.append(
                f"Critical column '{col}' has {null_count} null values "
                f"({null_count/len(df)*100:.1f}%)"
            )

    # -- Check 4: Numeric range validation -----------------
    for col_schema in ALL_FEATURE_COLUMNS:
        if col_schema.dtype not in ("float", "int"):
            continue
        if col_schema.name not in df.columns:
            continue

        col = df[col_schema.name]

        if col_schema.min_value is not None:
            below_min = (col < col_schema.min_value).sum()
            if below_min > 0:
                errors.append(
                    f"Column '{col_schema.name}' has {below_min} values "
                    f"below minimum {col_schema.min_value}"
                )

        if col_schema.max_value is not None:
            above_max = (col > col_schema.max_value).sum()
            if above_max > 0:
                errors.append(
                    f"Column '{col_schema.name}' has {above_max} values "
                    f"above maximum {col_schema.max_value}"
                )

    # -- Check 5: Categorical value validation --------------
    for col_schema in ALL_FEATURE_COLUMNS:
        if col_schema.allowed_values is None:
            continue
        if col_schema.name not in df.columns:
            continue

        unexpected = set(df[col_schema.name].dropna().unique()) - set(
            col_schema.allowed_values
        )
        if unexpected:
            errors.append(
                f"Column '{col_schema.name}' contains unexpected values: "
                f"{unexpected}. Allowed: {col_schema.allowed_values}"
            )

    # -- Check 6: Target variable is binary ----------------
    if "risk_label" in df.columns:
        unique_labels = set(df["risk_label"].unique())
        if not unique_labels.issubset({0, 1}):
            errors.append(
                f"risk_label must only contain 0 and 1. "
                f"Found: {unique_labels}"
            )

    # -- Check 7: Class imbalance warning ------------------
    if "risk_label" in df.columns:
        risk_rate = df["risk_label"].mean()
        if risk_rate < 0.05:
            warnings.append(
                f"Very low risk rate: {risk_rate:.1%}. "
                f"Consider SMOTE or class weights."
            )
        elif risk_rate > 0.40:
            warnings.append(
                f"Unusually high risk rate: {risk_rate:.1%}. "
                f"Verify data generation logic."
            )

    # -- Check 8: Duplicate shipment IDs -------------------
    if "shipment_id" in df.columns:
        dup_count = df["shipment_id"].duplicated().sum()
        if dup_count > 0:
            warnings.append(
                f"Found {dup_count} duplicate shipment_id values"
            )

    success = len(errors) == 0

    if success:
        logger.info(
            "validation passed",
            n_rows=len(df),
            warnings=len(warnings),
        )
    else:
        logger.error(
            "validation failed",
            n_errors=len(errors),
            errors=errors,
        )

    return _result(success, errors, warnings, df)


def _result(
    success: bool,
    errors: list,
    warnings: list,
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """Build the validation result dictionary."""
    stats = {}
    if len(df) > 0:
        stats = {
            "n_rows": len(df),
            "n_columns": len(df.columns),
            "risk_rate": float(df["risk_label"].mean())
            if "risk_label" in df.columns
            else None,
            "null_counts": df.isnull().sum().to_dict(),
        }
    return {
        "success": success,
        "errors": errors,
        "warnings": warnings,
        "stats": stats,
    }
