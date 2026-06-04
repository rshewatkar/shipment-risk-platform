"""
Shipment dataset schema definition.
Single source of truth for column names, types, and valid ranges.
Used by: data generator, validator, feature pipeline, API schemas.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class ColumnSchema:
    """Schema definition for a single dataset column."""
    name: str
    dtype: str
    nullable: bool = False
    min_value: float = None
    max_value: float = None
    allowed_values: List[str] = None
    description: str = ""


# -- Identity columns (not used as features) ---------------
IDENTITY_COLUMNS = [
    ColumnSchema("shipment_id", "string", description="Unique shipment identifier"),
    ColumnSchema("customer_id", "string", description="Customer identifier"),
    ColumnSchema("created_at", "datetime", description="Label creation timestamp"),
]

# -- Shipment value and package -----------------------------
SHIPMENT_COLUMNS = [
    ColumnSchema(
        "shipment_value", "float",
        min_value=5.0, max_value=5000.0,
        description="Declared value in USD"
    ),
    ColumnSchema(
        "package_weight", "float",
        min_value=0.1, max_value=70.0,
        description="Weight in kg"
    ),
    ColumnSchema(
        "shipment_type", "string",
        allowed_values=["STANDARD", "EXPRESS", "OVERNIGHT", "ECONOMY"],
        description="Service level"
    ),
    ColumnSchema(
        "is_high_value", "int",
        min_value=0, max_value=1,
        description="1 if shipment_value > 500"
    ),
    ColumnSchema(
        "value_to_weight_ratio", "float",
        min_value=0.0,
        description="Derived: shipment_value / package_weight"
    ),
]

# -- Carrier and delivery -----------------------------------
CARRIER_COLUMNS = [
    ColumnSchema(
        "carrier_name", "string",
        allowed_values=["FedEx", "UPS", "USPS", "DHL", "OnTrac"],
        description="Carrier handling the shipment"
    ),
    ColumnSchema(
        "carrier_reliability_score", "float",
        min_value=0.0, max_value=1.0,
        description="Historical reliability for this carrier+route"
    ),
    ColumnSchema(
        "delivery_zone", "string",
        allowed_values=["URBAN", "SUBURBAN", "RURAL", "REMOTE"],
        description="Delivery area classification"
    ),
    ColumnSchema(
        "delivery_distance_km", "float",
        min_value=1.0, max_value=3000.0,
        description="Route distance in km"
    ),
    ColumnSchema(
        "days_to_delivery", "int",
        min_value=1, max_value=14,
        description="Estimated delivery days"
    ),
]

# -- Address intelligence -----------------------------------
ADDRESS_COLUMNS = [
    ColumnSchema(
        "address_type", "string",
        allowed_values=["RESIDENTIAL", "APARTMENT", "BUSINESS", "PO_BOX"],
        description="Type of delivery address"
    ),
    ColumnSchema(
        "address_risk_score", "float",
        min_value=0.0, max_value=1.0,
        description="Historical risk score for this zip area"
    ),
    ColumnSchema(
        "is_forwarding_address", "int",
        min_value=0, max_value=1,
        description="1 if address is a mail forwarding service"
    ),
    ColumnSchema(
        "delivery_attempt_history", "int",
        min_value=0, max_value=5,
        description="Prior failed delivery attempts at this address"
    ),
]

# -- Customer behavior --------------------------------------
CUSTOMER_COLUMNS = [
    ColumnSchema(
        "customer_order_count", "int",
        min_value=1, max_value=500,
        description="Total historical orders"
    ),
    ColumnSchema(
        "customer_dispute_rate", "float",
        min_value=0.0, max_value=1.0,
        description="Fraction of orders with disputes"
    ),
    ColumnSchema(
        "customer_return_rate", "float",
        min_value=0.0, max_value=1.0,
        description="Fraction of orders returned"
    ),
    ColumnSchema(
        "customer_account_age_days", "int",
        min_value=1, max_value=3650,
        description="Account age in days"
    ),
    ColumnSchema(
        "velocity_7d", "int",
        min_value=0, max_value=50,
        description="Orders in last 7 days"
    ),
    ColumnSchema(
        "velocity_30d", "int",
        min_value=0, max_value=200,
        description="Orders in last 30 days"
    ),
]

# -- Payment and order context ------------------------------
PAYMENT_COLUMNS = [
    ColumnSchema(
        "payment_method", "string",
        allowed_values=["CREDIT_CARD", "DEBIT_CARD", "PAYPAL", "CRYPTO", "COD"],
        description="Payment method used"
    ),
    ColumnSchema(
        "is_weekend_order", "int",
        min_value=0, max_value=1,
        description="1 if order placed on weekend"
    ),
    ColumnSchema(
        "is_holiday_period", "int",
        min_value=0, max_value=1,
        description="1 if order placed during holiday period"
    ),
    ColumnSchema(
        "order_hour", "int",
        min_value=0, max_value=23,
        description="Hour of day when order was placed"
    ),
]

# -- Target variable ----------------------------------------
TARGET_COLUMN = ColumnSchema(
    "risk_label", "int",
    min_value=0, max_value=1,
    description="1 if shipment resulted in loss, theft, fraud, or damage"
)

# -- Master feature list (used by ML pipeline) -------------
ALL_FEATURE_COLUMNS = (
    SHIPMENT_COLUMNS
    + CARRIER_COLUMNS
    + ADDRESS_COLUMNS
    + CUSTOMER_COLUMNS
    + PAYMENT_COLUMNS
)

NUMERIC_FEATURES = [
    col.name for col in ALL_FEATURE_COLUMNS
    if col.dtype in ("float", "int")
]

CATEGORICAL_FEATURES = [
    col.name for col in ALL_FEATURE_COLUMNS
    if col.dtype == "string"
]

ALL_COLUMNS = (
    IDENTITY_COLUMNS
    + ALL_FEATURE_COLUMNS
    + [TARGET_COLUMN]
)

COLUMN_NAMES = [col.name for col in ALL_COLUMNS]
