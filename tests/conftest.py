"""
Shared pytest fixtures available to all tests.
Add reusable setup here - loaded automatically by pytest.
"""

import pytest

from src.utils.config import load_config


@pytest.fixture(scope="session")
def config():
    """Provide loaded config to any test that needs it."""
    return load_config()


@pytest.fixture(scope="session")
def sample_shipment():
    """A valid shipment payload for API and model tests."""
    return {
        "shipment_id": "TEST-001",
        "shipment_value": 250.0,
        "package_weight": 2.5,
        "carrier_name": "FedEx",
        "shipment_type": "STANDARD",
        "delivery_zone": "URBAN",
        "payment_method": "CREDIT_CARD",
        "address_type": "RESIDENTIAL",
        "customer_order_count": 12,
        "customer_dispute_rate": 0.02,
        "customer_return_rate": 0.05,
        "delivery_distance_km": 45.0,
    }
