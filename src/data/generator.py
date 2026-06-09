"""
Synthetic Shipment Data Generator.

Generates realistic shipment records with injected risk patterns
for training the shipment risk ML model.

Key design decisions:
    - Seeded for full reproducibility
    - Risk patterns injected with realistic base rates
    - Feature correlations mimic real-world logistics data
    - Each risk type (fraud, theft, damage) has distinct feature signatures

Usage:
    from src.data.generator import ShipmentDataGenerator

    generator = ShipmentDataGenerator(n_samples=50000, seed=42)
    df = generator.generate()
    generator.save(df, "data/raw/shipments_raw.csv")
"""

import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ShipmentDataGenerator:
    """
    Generates synthetic shipment records with realistic risk patterns.

    Risk injection strategy:
        - FRAUD     (~4%): high-value + crypto/COD + new account
                           + forwarding address + high velocity
        - THEFT     (~3%): high-value + apartment/PO_BOX
                           + urban zone + low carrier reliability
        - DAMAGE    (~5%): heavy package + long distance
                           + rural/remote zone + low reliability
        - SAFE      (~88%): normal distributions across all features

    Args:
        n_samples : Total number of shipment records to generate.
        seed      : Random seed for full reproducibility.
    """

    REFERENCE_DATETIME = datetime(2026, 1, 1, 12, 0, 0)

    # ── Constants ──────────────────────────────────────────
    CARRIERS = ["FedEx", "UPS", "USPS", "DHL", "OnTrac"]

    CARRIER_RELIABILITY = {
        "FedEx":  (0.88, 0.06),   # (mean, std)
        "UPS":    (0.85, 0.07),
        "USPS":   (0.78, 0.09),
        "DHL":    (0.82, 0.08),
        "OnTrac": (0.74, 0.10),
    }

    SHIPMENT_TYPES = ["STANDARD", "EXPRESS", "OVERNIGHT", "ECONOMY"]
    SHIPMENT_TYPE_WEIGHTS = [0.55, 0.25, 0.10, 0.10]

    DELIVERY_ZONES = ["URBAN", "SUBURBAN", "RURAL", "REMOTE"]
    ZONE_WEIGHTS = [0.45, 0.35, 0.15, 0.05]

    ZONE_DISTANCE = {
        "URBAN":    (10,  80),    # (min_km, max_km)
        "SUBURBAN": (30,  300),
        "RURAL":    (100, 800),
        "REMOTE":   (300, 3000),
    }

    ADDRESS_TYPES = ["RESIDENTIAL", "APARTMENT", "BUSINESS", "PO_BOX"]
    ADDRESS_WEIGHTS = [0.45, 0.30, 0.20, 0.05]

    PAYMENT_METHODS = [
        "CREDIT_CARD", "DEBIT_CARD", "PAYPAL", "CRYPTO", "COD"
    ]
    PAYMENT_WEIGHTS = [0.50, 0.25, 0.15, 0.05, 0.05]

    def __init__(self, n_samples: int = 50000, seed: int = 42):
        self.n_samples = n_samples
        self.seed = seed

        # Set all random seeds for reproducibility
        np.random.seed(seed)
        random.seed(seed)

        logger.info(
            "data generator initialised",
            n_samples=n_samples,
            seed=seed,
        )

    # ──────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────

    def generate(self) -> pd.DataFrame:
        """
        Generate the full synthetic shipment dataset.

        Returns:
            pd.DataFrame: Complete shipment dataset with risk labels.
        """
        np.random.seed(self.seed)
        random.seed(self.seed)

        logger.info("generating shipment records...")

        records = []
        for i in range(self.n_samples):
            record = self._generate_single_record(i)
            records.append(record)

            if (i + 1) % 10000 == 0:
                logger.info(f"generated {i + 1:,} / {self.n_samples:,} records")

        df = pd.DataFrame(records)
        df = self._post_process(df)

        risk_rate = df["risk_label"].mean()
        logger.info(
            "dataset generation complete",
            n_rows=len(df),
            risk_rate=f"{risk_rate:.1%}",
            n_risky=int(df["risk_label"].sum()),
        )

        return df

    def save(self, df: pd.DataFrame, path: str) -> None:
        """
        Save the generated dataset to CSV.

        Args:
            df   : Generated DataFrame.
            path : Output file path.
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info("dataset saved", path=str(output_path), n_rows=len(df))

    # ──────────────────────────────────────────────────────
    # Private: Single Record Generation
    # ──────────────────────────────────────────────────────

    def _generate_single_record(self, index: int) -> dict:
        """Generate one shipment record with a randomly assigned risk type."""

        # Assign risk type first — drives correlated feature generation
        risk_type = self._assign_risk_type()

        # Generate base features
        record = {}
        record.update(self._generate_identity(index))
        record.update(self._generate_shipment_features(risk_type))
        record.update(self._generate_carrier_features(risk_type))
        record.update(self._generate_address_features(risk_type))
        record.update(self._generate_customer_features(risk_type))
        record.update(self._generate_payment_features(risk_type))

        # Set target label
        record["risk_label"] = 0 if risk_type == "SAFE" else 1

        return record

    def _assign_risk_type(self) -> str:
        """
        Randomly assign a risk type based on configured base rates.

        Returns:
            str: One of SAFE, FRAUD, THEFT, DAMAGE
        """
        roll = np.random.random()
        if roll < 0.04:
            return "FRAUD"
        elif roll < 0.07:   # 0.04 + 0.03
            return "THEFT"
        elif roll < 0.12:   # 0.07 + 0.05
            return "DAMAGE"
        else:
            return "SAFE"

    # ──────────────────────────────────────────────────────
    # Private: Feature Group Generators
    # ──────────────────────────────────────────────────────

    def _generate_identity(self, index: int) -> dict:
        """Generate shipment identity fields."""
        # Random timestamp in last 2 years
        days_ago = np.random.randint(0, 730)
        created_at = self.REFERENCE_DATETIME - timedelta(days=int(days_ago))

        return {
            "shipment_id": f"SHP-{created_at.year}-{index + 1:05d}",
            "customer_id": f"CUST-{np.random.randint(1, 15000):05d}",
            "created_at": created_at.strftime("%Y-%m-%d %H:%M:%S"),
        }

    def _generate_shipment_features(self, risk_type: str) -> dict:
        """
        Generate shipment value and package features.
        FRAUD and THEFT shipments skew toward higher values.
        DAMAGE shipments skew toward heavier packages.
        """
        if risk_type == "FRAUD":
            # Fraudsters target high-value items
            value = np.random.uniform(400, 5000)
            weight = np.random.uniform(0.1, 5.0)
        elif risk_type == "THEFT":
            # Theft also targets high-value
            value = np.random.uniform(300, 5000)
            weight = np.random.uniform(0.5, 15.0)
        elif risk_type == "DAMAGE":
            # Damage more common with heavy/bulky items
            value = np.random.uniform(5, 1000)
            weight = np.random.uniform(10.0, 70.0)
        else:
            # Normal distribution for safe shipments
            value = abs(np.random.lognormal(mean=4.5, sigma=1.2))
            value = np.clip(value, 5.0, 5000.0)
            weight = abs(np.random.lognormal(mean=1.0, sigma=0.8))
            weight = np.clip(weight, 0.1, 70.0)

        shipment_type = np.random.choice(
            self.SHIPMENT_TYPES,
            p=self.SHIPMENT_TYPE_WEIGHTS,
        )

        return {
            "shipment_value": round(float(value), 2),
            "package_weight": round(float(weight), 2),
            "shipment_type": shipment_type,
            "is_high_value": 1 if value > 500 else 0,
            "value_to_weight_ratio": round(float(value / max(weight, 0.1)), 2),
        }

    def _generate_carrier_features(self, risk_type: str) -> dict:
        """
        Generate carrier and delivery features.
        THEFT skews toward low-reliability carriers in urban zones.
        DAMAGE skews toward remote zones and long distances.
        """
        if risk_type == "THEFT":
            # Theft happens more with less reliable carriers
            carrier = np.random.choice(["USPS", "OnTrac", "DHL"],
                                       p=[0.40, 0.35, 0.25])
            zone = np.random.choice(["URBAN", "SUBURBAN"], p=[0.75, 0.25])
        elif risk_type == "DAMAGE":
            # Damage happens more in remote zones / long hauls
            carrier = np.random.choice(self.CARRIERS)
            zone = np.random.choice(
                ["RURAL", "REMOTE", "SUBURBAN", "URBAN"],
                p=[0.40, 0.30, 0.20, 0.10],
            )
        else:
            carrier = np.random.choice(self.CARRIERS)
            zone = np.random.choice(
                self.DELIVERY_ZONES, p=self.ZONE_WEIGHTS
            )

        # Carrier reliability score with noise
        mean_rel, std_rel = self.CARRIER_RELIABILITY[carrier]
        if risk_type in ("THEFT", "DAMAGE"):
            # Bad outcome → reliability was lower
            reliability = np.random.normal(mean_rel - 0.10, std_rel)
        else:
            reliability = np.random.normal(mean_rel, std_rel)
        reliability = float(np.clip(reliability, 0.0, 1.0))

        # Distance based on zone
        min_dist, max_dist = self.ZONE_DISTANCE[zone]
        distance = float(np.random.uniform(min_dist, max_dist))

        # Days to delivery based on zone + shipment type
        zone_base_days = {"URBAN": 2, "SUBURBAN": 3, "RURAL": 5, "REMOTE": 8}
        base_days = zone_base_days[zone]
        days = int(np.clip(
            np.random.normal(base_days, 1.5), 1, 14
        ))

        return {
            "carrier_name": carrier,
            "carrier_reliability_score": round(reliability, 3),
            "delivery_zone": zone,
            "delivery_distance_km": round(distance, 1),
            "days_to_delivery": days,
        }

    def _generate_address_features(self, risk_type: str) -> dict:
        """
        Generate address intelligence features.
        FRAUD skews toward forwarding addresses and PO boxes.
        THEFT skews toward apartments.
        """
        if risk_type == "FRAUD":
            address_type = np.random.choice(
                ["PO_BOX", "APARTMENT", "RESIDENTIAL", "BUSINESS"],
                p=[0.35, 0.35, 0.20, 0.10],
            )
            is_forwarding = 1 if np.random.random() < 0.55 else 0
            address_risk = float(np.random.uniform(0.55, 1.0))
            attempt_history = int(np.random.choice([0, 1, 2], p=[0.5, 0.3, 0.2]))

        elif risk_type == "THEFT":
            address_type = np.random.choice(
                ["APARTMENT", "RESIDENTIAL", "PO_BOX", "BUSINESS"],
                p=[0.60, 0.25, 0.10, 0.05],
            )
            is_forwarding = 1 if np.random.random() < 0.15 else 0
            address_risk = float(np.random.uniform(0.45, 0.95))
            attempt_history = int(np.random.choice([0, 1, 2, 3], p=[0.4, 0.3, 0.2, 0.1]))

        elif risk_type == "DAMAGE":
            address_type = np.random.choice(
                self.ADDRESS_TYPES, p=self.ADDRESS_WEIGHTS
            )
            is_forwarding = 0
            address_risk = float(np.random.uniform(0.2, 0.7))
            attempt_history = int(np.random.choice([0, 1, 2, 3, 4], p=[0.3, 0.3, 0.2, 0.1, 0.1]))

        else:  # SAFE
            address_type = np.random.choice(
                self.ADDRESS_TYPES, p=self.ADDRESS_WEIGHTS
            )
            is_forwarding = 1 if np.random.random() < 0.03 else 0
            address_risk = float(np.random.beta(2, 8))  # skews low
            attempt_history = int(np.random.choice([0, 1, 2], p=[0.85, 0.12, 0.03]))

        return {
            "address_type": address_type,
            "address_risk_score": round(address_risk, 3),
            "is_forwarding_address": is_forwarding,
            "delivery_attempt_history": attempt_history,
        }

    def _generate_customer_features(self, risk_type: str) -> dict:
        """
        Generate customer behavior features.
        FRAUD skews toward new accounts with high velocity.
        All risk types show elevated dispute/return rates.
        """
        if risk_type == "FRAUD":
            # New accounts placing many orders quickly
            order_count = int(np.random.choice(
                range(1, 10), p=[0.25, 0.20, 0.15, 0.12, 0.10,
                                  0.07, 0.05, 0.04, 0.02]
            ))
            account_age = int(np.random.uniform(1, 60))
            dispute_rate = float(np.random.uniform(0.25, 1.0))
            return_rate = float(np.random.uniform(0.20, 0.90))
            velocity_7d = int(np.random.uniform(5, 50))
            velocity_30d = int(np.random.uniform(10, 200))

        elif risk_type in ("THEFT", "DAMAGE"):
            order_count = int(np.random.uniform(1, 100))
            account_age = int(np.random.uniform(30, 1000))
            dispute_rate = float(np.random.uniform(0.05, 0.40))
            return_rate = float(np.random.uniform(0.05, 0.35))
            velocity_7d = int(np.random.uniform(1, 10))
            velocity_30d = int(np.random.uniform(2, 40))

        else:  # SAFE
            order_count = int(abs(np.random.lognormal(3.0, 1.0)))
            order_count = max(1, min(order_count, 500))
            account_age = int(np.random.uniform(90, 3650))
            dispute_rate = float(np.random.beta(1, 20))   # skews low
            return_rate = float(np.random.beta(2, 15))    # skews low
            velocity_7d = int(np.random.choice(
                [0, 1, 2, 3, 4, 5],
                p=[0.50, 0.25, 0.12, 0.07, 0.04, 0.02],
            ))
            velocity_30d = velocity_7d + int(np.random.uniform(0, 10))

        return {
            "customer_order_count": order_count,
            "customer_dispute_rate": round(min(dispute_rate, 1.0), 3),
            "customer_return_rate": round(min(return_rate, 1.0), 3),
            "customer_account_age_days": account_age,
            "velocity_7d": min(velocity_7d, 50),
            "velocity_30d": min(velocity_30d, 200),
        }

    def _generate_payment_features(self, risk_type: str) -> dict:
        """
        Generate payment and order context features.
        FRAUD skews heavily toward CRYPTO and COD payment methods.
        """
        if risk_type == "FRAUD":
            payment = np.random.choice(
                ["CRYPTO", "COD", "PAYPAL", "DEBIT_CARD", "CREDIT_CARD"],
                p=[0.35, 0.30, 0.20, 0.10, 0.05],
            )
            # Fraudsters often order late at night
            order_hour = int(np.random.choice(
                list(range(0, 6)) + list(range(22, 24)),
                p=[0.10, 0.10, 0.10, 0.10, 0.10, 0.10,
                   0.20, 0.20],
            ))
        else:
            payment = np.random.choice(
                self.PAYMENT_METHODS, p=self.PAYMENT_WEIGHTS
            )
            # Normal order hours: peak 9am-9pm
            order_hour = int(np.random.choice(
                range(24),
                p=[0.01, 0.01, 0.01, 0.01, 0.02, 0.03,
                   0.05, 0.06, 0.06, 0.06, 0.07, 0.07,
                   0.07, 0.06, 0.06, 0.06, 0.06, 0.06,
                   0.05, 0.05, 0.03, 0.02, 0.01, 0.01],
            ))

        # Weekend and holiday flags
        created_at = self.REFERENCE_DATETIME
        is_weekend = 1 if created_at.weekday() >= 5 else 0
        month = created_at.month
        is_holiday = 1 if month in (11, 12, 1) else 0

        return {
            "payment_method": payment,
            "is_weekend_order": is_weekend,
            "is_holiday_period": is_holiday,
            "order_hour": order_hour,
        }

    # ──────────────────────────────────────────────────────
    # Private: Post Processing
    # ──────────────────────────────────────────────────────

    def _post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply final data quality steps after generation.
        - Ensures velocity_30d >= velocity_7d
        - Clips all values to schema bounds
        - Reorders columns to match schema definition
        """
        # velocity_30d must be >= velocity_7d
        df["velocity_30d"] = df.apply(
            lambda r: max(r["velocity_30d"], r["velocity_7d"]), axis=1
        )

        # Clip all numeric columns to schema bounds
        df["shipment_value"] = df["shipment_value"].clip(5.0, 5000.0)
        df["package_weight"] = df["package_weight"].clip(0.1, 70.0)
        df["carrier_reliability_score"] = df[
            "carrier_reliability_score"
        ].clip(0.0, 1.0)
        df["address_risk_score"] = df["address_risk_score"].clip(0.0, 1.0)
        df["customer_dispute_rate"] = df["customer_dispute_rate"].clip(0.0, 1.0)
        df["customer_return_rate"] = df["customer_return_rate"].clip(0.0, 1.0)
        df["delivery_distance_km"] = df["delivery_distance_km"].clip(1.0, 3000.0)

        # Reorder columns to match schema
        from src.data.schema import COLUMN_NAMES
        df = df[[col for col in COLUMN_NAMES if col in df.columns]]

        return df
