# Shipment Risk Platform — Dataset Schema

## Target Variable

| Column | Type | Values | Business Meaning |
|---|---|---|---|
| risk_label | int | 0 = safe, 1 = risky | Ground truth for model training. A shipment is risky if it results in loss, theft, fraud, or damage. |

---

## Feature Groups

### Group 1 — Shipment Identity
These columns uniquely identify a shipment. Not used as model features.

| Column | Type | Example | Notes |
|---|---|---|---|
| shipment_id | string | SHP-2024-00001 | Unique ID, format SHP-YYYY-NNNNN |
| created_at | datetime | 2024-03-15 14:32:00 | Timestamp of label creation |

---

### Group 2 — Shipment Value and Package
Higher value shipments attract more theft. Unusual weight/value ratios signal fraud.

| Column | Type | Range | Business Meaning |
|---|---|---|---|
| shipment_value | float | 5.0 – 5000.0 | Declared shipment value in USD |
| package_weight | float | 0.1 – 70.0 | Weight in kg |
| shipment_type | string | STANDARD, EXPRESS, OVERNIGHT, ECONOMY | Express/Overnight to risky addresses = higher risk |
| is_high_value | int | 0 or 1 | 1 if shipment_value > 500 |
| value_to_weight_ratio | float | derived | Unusually high ratio may indicate fraud |

---

### Group 3 — Carrier and Delivery
Different carriers have different reliability profiles on different routes.

| Column | Type | Values | Business Meaning |
|---|---|---|---|
| carrier_name | string | FedEx, UPS, USPS, DHL, OnTrac | Carrier handling the shipment |
| carrier_reliability_score | float | 0.0 – 1.0 | Historical on-time + no-damage rate for this carrier on this route |
| delivery_zone | string | URBAN, SUBURBAN, RURAL, REMOTE | Rural and remote zones have higher loss rates |
| delivery_distance_km | float | 1.0 – 3000.0 | Longer routes = more handling = more risk |
| days_to_delivery | int | 1 – 14 | Estimated days. Longer = more exposure |

---

### Group 4 — Address Intelligence
Address type and history are among the strongest predictors of risk.

| Column | Type | Values | Business Meaning |
|---|---|---|---|
| address_type | string | RESIDENTIAL, APARTMENT, BUSINESS, PO_BOX | Apartments have higher theft rates. PO boxes flag fraud. |
| address_risk_score | float | 0.0 – 1.0 | Historical risk score for this zip/area |
| is_forwarding_address | int | 0 or 1 | Forwarding addresses are a strong fraud signal |
| delivery_attempt_history | int | 0 – 5 | Prior failed delivery attempts at this address |

---

### Group 5 — Customer Behavior
Customer history is the strongest combined predictor.

| Column | Type | Range | Business Meaning |
|---|---|---|---|
| customer_id | string | CUST-NNNNN | Customer identifier |
| customer_order_count | int | 1 – 500 | Total historical orders. New customers = higher risk. |
| customer_dispute_rate | float | 0.0 – 1.0 | Fraction of orders that resulted in disputes |
| customer_return_rate | float | 0.0 – 1.0 | Fraction of orders returned |
| customer_account_age_days | int | 1 – 3650 | New accounts are higher risk |
| velocity_7d | int | 0 – 50 | Orders placed in last 7 days. High velocity = possible fraud. |
| velocity_30d | int | 0 – 200 | Orders placed in last 30 days |

---

### Group 6 — Payment and Order Context
Payment method and order timing provide fraud signals.

| Column | Type | Values | Business Meaning |
|---|---|---|---|
| payment_method | string | CREDIT_CARD, DEBIT_CARD, PAYPAL, CRYPTO, COD | Crypto and COD have higher fraud rates |
| is_weekend_order | int | 0 or 1 | Weekend orders have slightly higher risk |
| is_holiday_period | int | 0 or 1 | Holiday periods see spikes in theft |
| order_hour | int | 0 – 23 | Late night orders are a mild fraud signal |

---

## Risk Label Logic

A shipment gets risk_label = 1 if ANY of the following:
- Package reported stolen or lost
- Customer filed a fraud dispute
- Package damaged on delivery
- Delivery failed 3+ times
- Chargeback initiated

Base rates in our synthetic dataset:
- Overall risk rate: ~12% (combining all risk types)
- Fraud: 4%
- Loss/theft: 3%
- Damage: 5%

---

## Feature Importance Hypothesis (before modeling)

Expected top features based on domain knowledge:
1. address_risk_score — strongest single predictor
2. customer_dispute_rate — strong behavioral signal
3. is_forwarding_address — binary fraud flag
4. payment_method (CRYPTO/COD) — fraud proxy
5. velocity_7d — burst ordering pattern
6. shipment_value — theft attractiveness
7. carrier_reliability_score — delivery success history
8. customer_account_age_days — new account risk
