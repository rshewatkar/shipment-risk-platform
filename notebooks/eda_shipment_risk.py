"""
Exploratory Data Analysis — Shipment Risk Platform
===================================================
Generates 15+ business-focused visualisations answering:

    1.  What is the overall risk distribution?
    2.  Which carriers have the highest risk rates?
    3.  How does shipment value relate to risk?
    4.  Which address types are riskiest?
    5.  How does payment method signal fraud?
    6.  What does customer velocity look like for fraud?
    7.  How do delivery zones affect risk?
    8.  What is the dispute rate distribution by risk?
    9.  How does account age relate to fraud?
    10. What are the temporal patterns (hour/weekend)?
    11. Feature correlation heatmap
    12. Risk rate by carrier AND zone combined
    13. Shipment value distribution by risk type
    14. Forwarding address fraud signal
    15. Feature importance preview (value counts)

Usage:
    python notebooks/eda_shipment_risk.py
"""

from pathlib import Path

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Setup ──────────────────────────────────────────────────
DATA_PATH = "data/raw/shipments_raw.csv"
OUTPUT_DIR = Path("reports/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"  Shape     : {df.shape}")
print(f"  Risk rate : {df['risk_label'].mean():.1%}")
print()

# Colour palette — consistent across all charts
COLORS = {
    "safe":     "#1D9E75",
    "risky":    "#D85A30",
    "neutral":  "#534AB7",
    "accent":   "#F5A623",
    "bg":       "#FAFAF8",
    "grid":     "#EBEBEB",
}

RISK_COLORS = [COLORS["safe"], COLORS["risky"]]


def save_fig(fig: go.Figure, filename: str) -> None:
    """Save figure as HTML for interactive viewing."""
    path = OUTPUT_DIR / f"{filename}.html"
    fig.write_html(str(path))
    print(f"  Saved: {path}")


def style_fig(fig: go.Figure, title: str) -> go.Figure:
    """Apply consistent styling to all figures."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#1A1A18")),
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["bg"],
        font=dict(family="Inter, Arial, sans-serif", size=12),
        margin=dict(t=60, b=40, l=40, r=40),
    )
    fig.update_xaxes(gridcolor=COLORS["grid"], showgrid=True)
    fig.update_yaxes(gridcolor=COLORS["grid"], showgrid=True)
    return fig


# ══════════════════════════════════════════════════════════
# Chart 1 — Overall Risk Distribution
# ══════════════════════════════════════════════════════════
print("Chart 1: Overall risk distribution")

risk_counts = df["risk_label"].value_counts().reset_index()
risk_counts.columns = ["risk_label", "count"]
risk_counts["label"] = risk_counts["risk_label"].map({0: "Safe", 1: "Risky"})
risk_counts["pct"] = (risk_counts["count"] / len(df) * 100).round(1)

fig1 = px.bar(
    risk_counts,
    x="label",
    y="count",
    color="label",
    color_discrete_map={"Safe": COLORS["safe"], "Risky": COLORS["risky"]},
    text=risk_counts["pct"].astype(str) + "%",
)
fig1 = style_fig(
    fig1,
    "Chart 1 — Overall Risk Distribution<br>"
    "<sup>Class imbalance check — expected ~12% risky</sup>",
)
fig1.update_traces(textposition="outside")
save_fig(fig1, "01_risk_distribution")


# ══════════════════════════════════════════════════════════
# Chart 2 — Risk Rate by Carrier
# ══════════════════════════════════════════════════════════
print("Chart 2: Risk rate by carrier")

carrier_risk = (
    df.groupby("carrier_name")["risk_label"]
    .agg(["mean", "count"])
    .reset_index()
)
carrier_risk.columns = ["carrier", "risk_rate", "shipment_count"]
carrier_risk["risk_pct"] = (carrier_risk["risk_rate"] * 100).round(1)
carrier_risk = carrier_risk.sort_values("risk_rate", ascending=False)

fig2 = px.bar(
    carrier_risk,
    x="carrier",
    y="risk_pct",
    color="risk_pct",
    color_continuous_scale=["#1D9E75", "#F5A623", "#D85A30"],
    text=carrier_risk["risk_pct"].astype(str) + "%",
)
fig2 = style_fig(
    fig2,
    "Chart 2 — Risk Rate by Carrier<br>"
    "<sup>Which carrier has the most risky shipments?</sup>",
)
fig2.update_traces(textposition="outside")
save_fig(fig2, "02_risk_by_carrier")


# ══════════════════════════════════════════════════════════
# Chart 3 — Shipment Value Distribution by Risk Label
# ══════════════════════════════════════════════════════════
print("Chart 3: Shipment value distribution")

fig3 = px.histogram(
    df,
    x="shipment_value",
    color="risk_label",
    color_discrete_sequence=RISK_COLORS,
    nbins=60,
    barmode="overlay",
    opacity=0.7,
    labels={"risk_label": "Risk", "shipment_value": "Shipment Value (USD)"},
)
fig3 = style_fig(
    fig3,
    "Chart 3 — Shipment Value Distribution by Risk<br>"
    "<sup>Do risky shipments cluster at higher values?</sup>",
)
fig3.update_layout(
    legend=dict(
        title="Risk Label",
        itemsizing="constant",
    )
)
save_fig(fig3, "03_value_distribution")


# ══════════════════════════════════════════════════════════
# Chart 4 — Risk Rate by Address Type
# ══════════════════════════════════════════════════════════
print("Chart 4: Risk by address type")

addr_risk = (
    df.groupby("address_type")["risk_label"]
    .agg(["mean", "count"])
    .reset_index()
)
addr_risk.columns = ["address_type", "risk_rate", "count"]
addr_risk["risk_pct"] = (addr_risk["risk_rate"] * 100).round(1)
addr_risk = addr_risk.sort_values("risk_rate", ascending=True)

fig4 = px.bar(
    addr_risk,
    x="risk_pct",
    y="address_type",
    orientation="h",
    color="risk_pct",
    color_continuous_scale=["#1D9E75", "#F5A623", "#D85A30"],
    text=addr_risk["risk_pct"].astype(str) + "%",
)
fig4 = style_fig(
    fig4,
    "Chart 4 — Risk Rate by Address Type<br>"
    "<sup>PO_BOX and APARTMENT expected highest risk</sup>",
)
save_fig(fig4, "04_risk_by_address_type")


# ══════════════════════════════════════════════════════════
# Chart 5 — Risk Rate by Payment Method
# ══════════════════════════════════════════════════════════
print("Chart 5: Risk by payment method")

pay_risk = (
    df.groupby("payment_method")["risk_label"]
    .agg(["mean", "count"])
    .reset_index()
)
pay_risk.columns = ["payment_method", "risk_rate", "count"]
pay_risk["risk_pct"] = (pay_risk["risk_rate"] * 100).round(1)
pay_risk = pay_risk.sort_values("risk_rate", ascending=False)

fig5 = px.bar(
    pay_risk,
    x="payment_method",
    y="risk_pct",
    color="risk_pct",
    color_continuous_scale=["#1D9E75", "#F5A623", "#D85A30"],
    text=pay_risk["risk_pct"].astype(str) + "%",
)
fig5 = style_fig(
    fig5,
    "Chart 5 — Risk Rate by Payment Method<br>"
    "<sup>CRYPTO and COD expected as top fraud signals</sup>",
)
fig5.update_traces(textposition="outside")
save_fig(fig5, "05_risk_by_payment")


# ══════════════════════════════════════════════════════════
# Chart 6 — Customer Dispute Rate: Safe vs Risky
# ══════════════════════════════════════════════════════════
print("Chart 6: Dispute rate distribution")

fig6 = px.box(
    df,
    x="risk_label",
    y="customer_dispute_rate",
    color="risk_label",
    color_discrete_sequence=RISK_COLORS,
    labels={
        "risk_label": "Risk Label",
        "customer_dispute_rate": "Customer Dispute Rate",
    },
    points="outliers",
)
fig6 = style_fig(
    fig6,
    "Chart 6 — Customer Dispute Rate: Safe vs Risky<br>"
    "<sup>Strong behavioral signal — risky customers dispute more</sup>",
)
save_fig(fig6, "06_dispute_rate_boxplot")


# ══════════════════════════════════════════════════════════
# Chart 7 — Velocity 7-Day: Safe vs Risky
# ══════════════════════════════════════════════════════════
print("Chart 7: Order velocity signal")

fig7 = px.histogram(
    df,
    x="velocity_7d",
    color="risk_label",
    color_discrete_sequence=RISK_COLORS,
    barmode="overlay",
    opacity=0.75,
    nbins=30,
    labels={
        "velocity_7d": "Orders in Last 7 Days",
        "risk_label": "Risk",
    },
)
fig7 = style_fig(
    fig7,
    "Chart 7 — 7-Day Order Velocity: Safe vs Risky<br>"
    "<sup>Burst ordering is a fraud pattern</sup>",
)
save_fig(fig7, "07_velocity_7d")


# ══════════════════════════════════════════════════════════
# Chart 8 — Risk Rate by Delivery Zone
# ══════════════════════════════════════════════════════════
print("Chart 8: Risk by delivery zone")

zone_risk = (
    df.groupby("delivery_zone")["risk_label"]
    .agg(["mean", "count"])
    .reset_index()
)
zone_risk.columns = ["zone", "risk_rate", "count"]
zone_risk["risk_pct"] = (zone_risk["risk_rate"] * 100).round(1)

zone_order = ["URBAN", "SUBURBAN", "RURAL", "REMOTE"]
zone_risk["zone"] = pd.Categorical(zone_risk["zone"], categories=zone_order, ordered=True)
zone_risk = zone_risk.sort_values("zone")

fig8 = px.bar(
    zone_risk,
    x="zone",
    y="risk_pct",
    color="risk_pct",
    color_continuous_scale=["#1D9E75", "#F5A623", "#D85A30"],
    text=zone_risk["risk_pct"].astype(str) + "%",
)
fig8 = style_fig(
    fig8,
    "Chart 8 — Risk Rate by Delivery Zone<br>"
    "<sup>Does remote delivery correlate with higher damage rates?</sup>",
)
save_fig(fig8, "08_risk_by_zone")


# ══════════════════════════════════════════════════════════
# Chart 9 — Account Age Distribution: Safe vs Risky
# ══════════════════════════════════════════════════════════
print("Chart 9: Account age vs risk")

fig9 = px.histogram(
    df,
    x="customer_account_age_days",
    color="risk_label",
    color_discrete_sequence=RISK_COLORS,
    barmode="overlay",
    opacity=0.75,
    nbins=50,
    labels={
        "customer_account_age_days": "Account Age (Days)",
        "risk_label": "Risk",
    },
)
fig9 = style_fig(
    fig9,
    "Chart 9 — Customer Account Age: Safe vs Risky<br>"
    "<sup>New accounts are a key fraud indicator</sup>",
)
save_fig(fig9, "09_account_age")


# ══════════════════════════════════════════════════════════
# Chart 10 — Order Hour Heatmap
# ══════════════════════════════════════════════════════════
print("Chart 10: Order hour risk pattern")

hour_risk = (
    df.groupby("order_hour")["risk_label"]
    .agg(["mean", "count"])
    .reset_index()
)
hour_risk.columns = ["hour", "risk_rate", "count"]
hour_risk["risk_pct"] = (hour_risk["risk_rate"] * 100).round(1)

fig10 = px.bar(
    hour_risk,
    x="hour",
    y="risk_pct",
    color="risk_pct",
    color_continuous_scale=["#1D9E75", "#F5A623", "#D85A30"],
    labels={"hour": "Order Hour (0-23)", "risk_pct": "Risk Rate (%)"},
)
fig10 = style_fig(
    fig10,
    "Chart 10 — Risk Rate by Order Hour<br>"
    "<sup>Late night orders (0-5am) signal fraud</sup>",
)
save_fig(fig10, "10_order_hour_risk")


# ══════════════════════════════════════════════════════════
# Chart 11 — Forwarding Address Fraud Signal
# ══════════════════════════════════════════════════════════
print("Chart 11: Forwarding address signal")

fwd_risk = (
    df.groupby("is_forwarding_address")["risk_label"]
    .agg(["mean", "count"])
    .reset_index()
)
fwd_risk.columns = ["is_forwarding", "risk_rate", "count"]
fwd_risk["label"] = fwd_risk["is_forwarding"].map(
    {0: "Standard Address", 1: "Forwarding Address"}
)
fwd_risk["risk_pct"] = (fwd_risk["risk_rate"] * 100).round(1)

fig11 = px.bar(
    fwd_risk,
    x="label",
    y="risk_pct",
    color="label",
    color_discrete_map={
        "Standard Address": COLORS["safe"],
        "Forwarding Address": COLORS["risky"],
    },
    text=fwd_risk["risk_pct"].astype(str) + "%",
)
fig11 = style_fig(
    fig11,
    "Chart 11 — Forwarding Address as Fraud Signal<br>"
    "<sup>Expected: forwarding addresses have 3-5x higher risk rate</sup>",
)
fig11.update_traces(textposition="outside")
save_fig(fig11, "11_forwarding_address")


# ══════════════════════════════════════════════════════════
# Chart 12 — Carrier Reliability Score Distribution
# ══════════════════════════════════════════════════════════
print("Chart 12: Carrier reliability distribution")

fig12 = px.box(
    df,
    x="carrier_name",
    y="carrier_reliability_score",
    color="carrier_name",
    points="outliers",
    labels={
        "carrier_name": "Carrier",
        "carrier_reliability_score": "Reliability Score",
    },
)
fig12 = style_fig(
    fig12,
    "Chart 12 — Carrier Reliability Score Distribution<br>"
    "<sup>Spread shows variance across routes for each carrier</sup>",
)
save_fig(fig12, "12_carrier_reliability")


# ══════════════════════════════════════════════════════════
# Chart 13 — Risk Rate: Carrier × Zone Heatmap
# ══════════════════════════════════════════════════════════
print("Chart 13: Carrier x zone heatmap")

cross_risk = (
    df.groupby(["carrier_name", "delivery_zone"])["risk_label"]
    .mean()
    .reset_index()
)
cross_pivot = cross_risk.pivot(
    index="carrier_name",
    columns="delivery_zone",
    values="risk_label",
)

fig13 = px.imshow(
    cross_pivot,
    color_continuous_scale=["#1D9E75", "#F5A623", "#D85A30"],
    aspect="auto",
    text_auto=".1%",
    labels=dict(color="Risk Rate"),
)
fig13 = style_fig(
    fig13,
    "Chart 13 — Risk Rate Heatmap: Carrier × Zone<br>"
    "<sup>Identifies high-risk carrier and zone combinations</sup>",
)
save_fig(fig13, "13_carrier_zone_heatmap")


# ══════════════════════════════════════════════════════════
# Chart 14 — Feature Correlation Heatmap
# ══════════════════════════════════════════════════════════
print("Chart 14: Feature correlation heatmap")

from src.data.schema import NUMERIC_FEATURES

numeric_cols = [c for c in NUMERIC_FEATURES if c in df.columns] + ["risk_label"]
corr_matrix = df[numeric_cols].corr()

fig14 = px.imshow(
    corr_matrix,
    color_continuous_scale="RdBu_r",
    aspect="auto",
    text_auto=".2f",
    zmin=-1,
    zmax=1,
)
fig14 = style_fig(
    fig14,
    "Chart 14 — Feature Correlation Heatmap<br>"
    "<sup>Identifies multicollinearity and strongest risk correlations</sup>",
)
fig14.update_layout(
    width=900,
    height=700,
)
save_fig(fig14, "14_correlation_heatmap")


# ══════════════════════════════════════════════════════════
# Chart 15 — Address Risk Score: Safe vs Risky
# ══════════════════════════════════════════════════════════
print("Chart 15: Address risk score distribution")

fig15 = px.histogram(
    df,
    x="address_risk_score",
    color="risk_label",
    color_discrete_sequence=RISK_COLORS,
    barmode="overlay",
    opacity=0.75,
    nbins=40,
    labels={
        "address_risk_score": "Address Risk Score",
        "risk_label": "Risk Label",
    },
)
fig15 = style_fig(
    fig15,
    "Chart 15 — Address Risk Score Distribution: Safe vs Risky<br>"
    "<sup>Expected: risky shipments cluster at higher address risk scores</sup>",
)
save_fig(fig15, "15_address_risk_score")


# ══════════════════════════════════════════════════════════
# Chart 16 — High Value Flag vs Risk
# ══════════════════════════════════════════════════════════
print("Chart 16: High value shipment risk")

hv_risk = (
    df.groupby("is_high_value")["risk_label"]
    .agg(["mean", "count"])
    .reset_index()
)
hv_risk["label"] = hv_risk["is_high_value"].map(
    {0: "Standard Value (< $500)", 1: "High Value (> $500)"}
)
hv_risk["risk_pct"] = (hv_risk["mean"] * 100).round(1)

fig16 = px.bar(
    hv_risk,
    x="label",
    y="risk_pct",
    color="label",
    color_discrete_map={
        "Standard Value (< $500)": COLORS["safe"],
        "High Value (> $500)": COLORS["risky"],
    },
    text=hv_risk["risk_pct"].astype(str) + "%",
)
fig16 = style_fig(
    fig16,
    "Chart 16 — High Value Shipments vs Risk Rate<br>"
    "<sup>High value items are theft and fraud magnets</sup>",
)
fig16.update_traces(textposition="outside")
save_fig(fig16, "16_high_value_risk")


# ══════════════════════════════════════════════════════════
# Summary Statistics Print
# ══════════════════════════════════════════════════════════
print()
print("=" * 55)
print("EDA SUMMARY — KEY BUSINESS INSIGHTS")
print("=" * 55)

print(f"\n  Dataset shape     : {df.shape}")
print(f"  Overall risk rate : {df['risk_label'].mean():.1%}")
print(f"  Total risky       : {df['risk_label'].sum():,}")
print(f"  Total safe        : {(df['risk_label']==0).sum():,}")

print("\n  Top 3 riskiest carriers:")
print(carrier_risk[["carrier", "risk_pct"]].head(3).to_string(index=False))

print("\n  Risk rate by address type:")
print(addr_risk[["address_type", "risk_pct"]].to_string(index=False))

print("\n  Risk rate by payment method:")
print(pay_risk[["payment_method", "risk_pct"]].head(3).to_string(index=False))

fwd_high = fwd_risk[fwd_risk["is_forwarding"] == 1]["risk_pct"].values[0]
fwd_low = fwd_risk[fwd_risk["is_forwarding"] == 0]["risk_pct"].values[0]
print(f"\n  Forwarding address risk : {fwd_high}%")
print(f"  Standard address risk   : {fwd_low}%")
print(f"  Risk multiplier         : {fwd_high/fwd_low:.1f}x")

print("\n  Top correlations with risk_label:")
risk_corr = (
    corr_matrix["risk_label"]
    .drop("risk_label")
    .abs()
    .sort_values(ascending=False)
    .head(8)
)
for feat, corr_val in risk_corr.items():
    print(f"    {feat:<35} {corr_val:.3f}")

print("\n" + "=" * 55)
print(f"  All 16 charts saved to: reports/figures/")
print("=" * 55)