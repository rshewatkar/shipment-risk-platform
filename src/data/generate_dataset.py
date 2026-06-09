"""
Dataset generation script.
Run this once to produce data/raw/shipments_raw.csv

Usage:
    python -m src.data.generate_dataset
"""
from src.data.generator import ShipmentDataGenerator
from src.data.validation import validate_dataset
from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Generate, validate, and save the shipment dataset."""

    cfg = config["data"]

    # ── Step 1: Generate ──────────────────────────────────
    generator = ShipmentDataGenerator(
        n_samples=cfg["n_samples"],
        seed=cfg["random_state"],
    )
    df = generator.generate()

    # ── Step 2: Validate ──────────────────────────────────
    logger.info("running validation on generated dataset...")
    result = validate_dataset(df)

    if result["warnings"]:
        for w in result["warnings"]:
            logger.warning("validation warning", message=w)

    if not result["success"]:
        logger.error("dataset failed validation", errors=result["errors"])
        raise ValueError(
            f"Generated dataset failed validation:\n"
            + "\n".join(result["errors"])
        )

    # ── Step 3: Print summary ─────────────────────────────
    stats = result["stats"]
    logger.info(
        "dataset summary",
        n_rows=stats["n_rows"],
        n_columns=stats["n_columns"],
        risk_rate=f"{stats['risk_rate']:.1%}",
    )

    print("\n" + "=" * 50)
    print("DATASET GENERATION COMPLETE")
    print("=" * 50)
    print(f"  Total records   : {stats['n_rows']:,}")
    print(f"  Total features  : {stats['n_columns']}")
    print(f"  Risk rate       : {stats['risk_rate']:.1%}")
    print(f"  Risky shipments : {int(df['risk_label'].sum()):,}")
    print(f"  Safe shipments  : {int((df['risk_label'] == 0).sum()):,}")
    print("=" * 50)

    # ── Step 4: Save ──────────────────────────────────────
    generator.save(df, cfg["raw_path"])
    print(f"\n  Saved to: {cfg['raw_path']}")
    print("  Run EDA next: jupyter notebook notebooks/")


if __name__ == "__main__":
    main()