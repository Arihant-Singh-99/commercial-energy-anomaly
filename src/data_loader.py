from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "building_id",
    "timestamp",
    "consumption_kwh",
    "baseline_kwh",
    "anomaly_prediction",
    "anomaly_score",
    "occupancy_index",
    "temp_c",
    "hour",
}

SOURCE_COLUMNS = {
    "meter_id",
    "timestamp",
    "consumption_kwh",
    "roll24_mean_kwh",
    "roll24_std_kwh",
    "anomaly_flag",
    "occupancy_index",
    "temp_c",
    "hour",
}

OPTIONAL_COLUMNS = {
    "severity",
    "anomaly_category",
    "possible_cause",
    "recommendation",
    "potential_savings_kwh",
}

NUMERIC_COLUMNS = {
    "consumption_kwh",
    "baseline_kwh",
    "anomaly_prediction",
    "anomaly_score",
    "occupancy_index",
    "temp_c",
    "hour",
    "potential_savings_kwh",
    "roll24_mean_kwh",
    "roll24_std_kwh",
    "anomaly_flag",
    "high_usage_flag",
    "sensor_health",
    "outage_risk_score",
}


def validate_columns(df: pd.DataFrame) -> None:
    """Validate that all required integration columns are present."""
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))

    if missing:
        raise ValueError(
            "Energy data is missing required columns: "
            + ", ".join(missing)
        )


def adapt_energy_data(df: pd.DataFrame) -> pd.DataFrame:
    """Adapt source data using a temporary dataset-derived anomaly score."""
    adapted = df.copy()

    if REQUIRED_COLUMNS.issubset(adapted.columns):
        return adapted

    missing = sorted(SOURCE_COLUMNS - set(adapted.columns))
    if missing:
        raise ValueError(
            "Energy data is missing required source columns: "
            + ", ".join(missing)
        )

    for column in SOURCE_COLUMNS - {"meter_id", "timestamp"}:
        adapted[column] = pd.to_numeric(adapted[column], errors="coerce")

    adapted["building_id"] = adapted["meter_id"]
    adapted["baseline_kwh"] = adapted["roll24_mean_kwh"]
    adapted["anomaly_prediction"] = adapted["anomaly_flag"].map(
        {1: -1, 0: 1}
    )

    rolling_std = adapted["roll24_std_kwh"].replace(0, np.nan)
    adapted["anomaly_score"] = (
        adapted["consumption_kwh"] - adapted["roll24_mean_kwh"]
    ) / rolling_std

    return adapted


def prepare_energy_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and enrich an energy DataFrame for downstream processing."""
    prepared = adapt_energy_data(df)

    validate_columns(prepared)

    prepared["timestamp"] = pd.to_datetime(
        prepared["timestamp"],
        errors="coerce",
        utc=True,
    )

    for column in NUMERIC_COLUMNS:
        if column in prepared.columns:
            prepared[column] = pd.to_numeric(
                prepared[column],
                errors="coerce",
            )

    optional_defaults = {
        "severity": "Unknown",
        "anomaly_category": "Uncategorized",
        "possible_cause": "",
        "recommendation": "",
        "potential_savings_kwh": 0.0,
    }

    for column, default in optional_defaults.items():
        if column not in prepared.columns:
            prepared[column] = default

    prepared["deviation_kwh"] = (
        prepared["consumption_kwh"] - prepared["baseline_kwh"]
    )

    baseline = prepared["baseline_kwh"].replace(0, np.nan)

    prepared["deviation_percent"] = (
        prepared["deviation_kwh"] / baseline
    ) * 100

    return prepared


def load_energy_data(path: str | Path) -> pd.DataFrame:
    """Load, validate, and prepare energy data from Excel or CSV."""
    data_path = Path(path)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Energy data file was not found: {data_path}"
        )

    try:
        if data_path.suffix.lower() in {".xlsx", ".xls"}:
            raw_data = pd.read_excel(data_path)
        elif data_path.suffix.lower() == ".csv":
            raw_data = pd.read_csv(data_path)
        else:
            raise ValueError(
                "Energy data must be an .xlsx, .xls, or .csv file."
            )
    except Exception as exc:
        raise ValueError(
            f"Unable to read energy data: {exc}"
        ) from exc

    return prepare_energy_data(raw_data)