"""Preprocessing helpers for the unified energy data contract."""

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

NUMERIC_COLUMNS = {
    "consumption_kwh",
    "baseline_kwh",
    "anomaly_prediction",
    "anomaly_score",
    "occupancy_index",
    "temp_c",
    "hour",
    "potential_savings_kwh",
}

OPTIONAL_DEFAULTS = {
    "severity": "Unknown",
    "anomaly_category": "Uncategorized",
    "possible_cause": "",
    "recommendation": "",
    "potential_savings_kwh": 0.0,
}


def _validate_columns(df: pd.DataFrame) -> None:
    """Raise a clear error when the unified input contract is incomplete."""
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))

    if missing:
        raise ValueError(
            "Energy data is missing required unified columns: "
            + ", ".join(missing)
        )


def _coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert known numeric fields without removing invalid rows."""
    converted = df.copy()

    for column in NUMERIC_COLUMNS:
        if column in converted.columns:
            converted[column] = pd.to_numeric(
                converted[column],
                errors="coerce",
            )

    return converted


def _normalize_hour(df: pd.DataFrame) -> pd.DataFrame:
    """Keep valid hours and derive missing or invalid ones from timestamps."""
    normalized = df.copy()
    hour = normalized["hour"]
    valid_hour = (
        hour.notna()
        & hour.between(0, 23)
        & hour.mod(1).eq(0)
    )

    normalized.loc[~valid_hour, "hour"] = normalized.loc[~valid_hour, "timestamp"].dt.hour
    return normalized


def _add_derived_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Add row-level deviation and anomaly indicators without predicting."""
    enriched = df.copy()

    enriched["deviation_kwh"] = (
        enriched["consumption_kwh"] - enriched["baseline_kwh"]
    )

    baseline = enriched["baseline_kwh"].replace(0, np.nan)
    enriched["deviation_percent"] = (
        enriched["deviation_kwh"] / baseline
    ) * 100

    enriched["is_anomaly"] = enriched["anomaly_prediction"].eq(-1)
    return enriched


def preprocess_energy_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a unified energy DataFrame for downstream dashboard use.

    This function preserves row count and does not calculate or replace any
    machine-learning predictions. Missing values remain missing when they
    cannot be derived safely, including initial rolling anomaly scores.
    """
    _validate_columns(df)

    processed = df.copy()

    processed["timestamp"] = pd.to_datetime(
        processed["timestamp"],
        errors="coerce",
        utc=True,
    )
    processed = _coerce_numeric_columns(processed)

    for column, default in OPTIONAL_DEFAULTS.items():
        if column not in processed.columns:
            processed[column] = default

    processed["anomaly_prediction"] = processed["anomaly_prediction"].where(
        processed["anomaly_prediction"].isin([-1, 1])
    )
    processed = _normalize_hour(processed)

    return _add_derived_fields(processed)