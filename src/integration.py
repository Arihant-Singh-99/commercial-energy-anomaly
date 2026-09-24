"""Integration boundary between model outputs and the Streamlit dashboard."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# Contract supplied by the final ML and recommendation pipeline. The dashboard
# consumes these names and does not depend on how predictions are produced.
FINAL_DASHBOARD_COLUMNS = (
    "building_id",
    "timestamp",
    "consumption_kwh",
    "baseline_kwh",
    "anomaly_prediction",
    "anomaly_score",
    "occupancy_index",
    "temp_c",
    "hour",
    "severity",
    "anomaly_category",
    "possible_cause",
    "recommendation",
    "potential_savings_kwh",
)

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


def validate_dashboard_data(df: pd.DataFrame) -> None:
    """Validate the final ML/recommendation-to-dashboard contract."""
    missing = sorted(set(FINAL_DASHBOARD_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(
            "Final dashboard data is missing required columns: "
            + ", ".join(missing)
        )


def prepare_dashboard_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare a final ML/recommendation DataFrame for dashboard consumption.

    This is the integration function for Team A and Team B output. It does
    not generate predictions or recommendations. It validates the shared
    column contract, normalizes display types, and adds UI-derived fields
    used by the existing dashboard such as ``is_anomaly`` and deviations.
    Original columns, including optional source context, are preserved.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Final dashboard data must be a pandas DataFrame.")

    validate_dashboard_data(df)
    prepared = df.copy()
    prepared["timestamp"] = pd.to_datetime(
        prepared["timestamp"],
        errors="coerce",
        utc=True,
    )

    for column in NUMERIC_COLUMNS:
        prepared[column] = pd.to_numeric(
            prepared[column],
            errors="coerce",
        )

    # These are deterministic presentation fields, not new ML predictions.
    prepared["is_anomaly"] = prepared["anomaly_prediction"].eq(-1)
    prepared["deviation_kwh"] = (
        prepared["consumption_kwh"] - prepared["baseline_kwh"]
    )
    safe_baseline = prepared["baseline_kwh"].replace(0, np.nan)
    prepared["deviation_percent"] = (
        prepared["deviation_kwh"] / safe_baseline
    ) * 100

    return prepared


def load_dashboard_data(source: str | Path | pd.DataFrame) -> pd.DataFrame:
    """Load final output or use the Excel pipeline as a development fallback.

    Passing a DataFrame is the intended production integration path. Passing
    the current workbook keeps local dashboard development working until the
    final ML and recommendation pipeline is available.
    """
    if isinstance(source, pd.DataFrame):
        return prepare_dashboard_data(source)

    from data_loader import load_energy_data
    from impact import calculate_event_savings
    from preprocessing import preprocess_energy_data
    from recommendations import generate_recommendations

    loaded = load_energy_data(source)
    preprocessed = preprocess_energy_data(loaded)
    recommended = generate_recommendations(preprocessed)
    fallback_output = calculate_event_savings(recommended)
    return prepare_dashboard_data(fallback_output)