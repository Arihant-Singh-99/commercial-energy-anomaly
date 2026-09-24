"""Energy impact and potential savings calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    """Return a numeric series aligned to the input rows."""
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def _anomaly_mask(df: pd.DataFrame) -> pd.Series:
    """Use the prepared boolean flag, with prediction-label fallback."""
    if "is_anomaly" in df.columns:
        return df["is_anomaly"].fillna(False).astype(bool)
    if "anomaly_prediction" in df.columns:
        return _numeric_series(df, "anomaly_prediction").eq(-1)
    return pd.Series(False, index=df.index, dtype=bool)


def _calculated_excess(df: pd.DataFrame) -> pd.Series:
    """Calculate positive consumption above baseline without division."""
    consumption = _numeric_series(df, "consumption_kwh")
    baseline = _numeric_series(df, "baseline_kwh")
    return (consumption - baseline).clip(lower=0)


def calculate_event_savings(df: pd.DataFrame) -> pd.DataFrame:
    """Add nonnegative potential savings for anomalous observations.

    Existing positive savings values are preserved so a future Team B output
    can replace this rule without changing the dashboard-facing column.
    Missing or default zero values are calculated from consumption and baseline.
    No rows are removed.
    """
    result = df.copy()
    anomaly = _anomaly_mask(result)
    calculated = _calculated_excess(result).where(anomaly, 0.0)

    if "potential_savings_kwh" in result.columns:
        existing = pd.to_numeric(
            result["potential_savings_kwh"],
            errors="coerce",
        )
        preserve_existing = anomaly & existing.gt(0) & existing.notna()
        result["potential_savings_kwh"] = calculated.where(
            ~preserve_existing,
            existing,
        )
    else:
        result["potential_savings_kwh"] = calculated

    return result


def calculate_total_impact(df: pd.DataFrame) -> dict[str, float | int]:
    """Return aggregate consumption, excess energy, and anomaly metrics."""
    if df.empty:
        return {
            "total_consumption_kwh": 0.0,
            "total_excess_energy_kwh": 0.0,
            "total_potential_savings_kwh": 0.0,
            "anomaly_count": 0,
            "total_observations": 0,
            "anomaly_rate_percent": 0.0,
        }

    event_data = calculate_event_savings(df)
    anomaly = _anomaly_mask(event_data)
    consumption = _numeric_series(event_data, "consumption_kwh")
    excess = _calculated_excess(event_data).where(anomaly, 0.0)
    savings = _numeric_series(event_data, "potential_savings_kwh")

    total_observations = len(event_data)
    anomaly_count = int(anomaly.sum())

    return {
        "total_consumption_kwh": float(consumption.sum(skipna=True)),
        "total_excess_energy_kwh": float(excess.sum(skipna=True)),
        "total_potential_savings_kwh": float(savings.sum(skipna=True)),
        "anomaly_count": anomaly_count,
        "total_observations": total_observations,
        "anomaly_rate_percent": (
            anomaly_count / total_observations * 100
        ),
    }


def calculate_building_impact(df: pd.DataFrame) -> pd.DataFrame:
    """Return consumption and anomaly impact metrics grouped by building."""
    columns = [
        "building_id",
        "consumption_kwh",
        "anomaly_count",
        "excess_energy_kwh",
        "potential_savings_kwh",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)

    event_data = calculate_event_savings(df)
    working = pd.DataFrame(
        {
            "building_id": event_data.get(
                "building_id",
                pd.Series("Unknown", index=event_data.index),
            ),
            "consumption_kwh": _numeric_series(
                event_data,
                "consumption_kwh",
            ),
            "anomaly_count": _anomaly_mask(event_data).astype(int),
            "excess_energy_kwh": _calculated_excess(event_data).where(
                _anomaly_mask(event_data),
                0.0,
            ),
            "potential_savings_kwh": _numeric_series(
                event_data,
                "potential_savings_kwh",
            ),
        }
    )

    return (
        working.groupby("building_id", dropna=False, as_index=False)
        .agg(
            consumption_kwh=("consumption_kwh", "sum"),
            anomaly_count=("anomaly_count", "sum"),
            excess_energy_kwh=("excess_energy_kwh", "sum"),
            potential_savings_kwh=("potential_savings_kwh", "sum"),
        )
    )[columns]