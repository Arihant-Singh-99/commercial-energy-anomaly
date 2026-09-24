"""Deterministic, rule-based explanations for energy anomalies."""

from __future__ import annotations

import numpy as np
import pandas as pd


SIGNIFICANT_DEVIATION_PERCENT = 20.0
SUBSTANTIAL_DEVIATION_PERCENT = 50.0
EXTREME_DEVIATION_PERCENT = 100.0
PERSISTENCE_WINDOW = pd.Timedelta(hours=2)

DEFAULT_RECOMMENDATION_VALUES = {
    "severity": "Unknown",
    "anomaly_category": "Uncategorized",
    "possible_cause": "",
    "recommendation": "",
}

RULES = {
    "after_hours": {
        "category": "After-hours",
        "cause": "HVAC/lighting operating outside expected hours",
        "recommendation": (
            "Check HVAC and lighting schedules and verify that non-essential "
            "systems are not operating after hours."
        ),
    },
    "low_occupancy": {
        "category": "Low-occupancy high consumption",
        "cause": (
            "Energy-intensive equipment or HVAC operation despite low occupancy"
        ),
        "recommendation": (
            "Check HVAC setpoints, lighting controls, and non-essential "
            "equipment operating in low-occupancy areas."
        ),
    },
    "high_consumption": {
        "category": "High consumption",
        "cause": "Energy consumption is significantly above the expected baseline.",
        "recommendation": (
            "Inspect major energy-consuming systems and compare their operating "
            "schedules with expected building usage."
        ),
    },
    "persistent": {
        "category": "Persistent anomaly",
        "cause": (
            "Abnormally high energy consumption is continuing across multiple "
            "observations."
        ),
        "recommendation": (
            "Investigate the affected building systems for sustained HVAC, "
            "lighting, equipment, or scheduling issues."
        ),
    },
    "fallback": {
        "category": "Unusual consumption",
        "cause": "Energy consumption differs significantly from the expected baseline.",
        "recommendation": (
            "Inspect building systems and operating schedules to identify the "
            "source of unusual energy consumption."
        ),
    },
}


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    """Return a numeric series aligned to the input rows."""
    return pd.to_numeric(df[column], errors="coerce")


def _deviation_percent(df: pd.DataFrame) -> pd.Series:
    """Calculate deviation percentage without dividing by a zero baseline."""
    consumption = _numeric_series(df, "consumption_kwh")
    baseline = _numeric_series(df, "baseline_kwh")
    safe_baseline = baseline.replace(0, np.nan)
    return ((consumption - baseline) / safe_baseline) * 100


def detect_persistent_anomalies(
    df: pd.DataFrame,
    window: pd.Timedelta = PERSISTENCE_WINDOW,
) -> pd.Series:
    """Mark anomalies repeated for a meter within a nearby time window."""
    persistent = pd.Series(False, index=df.index, dtype=bool)
    if df.empty:
        return persistent

    anomaly = df["is_anomaly"].fillna(False).astype(bool)
    building = df["building_id"].astype("string")
    timestamp = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    working = pd.DataFrame(
        {
            "building_id": building,
            "timestamp": timestamp,
            "is_anomaly": anomaly,
            "row_index": df.index,
        },
        index=df.index,
    )

    for _, group in working[working["is_anomaly"]].groupby(
        "building_id", sort=False, dropna=False
    ):
        ordered = group.sort_values(["timestamp", "row_index"], na_position="last")
        valid_timestamps = ordered["timestamp"].notna()
        if valid_timestamps.sum() < 2:
            continue

        times = ordered.loc[valid_timestamps, "timestamp"]
        row_indices = ordered.loc[valid_timestamps, "row_index"]
        close_to_previous = times.diff().le(window)
        close_to_next = times.diff(-1).abs().le(window)
        persistent.loc[row_indices[close_to_previous | close_to_next]] = True

    return persistent


def classify_anomaly(
    *,
    after_hours: bool,
    low_occupancy: bool,
    high_consumption: bool,
    persistent: bool,
) -> dict[str, str]:
    """Select the most specific deterministic explanation for one anomaly."""
    if after_hours:
        rule = RULES["after_hours"]
    elif low_occupancy:
        rule = RULES["low_occupancy"]
    elif persistent:
        rule = RULES["persistent"]
    elif high_consumption:
        rule = RULES["high_consumption"]
    else:
        rule = RULES["fallback"]

    return rule.copy()


def calculate_severity(
    *,
    deviation_percent: float,
    anomaly_score: float,
    after_hours: bool,
    very_low_occupancy: bool,
    persistent: bool,
) -> str:
    """Assign severity from deviation strength and contextual evidence."""
    extreme_deviation = (
        pd.notna(deviation_percent)
        and deviation_percent >= EXTREME_DEVIATION_PERCENT
    )
    strong_score = pd.notna(anomaly_score) and anomaly_score >= 3.0

    if persistent or extreme_deviation or (
        after_hours and very_low_occupancy
    ):
        return "HIGH"
    if (
        pd.notna(deviation_percent)
        and deviation_percent >= SUBSTANTIAL_DEVIATION_PERCENT
    ) or strong_score or after_hours:
        return "MEDIUM"
    return "LOW"


def _has_existing_value(series: pd.Series, invalid_values: set[str]) -> pd.Series:
    """Identify non-empty values that may have been supplied by Team B."""
    values = series.astype("string").str.strip()
    return series.notna() & values.ne("") & ~values.isin(invalid_values)


def generate_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic explanations and recommendations to energy data."""
    result = df.copy()

    for column, default in DEFAULT_RECOMMENDATION_VALUES.items():
        if column not in result.columns:
            result[column] = default

    if result.empty:
        result["is_anomaly"] = result.get(
            "is_anomaly", pd.Series(dtype=bool)
        ).astype(bool)
        return result

    anomaly = result["is_anomaly"].fillna(False).astype(bool)
    hour = _numeric_series(result, "hour")
    occupancy = _numeric_series(result, "occupancy_index")
    deviation = _deviation_percent(result)
    score = _numeric_series(result, "anomaly_score")
    persistent = detect_persistent_anomalies(result)

    after_hours = hour.ge(22) | hour.le(6)
    low_occupancy = occupancy.lt(0.15)
    very_low_occupancy = occupancy.lt(0.05)
    significant_consumption = deviation.ge(SIGNIFICANT_DEVIATION_PERCENT)
    substantial_consumption = deviation.ge(SUBSTANTIAL_DEVIATION_PERCENT)

    after_hours_rule = (
        anomaly & after_hours & occupancy.lt(0.10) & significant_consumption
    )
    low_occupancy_rule = anomaly & low_occupancy & significant_consumption
    high_consumption_rule = anomaly & substantial_consumption
    existing_values = {
        column: _has_existing_value(
            result[column],
            {"Unknown", "Uncategorized"},
        )
        for column in DEFAULT_RECOMMENDATION_VALUES
    }

    for position, row_index in enumerate(result.index):
        if not anomaly.iloc[position]:
            continue

        classification = classify_anomaly(
            after_hours=bool(after_hours.iloc[position] and after_hours_rule.iloc[position]),
            low_occupancy=bool(low_occupancy_rule.iloc[position]),
            high_consumption=bool(high_consumption_rule.iloc[position]),
            persistent=bool(persistent.iloc[position]),
        )
        severity = calculate_severity(
            deviation_percent=deviation.iloc[position],
            anomaly_score=score.iloc[position],
            after_hours=bool(after_hours_rule.iloc[position]),
            very_low_occupancy=bool(
                very_low_occupancy.iloc[position]
                and after_hours_rule.iloc[position]
            ),
            persistent=bool(persistent.iloc[position]),
        )

        generated = {
            "severity": severity,
            "anomaly_category": classification["category"],
            "possible_cause": classification["cause"],
            "recommendation": classification["recommendation"],
        }
        for column, value in generated.items():
            if not existing_values[column].iloc[position]:
                result.at[row_index, column] = value

    result["is_anomaly"] = anomaly
    return result