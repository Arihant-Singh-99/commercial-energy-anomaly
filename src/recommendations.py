import pandas as pd
from typing import Dict, Any


# ============================================================
# HELPERS
# ============================================================

def _is_anomaly(value: Any) -> bool:
    """
    Isolation Forest convention:
        -1 = anomaly
         1 = normal

    Only -1 is treated as an anomaly.
    """
    if pd.isna(value):
        return False

    try:
        return float(value) == -1
    except (TypeError, ValueError):
        return str(value).strip() == "-1"


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert a value to float."""
    if value is None:
        return default

    try:
        number = float(value)

        if pd.isna(number):
            return default

        return number

    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 12) -> int:
    """Safely convert a value to int."""
    if value is None:
        return default

    try:
        number = float(value)

        if pd.isna(number):
            return default

        return int(number)

    except (TypeError, ValueError):
        return default


# ============================================================
# PERSISTENCE DETECTION
# ============================================================

def determine_persistence(
    df: pd.DataFrame,
    anomaly_col: str = "anomaly_prediction",
    threshold_consecutive: int = 3
) -> pd.Series:
    """
    Detect persistent anomalies.

    An anomaly is considered persistent when it remains anomalous
    for at least `threshold_consecutive` consecutive observations.

    Isolation Forest convention:
        -1 = anomaly
         1 = normal

    If building_id exists, persistence is calculated independently
    for each building.
    """

    if df.empty:
        return pd.Series(False, index=df.index, dtype=bool)

    if anomaly_col not in df.columns:
        return pd.Series(False, index=df.index, dtype=bool)

    # Boolean anomaly mask.
    is_anomaly = df[anomaly_col].apply(_is_anomaly).astype(bool)

    # --------------------------------------------------------
    # If building_id exists, calculate persistence separately
    # for each building so anomalies from different buildings
    # are never considered consecutive.
    # --------------------------------------------------------

    if "building_id" in df.columns:

        result = pd.Series(
            False,
            index=df.index,
            dtype=bool
        )

        for _, group in df.groupby(
            "building_id",
            sort=False,
            dropna=False
        ):

            group_anomaly = is_anomaly.loc[group.index]

            groups = (
                group_anomaly != group_anomaly.shift()
            ).cumsum()

            consecutive_counts = (
                group_anomaly
                .groupby(groups)
                .cumsum()
            )

            persistent = (
                (consecutive_counts >= threshold_consecutive)
                & group_anomaly
            )

            result.loc[group.index] = persistent

        return result

    # --------------------------------------------------------
    # Fallback when building_id doesn't exist.
    # --------------------------------------------------------

    groups = (
        is_anomaly != is_anomaly.shift()
    ).cumsum()

    consecutive_counts = (
        is_anomaly
        .groupby(groups)
        .cumsum()
    )

    return (
        (consecutive_counts >= threshold_consecutive)
        & is_anomaly
    )


# ============================================================
# SINGLE-ROW RECOMMENDATION
# ============================================================

def generate_recommendation(
    row: Dict[str, Any],
    is_persistent: bool = False
) -> Dict[str, Any]:
    """
    Generate a recommendation for one observation.
    """

    pred = row.get("anomaly_prediction")

    # IMPORTANT:
    # Isolation Forest:
    # -1 = anomaly
    #  1 = normal
    is_anomaly = _is_anomaly(pred)

    # --------------------------------------------------------
    # NORMAL OPERATION
    # --------------------------------------------------------

    if not is_anomaly:

        return {
            "severity": "None",
            "anomaly_category": "Normal Operation",
            "possible_cause": (
                "Consumption aligns with baseline expectations."
            ),
            "recommendation": (
                "No action required."
            )
        }

    # --------------------------------------------------------
    # SAFE INPUT VALUES
    # --------------------------------------------------------

    actual = _safe_float(
        row.get("consumption_kwh"),
        0.0
    )

    baseline = _safe_float(
        row.get("baseline_kwh"),
        actual
    )

    occupancy = _safe_float(
        row.get("occupancy_index"),
        1.0
    )

    temp = _safe_float(
        row.get("temp_c"),
        22.0
    )

    hour = _safe_int(
        row.get("hour"),
        12
    )

    # Keep hour within a sensible range.
    hour = max(0, min(hour, 23))

    # --------------------------------------------------------
    # ENERGY DEVIATION
    # --------------------------------------------------------

    excess = max(
        actual - baseline,
        0.0
    )

    if baseline > 0:
        deviation_ratio = excess / baseline
    else:
        deviation_ratio = 0.0

    # --------------------------------------------------------
    # CONTEXT CONDITIONS
    # --------------------------------------------------------

    is_after_hours = (
        hour >= 20 or hour <= 6
    )

    is_low_occupancy = (
        occupancy < 0.25
    )

    is_high_temp = (
        temp >= 28.0
    )

    # --------------------------------------------------------
    # SEVERITY SCORE
    # --------------------------------------------------------

    score = 0

    # Energy deviation.
    if deviation_ratio < 0.20:
        score += 1

    elif deviation_ratio < 0.50:
        score += 2

    else:
        score += 3

    # Operating context.
    if is_after_hours and is_low_occupancy:
        score += 2

    elif is_after_hours or is_low_occupancy:
        score += 1

    # Persistence.
    if is_persistent:
        score += 2

    # Convert score to severity.
    if score <= 2:
        severity = "Low"

    elif score <= 4:
        severity = "Medium"

    else:
        severity = "High"

    # --------------------------------------------------------
    # RECOMMENDATION CATEGORIES
    # --------------------------------------------------------

    if is_persistent:

        category = (
            "Persistent Baseload Deviation"
        )

        cause = (
            "Continuous abnormal consumption sustained across "
            "multiple hours, indicating potential equipment "
            "override or failure to set back."
        )

        action = (
            "Conduct a walk-through or check BMS logs to verify "
            "whether equipment, pumps, or air handlers are "
            "running in manual/hand mode."
        )

    elif is_after_hours and is_low_occupancy:

        category = (
            "After-Hours Energy Use"
        )

        cause = (
            "Elevated consumption detected during unoccupied "
            "hours; potential non-essential lighting or HVAC "
            "systems may be active."
        )

        action = (
            "Verify automated BMS nighttime setback schedules "
            "and confirm lighting contactors and peripheral "
            "equipment shut down as scheduled."
        )

    elif is_low_occupancy and not is_after_hours:

        category = (
            "Low Occupancy Inefficiency"
        )

        cause = (
            "Energy draw remains high despite reduced occupant "
            "density, suggesting static ventilation or lighting "
            "output in low-utilization zones."
        )

        action = (
            "Review dynamic ventilation controls, "
            "demand-controlled ventilation (DCV) setpoints, "
            "and optimize zone setbacks for partially occupied "
            "floors."
        )

    elif is_high_temp:

        category = (
            "High-Temperature / Thermal Stress"
        )

        cause = (
            "Potential HVAC-related inefficiency or elevated "
            "cooling demand under high ambient outdoor conditions."
        )

        action = (
            "Inspect chiller and DX system setpoints, verify "
            "economizer dampers are not admitting unconditioned "
            "air, and inspect condenser coil cleanliness."
        )

    else:

        category = (
            "General Operational Deviation"
        )

        cause = (
            "Unusual consumption observed relative to the "
            "baseline without explicit after-hours or extreme "
            "thermal triggers."
        )

        action = (
            "Review sub-meter trends to isolate process loads, "
            "plug loads, or ancillary building services "
            "contributing to the spike."
        )

    # --------------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------------

    return {
        "severity": severity,
        "anomaly_category": category,
        "possible_cause": cause,
        "recommendation": action
    }


# ============================================================
# COMPLETE BUILDING STREAM
# ============================================================

def process_building_stream(
    df: pd.DataFrame,
    anomaly_col: str = "anomaly_prediction"
) -> pd.DataFrame:
    """
    Process a complete building data stream.

    Generates:
        - is_persistent
        - severity
        - anomaly_category
        - possible_cause
        - recommendation

    Existing recommendation columns are removed first so that
    duplicate DataFrame columns cannot be created.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "Input must be a pandas DataFrame."
        )

    if anomaly_col not in df.columns:
        raise ValueError(
            f"Required column '{anomaly_col}' "
            "was not found in the input DataFrame."
        )

    # Make a copy so the original DataFrame is never modified.
    df_out = df.copy()

    # --------------------------------------------------------
    # Remove duplicate column names already present in input.
    # Keep the FIRST occurrence.
    # --------------------------------------------------------

    if df_out.columns.duplicated().any():
        df_out = df_out.loc[
            :,
            ~df_out.columns.duplicated()
        ].copy()

    # --------------------------------------------------------
    # Calculate persistence.
    # --------------------------------------------------------

    persistence_series = determine_persistence(
        df_out,
        anomaly_col=anomaly_col,
        threshold_consecutive=3
    )

    df_out["is_persistent"] = (
        persistence_series
        .reindex(df_out.index)
        .fillna(False)
        .astype(bool)
    )

    # --------------------------------------------------------
    # Generate recommendations.
    # --------------------------------------------------------

    recommendations = []

    for index, row in df_out.iterrows():

        persistent = bool(
            row["is_persistent"]
        )

        rec = generate_recommendation(
            row.to_dict(),
            is_persistent=persistent
        )

        recommendations.append(rec)

    rec_df = pd.DataFrame(
        recommendations,
        index=df_out.index
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Remove any old recommendation columns before
    # concatenating the new ones.
    # --------------------------------------------------------

    recommendation_columns = [
        "severity",
        "category",
        "anomaly_category",
        "possible_cause",
        "recommendation"
    ]

    df_out = df_out.drop(
        columns=[
            column
            for column in recommendation_columns
            if column in df_out.columns
        ],
        errors="ignore"
    )

    # --------------------------------------------------------
    # Combine source data + fresh recommendations.
    # --------------------------------------------------------

    result = pd.concat(
        [df_out, rec_df],
        axis=1
    )

    # --------------------------------------------------------
    # Final duplicate-column safety check.
    # --------------------------------------------------------

    if result.columns.duplicated().any():

        result = result.loc[
            :,
            ~result.columns.duplicated()
        ].copy()

    return result


# ============================================================
# DASHBOARD COMPATIBILITY WRAPPER
# ============================================================

def generate_recommendations(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Compatibility wrapper used by integration.py.

    Runs the recommendation engine for the complete
    building data stream.
    """

    return process_building_stream(df)