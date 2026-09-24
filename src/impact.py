import pandas as pd
from typing import Optional, List


# ============================================================
# EVENT-LEVEL SAVINGS
# ============================================================

def calculate_event_savings(
    data: pd.DataFrame,
    actual_col: str = "consumption_kwh",
    baseline_col: str = "baseline_kwh",
    anomaly_col: str = "anomaly_prediction"
) -> pd.DataFrame:
    """
    Calculate potential energy savings for each detected anomaly.
    """

    df = data.copy()

    # Determine anomaly rows
    if anomaly_col in df.columns:

        values = df[anomaly_col]

        if values.dtype == bool:
            is_anomaly = values.fillna(False)

        else:
            is_anomaly = values.apply(
                lambda x: x in (-1, 1, True)
            )

    elif "is_anomaly" in df.columns:

        is_anomaly = df["is_anomaly"].astype(bool)

    else:

        is_anomaly = pd.Series(
            False,
            index=df.index
        )

    # Actual consumption
    actual = pd.to_numeric(
        df[actual_col],
        errors="coerce"
    ).fillna(0)

    # Baseline consumption
    baseline = pd.to_numeric(
        df[baseline_col],
        errors="coerce"
    ).fillna(0)

    # Energy above baseline
    excess = (
        actual - baseline
    ).clip(lower=0)

    # Only anomalous excess is considered potential savings
    df["potential_savings_kwh"] = (
        excess * is_anomaly
    ).round(3)

    return df


# ============================================================
# POTENTIAL SAVINGS
# ============================================================

def calculate_potential_savings(
    data: pd.DataFrame,
    actual_col: str = "consumption_kwh",
    baseline_col: str = "baseline_kwh",
    anomaly_col: str = "anomaly_prediction",
    price_per_kwh: Optional[float] = None,
    grid_emission_factor_kg_co2_per_kwh: Optional[float] = None
) -> pd.DataFrame:
    """
    Calculate energy, financial and environmental savings.
    """

    df = calculate_event_savings(
        data,
        actual_col=actual_col,
        baseline_col=baseline_col,
        anomaly_col=anomaly_col
    )

    if price_per_kwh is not None:

        df["estimated_potential_cost_savings"] = (
            df["potential_savings_kwh"]
            * price_per_kwh
        ).round(2)

    if grid_emission_factor_kg_co2_per_kwh is not None:

        df["potential_co2_reduction_kg"] = (
            df["potential_savings_kwh"]
            * grid_emission_factor_kg_co2_per_kwh
        ).round(3)

    return df


# ============================================================
# AGGREGATE SAVINGS
# ============================================================

def aggregate_savings(
    df: pd.DataFrame,
    by: str = "day",
    group_col: Optional[str] = "building_id",
    timestamp_col: str = "timestamp"
) -> pd.DataFrame:
    """
    Aggregate savings by day, week, month, anomaly or building.
    """

    df_work = df.copy()

    df_work[timestamp_col] = pd.to_datetime(
        df_work[timestamp_col],
        errors="coerce"
    )

    grouping_keys: List[str] = []

    if group_col and group_col in df_work.columns:
        grouping_keys.append(group_col)

    if by == "day":

        df_work["period"] = (
            df_work[timestamp_col]
            .dt.floor("D")
        )

        grouping_keys.append("period")

    elif by == "week":

        df_work["period"] = (
            df_work[timestamp_col]
            .dt.to_period("W")
            .dt.start_time
        )

        grouping_keys.append("period")

    elif by == "month":

        df_work["period"] = (
            df_work[timestamp_col]
            .dt.to_period("M")
            .dt.start_time
        )

        grouping_keys.append("period")

    elif by == "anomaly":

        if "category" in df_work.columns:

            grouping_keys.extend(
                ["category", "severity"]
            )

        elif "anomaly_category" in df_work.columns:

            grouping_keys.append(
                "anomaly_category"
            )

        else:

            grouping_keys.append(
                timestamp_col
            )

    elif by == "building":
        pass

    else:

        raise ValueError(
            f"Unsupported grouping: '{by}'"
        )

    agg_targets = {
        "potential_savings_kwh": "sum",
        "consumption_kwh": "sum",
        "baseline_kwh": "sum"
    }

    if "estimated_potential_cost_savings" in df_work.columns:

        agg_targets[
            "estimated_potential_cost_savings"
        ] = "sum"

    if "potential_co2_reduction_kg" in df_work.columns:

        agg_targets[
            "potential_co2_reduction_kg"
        ] = "sum"

    # Only aggregate columns that actually exist
    agg_targets = {
        column: operation
        for column, operation in agg_targets.items()
        if column in df_work.columns
    }

    if not agg_targets:

        return pd.DataFrame()

    if grouping_keys:

        return (
            df_work
            .groupby(
                grouping_keys,
                as_index=False
            )
            .agg(agg_targets)
        )

    return pd.DataFrame([
        df_work[
            list(agg_targets.keys())
        ].sum()
    ])


# ============================================================
# TOTAL IMPACT
# ============================================================

def calculate_total_impact(
    data: pd.DataFrame,
    actual_col: str = "consumption_kwh",
    baseline_col: str = "baseline_kwh",
    anomaly_col: str = "anomaly_prediction",
    price_per_kwh: float = 0.12,
    grid_emission_factor_kg_co2_per_kwh: float = 0.4
) -> dict:
    """
    Calculate all impact metrics required by the
    Streamlit dashboard.
    """

    df = data.copy()

    # --------------------------------------------------------
    # Find anomaly column
    # --------------------------------------------------------

    if anomaly_col not in df.columns:

        for candidate in [
            "is_anomaly",
            "anomaly_flag",
            "predicted_anomaly"
        ]:

            if candidate in df.columns:

                anomaly_col = candidate
                break

    # --------------------------------------------------------
    # Determine anomaly rows
    # --------------------------------------------------------

    if anomaly_col in df.columns:

        values = df[anomaly_col]

        if values.dtype == bool:

            is_anomaly = values.fillna(False)

        else:

            is_anomaly = values.apply(
                lambda x: x in (-1, 1, True)
            )

    else:

        is_anomaly = pd.Series(
            False,
            index=df.index
        )

    # --------------------------------------------------------
    # Total consumption
    # --------------------------------------------------------

    if actual_col in df.columns:

        actual = pd.to_numeric(
            df[actual_col],
            errors="coerce"
        ).fillna(0)

    else:

        actual = pd.Series(
            0.0,
            index=df.index
        )

    total_consumption_kwh = float(
        actual.sum()
    )

    # --------------------------------------------------------
    # Calculate excess energy
    # --------------------------------------------------------

    if baseline_col in df.columns:

        baseline = pd.to_numeric(
            df[baseline_col],
            errors="coerce"
        ).fillna(0)

        excess_energy = (
            actual - baseline
        ).clip(lower=0)

    elif "excess_energy_kwh" in df.columns:

        excess_energy = pd.to_numeric(
            df["excess_energy_kwh"],
            errors="coerce"
        ).fillna(0)

    else:

        excess_energy = pd.Series(
            0.0,
            index=df.index
        )

    # --------------------------------------------------------
    # Total excess energy
    # --------------------------------------------------------

    total_excess_energy_kwh = float(
        excess_energy.sum()
    )

    # --------------------------------------------------------
    # Potential savings
    # --------------------------------------------------------

    potential_savings = (
        excess_energy * is_anomaly
    )

    total_potential_savings_kwh = float(
        potential_savings.sum()
    )

    # --------------------------------------------------------
    # Anomaly count
    # --------------------------------------------------------

    anomaly_count = int(
        is_anomaly.sum()
    )

    # --------------------------------------------------------
    # Anomaly rate
    # --------------------------------------------------------

    total_records = len(df)

    if total_records > 0:

        anomaly_rate_percent = (
            anomaly_count
            / total_records
            * 100
        )

    else:

        anomaly_rate_percent = 0.0

    # --------------------------------------------------------
    # Financial savings
    # --------------------------------------------------------

    estimated_cost_savings = (
        total_potential_savings_kwh
        * price_per_kwh
    )

    # --------------------------------------------------------
    # CO2 reduction
    # --------------------------------------------------------

    potential_co2_reduction_kg = (
        total_potential_savings_kwh
        * grid_emission_factor_kg_co2_per_kwh
    )

    # --------------------------------------------------------
    # Return complete dictionary
    # --------------------------------------------------------

    return {

        # Dashboard KPIs
        "total_consumption_kwh": round(
            total_consumption_kwh,
            3
        ),

        "anomaly_count": anomaly_count,

        "total_excess_energy_kwh": round(
            total_excess_energy_kwh,
            3
        ),

        "total_potential_savings_kwh": round(
            total_potential_savings_kwh,
            3
        ),

        "anomaly_rate_percent": round(
            anomaly_rate_percent,
            2
        ),

        # Additional aliases / metrics
        "total_savings_kwh": round(
            total_potential_savings_kwh,
            3
        ),

        "estimated_cost_savings": round(
            estimated_cost_savings,
            2
        ),

        "potential_co2_reduction_kg": round(
            potential_co2_reduction_kg,
            3
        )
    }