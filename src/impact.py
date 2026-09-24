import pandas as pd
from typing import Optional, List

def calculate_potential_savings(
    data: pd.DataFrame,
    actual_col: str = "consumption_kwh",
    baseline_col: str = "baseline_kwh",
    anomaly_col: str = "anomaly_prediction",
    price_per_kwh: Optional[float] = None,
    grid_emission_factor_kg_co2_per_kwh: Optional[float] = None
) -> pd.DataFrame:
    df = data.copy()
    is_anomaly = df[anomaly_col].apply(lambda x: x in (-1, 1, True))
    
    excess = (df[actual_col] - df[baseline_col]).clip(lower=0.0)
    df["potential_savings_kwh"] = (excess * is_anomaly).round(3)
    
    if price_per_kwh is not None:
        df["estimated_potential_cost_savings"] = (df["potential_savings_kwh"] * price_per_kwh).round(2)
        
    if grid_emission_factor_kg_co2_per_kwh is not None:
        df["potential_co2_reduction_kg"] = (df["potential_savings_kwh"] * grid_emission_factor_kg_co2_per_kwh).round(3)
        
    return df

def aggregate_savings(
    df: pd.DataFrame,
    by: str = "day",
    group_col: Optional[str] = "building_id",
    timestamp_col: str = "timestamp"
) -> pd.DataFrame:
    df_work = df.copy()
    df_work[timestamp_col] = pd.to_datetime(df_work[timestamp_col])
    
    grouping_keys: List[str] = []
    if group_col and group_col in df_work.columns:
        grouping_keys.append(group_col)
        
    if by == "day":
        df_work["period"] = df_work[timestamp_col].dt.floor("D")
        grouping_keys.append("period")
    elif by == "week":
        df_work["period"] = df_work[timestamp_col].dt.to_period("W").dt.start_time
        grouping_keys.append("period")
    elif by == "month":
        df_work["period"] = df_work[timestamp_col].dt.to_period("M").dt.start_time
        grouping_keys.append("period")
    elif by == "anomaly":
        if "category" in df_work.columns:
            grouping_keys.extend(["category", "severity"])
        else:
            grouping_keys.append(timestamp_col)
    elif by == "building":
        pass
    else:
        raise ValueError(f"Unsupported grouping: '{by}'")
        
    agg_targets = {
        "potential_savings_kwh": "sum",
        "consumption_kwh": "sum",
        "baseline_kwh": "sum"
    }
    if "estimated_potential_cost_savings" in df_work.columns:
        agg_targets["estimated_potential_cost_savings"] = "sum"
    if "potential_co2_reduction_kg" in df_work.columns:
        agg_targets["potential_co2_reduction_kg"] = "sum"
        
    if grouping_keys:
        return df_work.groupby(grouping_keys, as_index=False).agg(agg_targets)
    return pd.DataFrame([df_work[list(agg_targets.keys())].sum()])
