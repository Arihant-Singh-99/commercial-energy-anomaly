import pandas as pd
from typing import Dict, Any

def determine_persistence(df: pd.DataFrame, anomaly_col: str = "anomaly_prediction", threshold_consecutive: int = 3) -> pd.Series:
    is_anomaly = df[anomaly_col].apply(lambda x: 1 if x in (-1, 1, True) else 0)
    groups = (is_anomaly != is_anomaly.shift()).cumsum()
    consecutive_counts = is_anomaly.groupby(groups).cumsum()
    return (consecutive_counts >= threshold_consecutive) & (is_anomaly == 1)

def generate_recommendation(row: Dict[str, Any], is_persistent: bool = False) -> Dict[str, Any]:
    pred = row.get("anomaly_prediction")
    is_anomaly = pred in (-1, 1, True)
    
    if not is_anomaly:
        return {
            "severity": "None",
            "category": "Normal Operation",
            "possible_cause": "Consumption aligns with baseline expectations.",
            "recommendation": "No action required."
        }
        
    actual = float(row.get("consumption_kwh", 0.0))
    baseline = float(row.get("baseline_kwh", actual))
    occupancy = float(row.get("occupancy_index", 1.0))
    temp = float(row.get("temp_c", 22.0))
    hour = int(row.get("hour", 12))
    
    excess = max(actual - baseline, 0.0)
    deviation_ratio = (excess / baseline) if baseline > 0 else 0.0
    
    is_after_hours = (hour >= 20 or hour <= 6)
    is_low_occupancy = (occupancy < 0.25)
    is_high_temp = (temp >= 28.0)
    
    score = 0
    if deviation_ratio < 0.20:
        score += 1
    elif deviation_ratio < 0.50:
        score += 2
    else:
        score += 3
        
    if is_after_hours and is_low_occupancy:
        score += 2
    elif is_after_hours or is_low_occupancy:
        score += 1
        
    if is_persistent:
        score += 2
        
    if score <= 2:
        severity = "Low"
    elif score <= 4:
        severity = "Medium"
    else:
        severity = "High"
        
    if is_persistent:
        category = "Persistent Baseload Deviation"
        cause = "Continuous abnormal consumption sustained across multiple hours, indicating potential equipment override or failure to set back."
        action = "Conduct walk-through or check BMS logs to verify if equipment, pumps, or air handlers are running in manual/hand mode."
    elif is_after_hours and is_low_occupancy:
        category = "After-Hours Energy Use"
        cause = "Elevated consumption detected during unoccupied hours; potential non-essential lighting or HVAC systems active."
        action = "Verify automated BMS nighttime setback schedules and confirm lighting contactors and peripheral equipment shut down as scheduled."
    elif is_low_occupancy and not is_after_hours:
        category = "Low Occupancy Inefficiency"
        cause = "Energy draw remains high despite reduced occupant density, suggesting static ventilation or lighting output in low-utilization zones."
        action = "Review dynamic ventilation controls, demand-controlled ventilation (DCV) setpoints, and optimize zone setbacks for partially occupied floors."
    elif is_high_temp:
        category = "High-Temperature / Thermal Stress"
        cause = "Potential HVAC-related inefficiency or elevated cooling demand under high ambient outdoor conditions."
        action = "Inspect chiller and DX system setpoints, verify economizer dampers are not admitting unconditioned air, and inspect condenser coil cleanliness."
    else:
        category = "General Operational Deviation"
        cause = "Unusual consumption observed relative to the baseline without explicit after-hours or extreme thermal triggers."
        action = "Review sub-meter trends to isolate process loads, plug loads, or ancillary building services contributing to the spike."
        
    return {
        "severity": severity,
        "category": category,
        "possible_cause": cause,
        "recommendation": action
    }

def process_building_stream(df: pd.DataFrame, anomaly_col: str = "anomaly_prediction") -> pd.DataFrame:
    df_out = df.copy()
    persistence_series = determine_persistence(df_out, anomaly_col=anomaly_col)
    df_out["is_persistent"] = persistence_series
    
    recommendations = []
    for _, row in df_out.iterrows():
        rec = generate_recommendation(row.to_dict(), is_persistent=bool(row["is_persistent"]))
        recommendations.append(rec)
        
    rec_df = pd.DataFrame(recommendations, index=df_out.index)
    return pd.concat([df_out, rec_df], axis=1)
