import os
import joblib
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

MODEL_PATH = os.path.join(
    MODEL_DIR, "final_isolation_forest_v2.joblib"
)

SCALER_PATH = os.path.join(
    MODEL_DIR, "final_isolation_forest_scaler_v2.joblib"
)

FEATURES_PATH = os.path.join(
    MODEL_DIR, "final_model_features_v2.joblib"
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
features = joblib.load(FEATURES_PATH)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(df):

    data = df.copy()

    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data.sort_values(
        ["meter_id", "timestamp"]
    ).reset_index(drop=True)

    # Time features
    data["hour"] = data["timestamp"].dt.hour
    data["day_of_week"] = data["timestamp"].dt.dayofweek
    data["is_weekend"] = (
        data["day_of_week"] >= 5
    ).astype(int)

    if "is_holiday" not in data.columns:
        data["is_holiday"] = 0

    # Per-meter lag features
    grouped = data.groupby("meter_id")["consumption_kwh"]

    data["lag1_kwh"] = grouped.shift(1)
    data["lag24_kwh"] = grouped.shift(24)

    # Rolling statistics
    data["roll24_mean_kwh"] = (
        data.groupby("meter_id")["consumption_kwh"]
        .transform(
            lambda x: x.shift(1).rolling(24).mean()
        )
    )

    data["roll24_std_kwh"] = (
        data.groupby("meter_id")["consumption_kwh"]
        .transform(
            lambda x: x.shift(1).rolling(24).std()
        )
    )

    data["meter_roll_12h"] = (
        data.groupby("meter_id")["consumption_kwh"]
        .transform(
            lambda x: x.shift(1).rolling(12).mean()
        )
    )

    data["meter_roll_24h"] = (
        data.groupby("meter_id")["consumption_kwh"]
        .transform(
            lambda x: x.shift(1).rolling(24).mean()
        )
    )

    data["meter_roll_48h"] = (
        data.groupby("meter_id")["consumption_kwh"]
        .transform(
            lambda x: x.shift(1).rolling(48).mean()
        )
    )

    data["meter_std_24h"] = (
        data.groupby("meter_id")["consumption_kwh"]
        .transform(
            lambda x: x.shift(1).rolling(24).std()
        )
    )

    data["meter_std_48h"] = (
        data.groupby("meter_id")["consumption_kwh"]
        .transform(
            lambda x: x.shift(1).rolling(48).std()
        )
    )

    # Basic deviations
    data["deviation_kwh"] = (
        data["consumption_kwh"]
        - data["roll24_mean_kwh"]
    )

    data["deviation_percentage"] = (
        data["deviation_kwh"]
        / data["roll24_mean_kwh"].abs().clip(lower=0.01)
    ) * 100

    data["change_from_previous"] = (
        data["consumption_kwh"]
        - data["lag1_kwh"]
    )

    data["change_from_previous_pct"] = (
        data["change_from_previous"]
        / data["lag1_kwh"].abs().clip(lower=0.01)
    ) * 100

    data["change_from_24h"] = (
        data["consumption_kwh"]
        - data["lag24_kwh"]
    )

    data["change_from_24h_pct"] = (
        data["change_from_24h"]
        / data["lag24_kwh"].abs().clip(lower=0.01)
    ) * 100

    # Meter-specific deviations
    data["meter_deviation_24h"] = (
        data["consumption_kwh"]
        - data["meter_roll_24h"]
    )

    data["meter_deviation_48h"] = (
        data["consumption_kwh"]
        - data["meter_roll_48h"]
    )

    data["meter_deviation_pct"] = (
        data["meter_deviation_24h"]
        / data["meter_roll_24h"].abs().clip(lower=0.01)
    ) * 100

    # Z-scores
    data["meter_z_score"] = (
        data["meter_deviation_24h"]
        / data["meter_std_24h"].abs().clip(lower=0.01)
    )

    data["meter_z_score_48h"] = (
        data["meter_deviation_48h"]
        / data["meter_std_48h"].abs().clip(lower=0.01)
    )

    data["baseline_z_score"] = (
        data["deviation_kwh"]
        / data["roll24_std_kwh"].abs().clip(lower=0.01)
    )

    # Contextual features
    if "temp_c" not in data.columns:
        data["temp_c"] = 25

    if "humidity_pct" not in data.columns:
        data["humidity_pct"] = 50

    if "occupancy_index" not in data.columns:
        data["occupancy_index"] = 0.5

    data["occupancy_consumption_ratio"] = (
        data["consumption_kwh"]
        / data["occupancy_index"].abs().clip(lower=0.01)
    )

    data["temp_consumption_interaction"] = (
        data["temp_c"] * data["consumption_kwh"]
    )

    # Cyclic time features
    data["hour_sin"] = np.sin(
        2 * np.pi * data["hour"] / 24
    )

    data["hour_cos"] = np.cos(
        2 * np.pi * data["hour"] / 24
    )

    data["day_sin"] = np.sin(
        2 * np.pi * data["day_of_week"] / 7
    )

    data["day_cos"] = np.cos(
        2 * np.pi * data["day_of_week"] / 7
    )

    return data


# ============================================================
# DETECTION FUNCTION
# ============================================================

def detect_anomalies(df):

    data = create_features(df)

    # Keep rows where all required features exist
    valid = data.dropna(
        subset=features
    ).copy()

    if valid.empty:
        raise ValueError(
            "Not enough historical data to create "
            "the required features."
        )

    # Prevent extreme values from dominating
    X = valid[features].replace(
        [np.inf, -np.inf],
        np.nan
    )

    valid = valid.loc[X.notna().all(axis=1)].copy()
    X = valid[features]

    # Scale using the SAME scaler used during training
    X_scaled = scaler.transform(X)

    # Isolation Forest prediction
    raw_prediction = model.predict(X_scaled)

    # Isolation Forest decision score
    if_score = model.decision_function(X_scaled)

    # More negative = more anomalous
    valid["if_prediction"] = raw_prediction
    valid["if_score"] = if_score

    # Convert prediction to readable result
    valid["anomaly"] = (
        valid["if_prediction"] == -1
    ).astype(int)

    valid["status"] = np.where(
        valid["anomaly"] == 1,
        "ANOMALY",
        "NORMAL"
    )

    # Calculate baseline
    valid["baseline_kwh"] = (
        valid["meter_roll_24h"]
    )

    valid["excess_energy_kwh"] = (
        valid["consumption_kwh"]
        - valid["baseline_kwh"]
    ).clip(lower=0)

    # Severity
    valid["severity"] = "Normal"

    valid.loc[
        (valid["anomaly"] == 1) &
        (valid["if_score"] < -0.15),
        "severity"
    ] = "High"

    valid.loc[
        (valid["anomaly"] == 1) &
        (valid["if_score"] >= -0.15),
        "severity"
    ] = "Medium"

    return valid


# ============================================================
# SIMPLE SUMMARY
# ============================================================

def get_summary(results):

    total = len(results)
    anomalies = int(
        results["anomaly"].sum()
    )

    excess_energy = results[
        "excess_energy_kwh"
    ].sum()

    anomaly_percentage = (
        anomalies / total * 100
        if total > 0 else 0
    )

    return {
        "total_records": total,
        "anomalies": anomalies,
        "anomaly_percentage": round(
            anomaly_percentage, 2
        ),
        "estimated_excess_energy_kwh": round(
            excess_energy, 2
        )
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("ISOLATION FOREST ANOMALY DETECTOR")
    print("=" * 60)

    print("\nModel loaded successfully.")
    print("Model:", MODEL_PATH)
    print("Scaler:", SCALER_PATH)
    print("Features:", FEATURES_PATH)

    print("\nDetector ready.")
    print("Use detect_anomalies(data) to analyze energy data.")