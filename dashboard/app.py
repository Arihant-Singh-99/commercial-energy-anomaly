import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import joblib


# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="Commercial Building Energy Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("⚡ Commercial Building Energy Intelligence")
st.caption(
    "AI-powered commercial building energy anomaly detection, "
    "impact analysis, and efficiency recommendations."
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

# Works when this file is placed in the project root.
# Also supports the common project structure:
# C:\...\commercial-energy-anomaly\
#   model\
#   data\
MODEL_DIR = ROOT / "model"

MODEL_PATH = MODEL_DIR / "final_isolation_forest_v2.joblib"
SCALER_PATH = MODEL_DIR / "final_isolation_forest_scaler_v2.joblib"
FEATURES_PATH = MODEL_DIR / "final_model_features_v2.joblib"

DEFAULT_DATA_FILES = [
    ROOT / "data" / "capstone_smartgrid_20000.xlsx",
    ROOT / "data" / "dashboard_input.csv",
    ROOT / "data" / "ml_output.csv",
]


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model_assets():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Scaler not found: {SCALER_PATH}")

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Feature list not found: {FEATURES_PATH}")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    features = joblib.load(FEATURES_PATH)

    if not isinstance(features, list):
        features = list(features)

    return model, scaler, features


# ============================================================
# DATA NORMALIZATION
# ============================================================

def normalize_columns(df):
    df = df.copy()

    # Remove accidental duplicate columns while preserving the first.
    df = df.loc[:, ~df.columns.duplicated()].copy()

    rename_map = {}

    for col in df.columns:
        clean = str(col).strip()
        low = clean.lower()

        if low in {"timestamp", "datetime", "date_time", "time"}:
            rename_map[col] = "timestamp"

        elif low in {
            "building",
            "building_id",
            "building id",
            "building_name",
        }:
            rename_map[col] = "building_id"

        elif low in {
            "meter",
            "meter_id",
            "meter id",
        }:
            rename_map[col] = "meter_id"

        elif low in {
            "consumption",
            "consumption_kwh",
            "consumption (kwh)",
            "energy",
            "energy_kwh",
            "energy consumption",
        }:
            rename_map[col] = "consumption_kwh"

        elif low in {
            "temperature",
            "temp",
            "temp_c",
            "temperature_c",
            "temperature (c)",
        }:
            rename_map[col] = "temp_c"

        elif low in {
            "humidity",
            "humidity_pct",
            "humidity (%)",
        }:
            rename_map[col] = "humidity_pct"

        elif low in {
            "occupancy",
            "occupancy_index",
            "occupancy (%)",
        }:
            rename_map[col] = "occupancy_index"

        elif low in {
            "holiday",
            "is_holiday",
        }:
            rename_map[col] = "is_holiday"

    df = df.rename(columns=rename_map)

    # Required energy column.
    if "consumption_kwh" not in df.columns:
        possible = [
            c for c in df.columns
            if "consumption" in str(c).lower()
            or "energy" in str(c).lower()
        ]

        if possible:
            df["consumption_kwh"] = pd.to_numeric(
                df[possible[0]], errors="coerce"
            )

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce"
        )

    if "consumption_kwh" in df.columns:
        df["consumption_kwh"] = pd.to_numeric(
            df["consumption_kwh"],
            errors="coerce"
        )
    if "building_id" not in df.columns:
        if "meter_id" in df.columns:
            meter_values = (
                df["meter_id"]
                .astype(str)
                .str.strip()
            )

            def meter_to_building(meter):
                try:
                    number = int(meter.replace("M", ""))
                    return f"Building {number}"
                except (ValueError, TypeError):
                    return f"Building {meter}"

            df["building_id"] = meter_values.map(meter_to_building)
        else:
            df["building_id"] = "Building 1"

    if "meter_id" not in df.columns:
        df["meter_id"] = df["building_id"].astype(str)

    if "temp_c" not in df.columns:
        df["temp_c"] = 25.0

    if "humidity_pct" not in df.columns:
        df["humidity_pct"] = 50.0

    if "occupancy_index" not in df.columns:
        df["occupancy_index"] = 0.5

    if "is_holiday" not in df.columns:
        df["is_holiday"] = 0

    return df



# ============================================================
# FEATURE ENGINEERING
# Matches the 35 features used by the trained model.
# ============================================================

def create_features(df):
    data = normalize_columns(df)

    if "consumption_kwh" not in data.columns:
        raise ValueError(
            "Input data must contain a consumption/energy column."
        )

    data = data.copy()

    if "timestamp" in data.columns:
        data = data.sort_values(
            ["meter_id", "timestamp"]
        ).reset_index(drop=True)

        data["hour"] = data["timestamp"].dt.hour
        data["day_of_week"] = data["timestamp"].dt.dayofweek
    else:
        data["hour"] = 12
        data["day_of_week"] = 0

    data["is_weekend"] = (
        data["day_of_week"] >= 5
    ).astype(int)

    data["is_holiday"] = pd.to_numeric(
        data["is_holiday"],
        errors="coerce"
    ).fillna(0)

    data["temp_c"] = pd.to_numeric(
        data["temp_c"],
        errors="coerce"
    ).fillna(25)

    data["humidity_pct"] = pd.to_numeric(
        data["humidity_pct"],
        errors="coerce"
    ).fillna(50)

    data["occupancy_index"] = pd.to_numeric(
        data["occupancy_index"],
        errors="coerce"
    ).fillna(0.5)

    data["consumption_kwh"] = pd.to_numeric(
        data["consumption_kwh"],
        errors="coerce"
    )

    # Global time-series features.
    grouped = data.groupby("meter_id")["consumption_kwh"]

    data["lag1_kwh"] = grouped.shift(1)

    data["lag24_kwh"] = grouped.shift(24)

    data["roll24_mean_kwh"] = (
        grouped.shift(1)
        .rolling(24)
        .mean()
        .reset_index(level=0, drop=True)
    )

    data["roll24_std_kwh"] = (
        grouped.shift(1)
        .rolling(24)
        .std()
        .reset_index(level=0, drop=True)
    )

    # Meter-specific rolling features.
    data["meter_roll_12h"] = (
        grouped.shift(1)
        .rolling(12)
        .mean()
        .reset_index(level=0, drop=True)
    )

    data["meter_roll_24h"] = (
        grouped.shift(1)
        .rolling(24)
        .mean()
        .reset_index(level=0, drop=True)
    )

    data["meter_roll_48h"] = (
        grouped.shift(1)
        .rolling(48)
        .mean()
        .reset_index(level=0, drop=True)
    )

    data["meter_std_24h"] = (
        grouped.shift(1)
        .rolling(24)
        .std()
        .reset_index(level=0, drop=True)
    )

    data["meter_std_48h"] = (
        grouped.shift(1)
        .rolling(48)
        .std()
        .reset_index(level=0, drop=True)
    )

    # Basic deviations.
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

    # Meter-specific deviations.
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

    # Z scores.
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

    # Contextual features.
    data["occupancy_consumption_ratio"] = (
        data["consumption_kwh"]
        / data["occupancy_index"].abs().clip(lower=0.01)
    )

    data["temp_consumption_interaction"] = (
        data["temp_c"]
        * data["consumption_kwh"]
    )

    # Cyclic time features.
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
# ANOMALY DETECTION
# ============================================================

def detect_anomalies(df, model, scaler, features):
    data = create_features(df)

    missing_features = [
        f for f in features
        if f not in data.columns
    ]

    if missing_features:
        raise ValueError(
            "The following model features could not be created: "
            + ", ".join(missing_features)
        )

    X = data[features].replace(
        [np.inf, -np.inf],
        np.nan
    )

    valid_mask = X.notna().all(axis=1)

    if not valid_mask.any():
        raise ValueError(
            "No rows contain enough historical data for prediction. "
            "The trained model needs lag/rolling history."
        )

    valid = data.loc[valid_mask].copy()
    X_valid = X.loc[valid_mask]

    X_scaled = scaler.transform(X_valid)

    valid["if_prediction"] = model.predict(X_scaled)
    valid["if_score"] = model.decision_function(X_scaled)

    valid["anomaly"] = (
        valid["if_prediction"] == -1
    ).astype(int)

    valid["status"] = np.where(
        valid["anomaly"].eq(1),
        "ANOMALY",
        "NORMAL"
    )

    # Baseline is the same 24-hour rolling baseline used by the detector.
    valid["baseline_kwh"] = valid["meter_roll_24h"]

    valid["deviation_kwh"] = (
        valid["consumption_kwh"]
        - valid["baseline_kwh"]
    )

    valid["excess_energy_kwh"] = (
        valid["deviation_kwh"]
        .clip(lower=0)
    )

    # Severity based on anomaly score.
    valid["severity"] = "Normal"

    high = (
        valid["anomaly"].eq(1)
        & valid["if_score"].lt(-0.15)
    )

    medium = (
        valid["anomaly"].eq(1)
        & valid["if_score"].ge(-0.15)
    )

    valid.loc[medium, "severity"] = "Medium"
    valid.loc[high, "severity"] = "High"

    return valid


# ============================================================
# RECOMMENDATIONS
# ============================================================

def add_recommendations(data):
    data = data.copy()

    categories = []
    causes = []
    recommendations = []

    for _, row in data.iterrows():

        if row["anomaly"] != 1:
            categories.append("Normal Operation")
            causes.append(
                "Consumption aligns with learned operating patterns."
            )
            recommendations.append(
                "No immediate action required."
            )
            continue

        occupancy = float(
            row.get("occupancy_index", 0.5)
        )

        temp = float(
            row.get("temp_c", 25)
        )

        hour = int(
            row.get("hour", 12)
        )

        if (
            hour >= 20 or hour <= 6
        ) and occupancy < 0.25:
            categories.append("After-Hours Energy Use")
            causes.append(
                "Elevated energy consumption during low-occupancy hours."
            )
            recommendations.append(
                "Check HVAC, lighting, pumps and BMS nighttime schedules."
            )

        elif occupancy < 0.25:
            categories.append("Low Occupancy Inefficiency")
            causes.append(
                "Energy demand remains high despite low occupancy."
            )
            recommendations.append(
                "Review HVAC setbacks, ventilation controls and lighting."
            )

        elif temp >= 28:
            categories.append(
                "High-Temperature / Thermal Stress"
            )
            causes.append(
                "High outdoor temperature may be increasing cooling demand."
            )
            recommendations.append(
                "Inspect HVAC setpoints, economizer operation and condenser condition."
            )

        else:
            categories.append(
                "General Operational Deviation"
            )
            causes.append(
                "Consumption is unusual relative to the learned baseline."
            )
            recommendations.append(
                "Review sub-meter trends and isolate the equipment or load causing the spike."
            )

    data["category"] = categories
    data["possible_cause"] = causes
    data["recommendation"] = recommendations

    return data


# ============================================================
# IMPACT CALCULATION
# ============================================================

def calculate_impact(data):
    total_consumption = float(
        pd.to_numeric(
            data["consumption_kwh"],
            errors="coerce"
        ).fillna(0).sum()
    )

    anomaly_count = int(
        data["anomaly"].sum()
    )

    total_excess = float(
        pd.to_numeric(
            data["excess_energy_kwh"],
            errors="coerce"
        ).fillna(0).sum()
    )

    anomaly_rate = (
        anomaly_count / len(data) * 100
        if len(data)
        else 0
    )

    # Potential savings are limited to anomalous rows.
    total_savings = float(
        data.loc[
            data["anomaly"].eq(1),
            "excess_energy_kwh"
        ].sum()
    )

    return {
        "total_consumption_kwh": total_consumption,
        "anomaly_count": anomaly_count,
        "total_excess_energy_kwh": total_excess,
        "total_potential_savings_kwh": total_savings,
        "anomaly_rate_percent": anomaly_rate,
    }


# ============================================================
# DATA LOADING
# ============================================================

def load_uploaded_file(uploaded_file):
    name = uploaded_file.name.lower()

    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)

    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)

    raise ValueError(
        "Please upload a CSV or Excel file."
    )


@st.cache_data(show_spinner=False)
def load_default_file(path_string):
    path = Path(path_string)

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    return pd.read_excel(path)


# ============================================================
# FORMATTING
# ============================================================

def fmt(value, decimals=1):
    try:
        if pd.isna(value):
            return "N/A"
        return f"{float(value):,.{decimals}f}"
    except Exception:
        return "N/A"


def safe_string(value):
    if value is None:
        return "N/A"

    if isinstance(value, pd.Series):
        if value.empty:
            return "N/A"
        value = value.iloc[0]

    try:
        if pd.isna(value):
            return "N/A"
    except Exception:
        pass

    return str(value)


# ============================================================
# DASHBOARD
# ============================================================

def render_kpis(data):
    impact = calculate_impact(data)

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Total Consumption",
        f"{fmt(impact['total_consumption_kwh'])} kWh"
    )

    c2.metric(
        "Anomalies",
        f"{impact['anomaly_count']:,}"
    )

    c3.metric(
        "Excess Energy",
        f"{fmt(impact['total_excess_energy_kwh'])} kWh"
    )

    c4.metric(
        "Potential Savings",
        f"{fmt(impact['total_potential_savings_kwh'])} kWh"
    )

    c5.metric(
        "Anomaly Rate",
        f"{impact['anomaly_rate_percent']:.2f}%"
    )


def render_anomaly_table(data):
    st.subheader("🚨 Detected Energy Anomalies")

    anomalies = data[
        data["anomaly"].eq(1)
    ].copy()

    if anomalies.empty:
        st.success(
            "No anomalies were detected in the available valid records."
        )
        return

    severity_rank = {
        "High": 1,
        "Medium": 2,
        "Normal": 3,
        "Low": 3,
    }

    anomalies["_severity_rank"] = (
        anomalies["severity"]
        .astype(str)
        .map(severity_rank)
        .fillna(4)
    )

    anomalies = anomalies.sort_values(
        ["_severity_rank", "if_score"],
        ascending=[True, True]
    )

    display = anomalies.copy()

    display["Timestamp"] = (
        display["timestamp"]
        .astype(str)
        if "timestamp" in display.columns
        else ""
    )

    display["Building"] = (
        display["building_id"]
        .astype(str)
    )

    display["Consumption (kWh)"] = (
        display["consumption_kwh"]
        .round(2)
    )

    display["Baseline (kWh)"] = (
        display["baseline_kwh"]
        .round(2)
    )

    display["Deviation (kWh)"] = (
        display["deviation_kwh"]
        .round(2)
    )

    display["Anomaly score"] = (
        display["if_score"]
        .round(4)
    )

    display["Severity"] = (
        display["severity"]
        .astype(str)
    )

    display["Category"] = (
        display["category"]
        .astype(str)
    )

    # IMPORTANT:
    # Select only these columns so duplicate source columns can never
    # reach Streamlit/Arrow.
    table = display[
        [
            "Timestamp",
            "Building",
            "Consumption (kWh)",
            "Baseline (kWh)",
            "Deviation (kWh)",
            "Anomaly score",
            "Severity",
            "Category",
        ]
    ].copy()

    # Final duplicate-column safety.
    table = table.loc[
        :,
        ~table.columns.duplicated()
    ]

    st.dataframe(
        table,
        width="stretch",
        hide_index=True,
    )

    st.markdown("---")
    st.subheader("🔎 Anomaly Details")

    selected = st.selectbox(
        "Select an anomaly",
        options=range(len(anomalies)),
        format_func=lambda i: (
            f"{i + 1}. "
            f"{safe_string(anomalies.iloc[i].get('building_id'))} — "
            f"{safe_string(anomalies.iloc[i].get('severity'))} — "
            f"{safe_string(anomalies.iloc[i].get('category'))}"
        ),
    )

    row = anomalies.iloc[int(selected)]

    a, b, c, d = st.columns(4)

    a.metric(
        "Severity",
        safe_string(row.get("severity"))
    )

    b.metric(
        "Consumption",
        f"{fmt(row.get('consumption_kwh'))} kWh"
    )

    c.metric(
        "Baseline",
        f"{fmt(row.get('baseline_kwh'))} kWh"
    )

    d.metric(
        "Excess Energy",
        f"{fmt(row.get('excess_energy_kwh'))} kWh"
    )

    st.write(
        "**Category:**",
        safe_string(row.get("category"))
    )

    st.write(
        "**Possible cause:**",
        safe_string(row.get("possible_cause"))
    )

    st.info(
        "💡 " + safe_string(row.get("recommendation"))
    )


def render_charts(data):
    st.subheader("📊 Energy Overview")

    chart_data = data.copy()

    # --------------------------------------------------------
    # Prepare timestamp
    # --------------------------------------------------------
    if "timestamp" not in chart_data.columns:
        st.warning("Timestamp column is missing from the dataset.")
        return

    chart_data["timestamp"] = pd.to_datetime(
        chart_data["timestamp"],
        errors="coerce"
    )

    chart_data = chart_data.dropna(subset=["timestamp"])

    # --------------------------------------------------------
    # Convert numeric columns safely
    # --------------------------------------------------------
    for col in ["consumption_kwh", "baseline_kwh"]:
        if col in chart_data.columns:
            chart_data[col] = pd.to_numeric(
                chart_data[col],
                errors="coerce"
            )

    chart_data = chart_data.dropna(
        subset=["consumption_kwh", "baseline_kwh"]
    )

    if chart_data.empty:
        st.info("No valid energy data available for the selected filters.")
        return

    # --------------------------------------------------------
    # Aggregate readings with the same timestamp
    # --------------------------------------------------------
    chart_data = (
        chart_data
        .groupby("timestamp", as_index=False)
        .agg(
            consumption_kwh=("consumption_kwh", "sum"),
            baseline_kwh=("baseline_kwh", "sum")
        )
        .sort_values("timestamp")
    )

    # --------------------------------------------------------
    # Keep the latest 500 time points
    # --------------------------------------------------------
    chart_data = chart_data.tail(200)

    # --------------------------------------------------------
    # Energy consumption vs baseline
    # --------------------------------------------------------
    st.caption(
        "Actual energy consumption compared with the expected baseline. "
        "Values are aggregated across the selected buildings."
    )

    energy_chart = (
        chart_data
        .set_index("timestamp")[
            ["consumption_kwh", "baseline_kwh"]
        ]
    )

    st.line_chart(
        energy_chart,
        height=400
    )

    # --------------------------------------------------------
    # Severity distribution
    # --------------------------------------------------------
    if "severity" in data.columns:

        severity_data = data.copy()

        # Make sure severity is a simple 1-D Series
        severity_series = severity_data["severity"]

        if isinstance(severity_series, pd.DataFrame):
            severity_series = severity_series.iloc[:, 0]

        severity_series = (
            severity_series
            .astype(str)
            .str.strip()
            .str.title()
        )

        severity_counts = (
            severity_series
            .value_counts()
            .rename_axis("Severity")
            .to_frame("Count")
        )

        # Consistent order
        severity_order = ["High", "Medium", "Low", "Normal"]

        severity_counts = severity_counts.reindex(
            severity_order
        ).fillna(0)

        severity_counts["Count"] = (
            severity_counts["Count"].astype(int)
        )

        st.subheader("🚨 Anomaly Severity Distribution")

        st.bar_chart(
            severity_counts,
            height=350
        )


# ============================================================
# MAIN
# ============================================================

def main():
    try:
        model, scaler, features = load_model_assets()
    except Exception as exc:
        st.error("Unable to load the trained model.")
        st.code(str(exc))
        st.info(
            "Make sure this app.py is inside "
            "C:\\Users\\hp\\commercial-energy-anomaly "
            "or change MODEL_DIR to your project's model folder."
        )
        st.stop()

    st.sidebar.header("⚙️ Data Source")

    uploaded = st.sidebar.file_uploader(
        "Upload CSV / Excel",
        type=["csv", "xlsx", "xls"],
    )

    raw_data = None
    source_name = None

    if uploaded is not None:
        try:
            raw_data = load_uploaded_file(uploaded)
            source_name = uploaded.name
        except Exception as exc:
            st.error(f"Could not read uploaded file: {exc}")
            st.stop()

    else:
        existing = [
            p for p in DEFAULT_DATA_FILES
            if p.exists()
        ]

        if existing:
            selected_default = st.sidebar.selectbox(
                "Use project data",
                existing,
                format_func=lambda p: p.name,
            )

            try:
                raw_data = load_default_file(
                    str(selected_default)
                )
                source_name = selected_default.name
            except Exception as exc:
                st.error(
                    f"Could not read {selected_default.name}: {exc}"
                )
                st.stop()

        else:
            st.warning(
                "Upload a CSV/Excel file or place the trained dataset "
                "inside the project's data folder."
            )
            st.stop()

    st.sidebar.success(
        f"Loaded: {source_name}"
    )

    st.sidebar.caption(
        f"Rows: {len(raw_data):,}"
    )

    # --------------------------------------------------------
    # Run model
    # --------------------------------------------------------

    try:
        with st.spinner(
            "Running the trained Isolation Forest model..."
        ):
            results = detect_anomalies(
                raw_data,
                model,
                scaler,
                features,
            )

            results = add_recommendations(results)

    except Exception as exc:
        st.error("The energy pipeline could not be completed.")
        st.exception(exc)
        st.info(
            "The trained model requires enough historical rows per meter "
            "to calculate its lag and rolling features."
        )
        st.stop()

    # --------------------------------------------------------
    # Filters
    # --------------------------------------------------------

    st.sidebar.header("🔍 Filters")

    filtered = results.copy()

    if "building_id" in filtered.columns:
        buildings = sorted(
            filtered["building_id"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        selected_buildings = st.sidebar.multiselect(
            "Building",
            buildings,
            default=buildings,
        )

        if selected_buildings:
            filtered = filtered[
                filtered["building_id"]
                .astype(str)
                .isin(selected_buildings)
            ]

    severities = st.sidebar.multiselect(
        "Severity",
        ["High", "Medium", "Normal"],
        default=["High", "Medium", "Normal"],
    )

    filtered = filtered[
        filtered["severity"].isin(severities)
    ]

    # --------------------------------------------------------
    # Dashboard
    # --------------------------------------------------------

    st.caption(
        f"Analyzing {len(filtered):,} valid records from {source_name}"
    )

    render_kpis(filtered)

    st.markdown("---")

    render_charts(filtered)

    st.markdown("---")

    render_anomaly_table(filtered)

    # --------------------------------------------------------
    # Download
    # --------------------------------------------------------

    st.markdown("---")
    st.subheader("⬇️ Export Results")

    export = filtered.copy()

    csv = export.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Download anomaly results CSV",
        data=csv,
        file_name="energy_anomaly_results.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
