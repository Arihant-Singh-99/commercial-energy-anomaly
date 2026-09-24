"""Streamlit dashboard for the Commercial Building Energy Copilot."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DATA_PATH = ROOT / "data" / "capstone_smartgrid_20000.xlsx"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from impact import calculate_total_impact
from integration import load_dashboard_data


st.set_page_config(
    page_title="Commercial Building Energy Copilot",
    page_icon=":material/bolt:",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner="Loading and preparing energy data...")
def load_pipeline(path: str) -> pd.DataFrame:
    """Load standardized dashboard data once per source path.

    The source can later be replaced with Team A/B's final DataFrame by
    calling ``load_dashboard_data(final_dataframe)`` without changing the UI.
    """
    return load_dashboard_data(path)


def _format_kwh(value: object) -> str:
    """Format a numeric energy value without exposing NaN in the UI."""
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return "N/A"
    return f"{number:,.1f} kWh"


def _format_value(value: object, suffix: str = "") -> str:
    """Format a scalar value safely for anomaly detail fields."""
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value}{suffix}"


def _display_occupancy(value: object) -> str:
    """Display occupancy index as a percentage when available."""
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return "N/A"
    return f"{number * 100:.1f}%"


def _select_demo_anomaly(data: pd.DataFrame) -> tuple[object, pd.Series] | None:
    """Select a real, high-severity anomaly for the presenter."""
    if data.empty or "is_anomaly" not in data.columns:
        return None

    anomalies = data[data["is_anomaly"].fillna(False)].copy()
    if anomalies.empty:
        return None

    hours = pd.to_numeric(anomalies["hour"], errors="coerce")
    occupancy = pd.to_numeric(anomalies["occupancy_index"], errors="coerce")
    deviation = pd.to_numeric(anomalies["deviation_kwh"], errors="coerce")
    severity = anomalies.get(
        "severity",
        pd.Series("LOW", index=anomalies.index),
    )
    severity_rank = severity.map(
        {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    ).fillna(3)
    ranking = pd.DataFrame(
        {
            "after_hours": (hours.ge(22) | hours.le(6)).astype(int),
            "severity_rank": severity_rank,
            "low_occupancy": occupancy.lt(0.15).fillna(False).astype(int),
            "deviation": deviation.fillna(-np.inf),
        },
        index=anomalies.index,
    )
    selected_index = ranking.sort_values(
        ["after_hours", "severity_rank", "low_occupancy", "deviation"],
        ascending=[False, True, False, False],
    ).index[0]
    return selected_index, data.loc[selected_index]


def _default_date_window(data: pd.DataFrame) -> tuple[object, object]:
    """Choose a short, anomaly-centered initial view without changing data."""
    timestamps = pd.to_datetime(data["timestamp"], errors="coerce", utc=True)
    valid = timestamps.dropna()
    if valid.empty:
        return (None, None)

    anomalies = data[data["is_anomaly"].fillna(False)].copy()
    if not anomalies.empty:
        anomaly_times = pd.to_datetime(
            anomalies["timestamp"], errors="coerce", utc=True
        ).dropna()
        if not anomaly_times.empty:
            anchor = anomaly_times.iloc[0]
        else:
            anchor = valid.iloc[0]
    else:
        anchor = valid.iloc[0]

    start = max(valid.min().date(), (anchor - pd.Timedelta(days=1)).date())
    end = min(valid.max().date(), (anchor + pd.Timedelta(days=1)).date())
    return (start, end)


def _build_filters(data: pd.DataFrame) -> dict[str, object]:
    """Render sidebar controls and return selections without mutating data."""
    st.sidebar.header("Filters")

    demo_mode = st.sidebar.toggle(
        "Demo Mode — real dataset",
        value=False,
        help="Select a real high-severity anomaly for a guided presentation path.",
    )
    presentation_mode = st.sidebar.toggle(
        "Presentation mode",
        value=False,
        help="Reduce technical helper text while keeping the dashboard controls.",
    )

    building_values = sorted(data["building_id"].dropna().astype(str).unique())
    selected_building = st.sidebar.selectbox(
        "Building",
        ["All buildings", *building_values],
    )

    timestamp = pd.to_datetime(data["timestamp"], errors="coerce", utc=True)
    valid_dates = timestamp.dropna()
    if valid_dates.empty:
        date_value = None
    else:
        default_start, default_end = _default_date_window(data)
        date_value = st.sidebar.date_input(
            "Date range",
            value=(default_start, default_end),
            help="Start with a focused window; expand this range for the full history.",
        )

    hour_range = st.sidebar.slider("Hour range", 0, 23, (0, 23))
    status = st.sidebar.selectbox(
        "Anomaly status",
        ["All observations", "Anomalies only", "Normal only"],
    )

    building_type = None
    if "building_type" in data.columns:
        type_values = sorted(data["building_type"].dropna().astype(str).unique())
        building_type = st.sidebar.selectbox(
            "Building type",
            ["All building types", *type_values],
        )

    return {
        "building": selected_building,
        "dates": date_value,
        "hours": hour_range,
        "status": status,
        "building_type": building_type,
        "demo_mode": demo_mode,
        "presentation_mode": presentation_mode,
    }


def _apply_filters(data: pd.DataFrame, filters: dict[str, object]) -> pd.DataFrame:
    """Return a filtered copy while leaving the cached pipeline unchanged."""
    filtered = data.copy()
    timestamp = pd.to_datetime(filtered["timestamp"], errors="coerce", utc=True)
    mask = pd.Series(True, index=filtered.index)

    if filters["building"] != "All buildings":
        mask &= filtered["building_id"].astype(str).eq(filters["building"])

    date_range = filters["dates"]
    if date_range and len(date_range) == 2:
        start = pd.Timestamp(date_range[0], tz="UTC")
        end = pd.Timestamp(date_range[1], tz="UTC") + pd.Timedelta(days=1)
        mask &= timestamp.ge(start) & timestamp.lt(end)

    hours = pd.to_numeric(filtered["hour"], errors="coerce")
    mask &= hours.between(filters["hours"][0], filters["hours"][1])

    if filters["status"] == "Anomalies only":
        mask &= filtered["is_anomaly"].fillna(False)
    elif filters["status"] == "Normal only":
        mask &= ~filtered["is_anomaly"].fillna(False)

    if filters["building_type"] not in (None, "All building types"):
        if "building_type" in filtered.columns:
            mask &= filtered["building_type"].astype(str).eq(filters["building_type"])
        else:
            mask &= False

    return filtered.loc[mask].copy()


def _render_kpis(data: pd.DataFrame) -> None:
    """Render KPI cards from the shared impact calculation."""
    impact = calculate_total_impact(data)
    with st.container(horizontal=True):
        st.metric(
            "Total energy consumption",
            _format_kwh(impact["total_consumption_kwh"]),
            border=True,
        )
        st.metric(
            "Detected anomalies",
            f"{impact['anomaly_count']:,}",
            border=True,
        )
        st.metric(
            "Potential excess energy",
            _format_kwh(impact["total_excess_energy_kwh"]),
            border=True,
        )
        st.metric(
            "Potential savings",
            _format_kwh(impact["total_potential_savings_kwh"]),
            border=True,
        )
        st.metric(
            "Anomaly rate",
            f"{impact['anomaly_rate_percent']:.2f}%",
            border=True,
        )


def _prepare_chart_data(data: pd.DataFrame, all_buildings: bool) -> pd.DataFrame:
    """Reduce chart density while retaining anomaly context and hover fields."""
    chart_data = data.copy()
    chart_data["timestamp"] = pd.to_datetime(
        chart_data["timestamp"], errors="coerce", utc=True
    )
    chart_data = chart_data.dropna(subset=["timestamp"])

    numeric_columns = [
        "consumption_kwh",
        "baseline_kwh",
        "occupancy_index",
        "anomaly_score",
    ]
    for column in numeric_columns:
        chart_data[column] = pd.to_numeric(chart_data[column], errors="coerce")

    if not all_buildings:
        return chart_data.sort_values("timestamp")

    chart_data["anomaly_consumption_kwh"] = chart_data["consumption_kwh"].where(
        chart_data["is_anomaly"].fillna(False)
    )
    chart_data["anomaly_baseline_kwh"] = chart_data["baseline_kwh"].where(
        chart_data["is_anomaly"].fillna(False)
    )
    return (
        chart_data.groupby("timestamp", as_index=False)
        .agg(
            consumption_kwh=("consumption_kwh", "sum"),
            baseline_kwh=("baseline_kwh", "sum"),
            occupancy_index=("occupancy_index", "mean"),
            anomaly_score=("anomaly_score", "max"),
            anomaly_consumption_kwh=("anomaly_consumption_kwh", "sum"),
            anomaly_baseline_kwh=("anomaly_baseline_kwh", "sum"),
        )
        .sort_values("timestamp")
    )


def _render_consumption_chart(
    data: pd.DataFrame,
    presentation_mode: bool = False,
) -> None:
    """Render actual, baseline, and anomaly observations in one chart."""
    st.subheader("Energy consumption")
    all_buildings = data["building_id"].astype(str).nunique() > 1
    chart_data = _prepare_chart_data(data, all_buildings)

    if chart_data.empty:
        st.info("No valid timestamps are available for the selected filters.")
        return

    customdata = chart_data[
        ["baseline_kwh", "occupancy_index", "anomaly_score"]
    ].to_numpy()
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=chart_data["timestamp"],
            y=chart_data["consumption_kwh"],
            mode="lines",
            name="Actual consumption",
            line={"color": "#1f6f8b", "width": 2},
            customdata=customdata,
            hovertemplate=(
                "<b>%{x|%Y-%m-%d %H:%M}</b><br>"
                "Actual: %{y:.2f} kWh<br>"
                "Baseline: %{customdata[0]:.2f} kWh<br>"
                "Occupancy: %{customdata[1]:.1%}<br>"
                "Anomaly score: %{customdata[2]:.2f}<extra></extra>"
            ),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=chart_data["timestamp"],
            y=chart_data["baseline_kwh"],
            mode="lines",
            name="Baseline",
            line={"color": "#9aa6b2", "width": 2, "dash": "dot"},
            hovertemplate="Baseline: %{y:.2f} kWh<extra></extra>",
        )
    )

    if all_buildings:
        anomaly_data = chart_data.dropna(subset=["anomaly_consumption_kwh"]).copy()
        anomaly_y = "anomaly_consumption_kwh"
        anomaly_baseline = "anomaly_baseline_kwh"
    else:
        anomaly_data = chart_data[chart_data["is_anomaly"]].copy()
        anomaly_y = "consumption_kwh"
        anomaly_baseline = "baseline_kwh"

    if not anomaly_data.empty:
        anomaly_customdata = anomaly_data[
            ["occupancy_index", "anomaly_score", anomaly_baseline]
        ].to_numpy()
        figure.add_trace(
            go.Scatter(
                x=anomaly_data["timestamp"],
                y=anomaly_data[anomaly_y],
                mode="markers",
                name="Anomaly",
                marker={
                    "color": "#d1495b",
                    "size": 12,
                    "symbol": "diamond",
                    "line": {"color": "#ffffff", "width": 1},
                },
                customdata=anomaly_customdata,
                hovertemplate=(
                    "<b>ANOMALY | %{x|%Y-%m-%d %H:%M}</b><br>"
                    "Actual: %{y:.2f} kWh<br>"
                    "Baseline: %{customdata[2]:.2f} kWh<br>"
                    "Occupancy: %{customdata[0]:.1%}<br>"
                    "Anomaly score: %{customdata[1]:.2f}<extra></extra>"
                ),
            )
        )

    figure.update_layout(
        height=430,
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.08, "x": 0},
        template="plotly_white",
        xaxis={"title": "Timestamp", "tickformat": "%b %d\n%H:%M"},
        yaxis={"title": "Energy (kWh)", "tickformat": ",.0f"},
    )
    if all_buildings and not presentation_mode:
        st.caption("All buildings are aggregated by timestamp; anomaly markers show anomalous energy.")
    st.plotly_chart(figure, width="stretch")


def _render_anomaly_detail(row: pd.Series) -> None:
    """Render the selected anomaly's contextual explanation and impact."""
    st.header("🚨 Anomaly detected")
    with st.container(horizontal=True):
        st.metric("Actual consumption", _format_kwh(row.get("consumption_kwh")), border=True)
        st.metric("Expected baseline", _format_kwh(row.get("baseline_kwh")), border=True)
        st.metric("Deviation", _format_kwh(row.get("deviation_kwh")), border=True)
        st.metric("Occupancy", _display_occupancy(row.get("occupancy_index")), border=True)
        st.metric(
            "Potential savings",
            _format_kwh(row.get("potential_savings_kwh")),
            border=True,
        )

    timestamp = row.get("timestamp")
    timestamp_text = (
        timestamp.strftime("%Y-%m-%d %H:%M UTC")
        if isinstance(timestamp, pd.Timestamp) and pd.notna(timestamp)
        else "N/A"
    )
    left, right = st.columns(2)
    with left:
        st.write(f"**Timestamp:** {timestamp_text}")
        st.write(f"**Temperature:** {_format_value(row.get('temp_c'), ' °C')}")
        severity = _format_value(row.get("severity"))
        if severity != "N/A":
            st.badge(severity, color="red" if severity == "HIGH" else "orange")
        else:
            st.write("**Severity:** N/A")
    with right:
        st.write(f"**Category:** {_format_value(row.get('anomaly_category'))}")
        with st.container(border=True):
            st.markdown("**Possible cause**")
            st.write(_format_value(row.get("possible_cause")))
            st.markdown("**Recommendation**")
            st.write(_format_value(row.get("recommendation")))

    occupancy = pd.to_numeric(
        pd.Series([row.get("occupancy_index")]), errors="coerce"
    ).iloc[0]
    hour = pd.to_numeric(pd.Series([row.get("hour")]), errors="coerce").iloc[0]
    deviation = pd.to_numeric(
        pd.Series([row.get("deviation_kwh")]), errors="coerce"
    ).iloc[0]
    deviation_percent = pd.to_numeric(
        pd.Series([row.get("deviation_percent")]), errors="coerce"
    ).iloc[0]
    category = str(row.get("anomaly_category", ""))
    reasons = []
    if pd.notna(occupancy) and occupancy < 0.15:
        reasons.append("low occupancy")
    if pd.notna(hour) and (hour >= 22 or hour <= 6):
        reasons.append("after-hours")
    if pd.notna(deviation) and deviation > 0:
        reasons.append("consumption above baseline")
    if "persistent" in category.lower():
        reasons.append("persistent abnormal consumption")
    elif not reasons and pd.notna(deviation_percent) and deviation_percent > 0:
        reasons.append("high consumption relative to baseline")
    with st.container(border=True):
        st.markdown("**Why was this flagged?**")
        st.write(
            " + ".join(reasons).capitalize() + "."
            if reasons
            else "The model marked this observation as anomalous."
        )


def _render_anomaly_table(
    data: pd.DataFrame,
    demo_mode: bool = False,
    demo_selection: tuple[object, pd.Series] | None = None,
) -> None:
    """Show recent anomalies and provide a robust row-selection control."""
    st.subheader("Recent anomalies")
    anomalies = data[data["is_anomaly"]].copy()
    if anomalies.empty:
        st.info("No anomalies match the selected filters.")
        if demo_mode and demo_selection is not None:
            st.caption("The real demo anomaly is outside the current filters.")
            _render_anomaly_detail(demo_selection[1])
        return

    severity_rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    anomalies["_severity_rank"] = anomalies["severity"].map(severity_rank).fillna(3)
    anomalies = anomalies.sort_values(
        ["_severity_rank", "timestamp"],
        ascending=[True, False],
        na_position="last",
    ).drop(columns="_severity_rank")
    display_anomalies = anomalies.head(15)
    demo_index = demo_selection[0] if demo_selection is not None else None
    if demo_mode and demo_index in anomalies.index and demo_index not in display_anomalies.index:
        display_anomalies = pd.concat(
            [anomalies.loc[[demo_index]], display_anomalies]
        ).drop_duplicates()
    table = display_anomalies[
        [
            "timestamp",
            "building_id",
            "consumption_kwh",
            "baseline_kwh",
            "deviation_kwh",
            "anomaly_score",
            "severity",
            "anomaly_category",
        ]
    ].rename(
        columns={
            "building_id": "Building",
            "timestamp": "Timestamp",
            "consumption_kwh": "Consumption (kWh)",
            "baseline_kwh": "Baseline (kWh)",
            "deviation_kwh": "Deviation (kWh)",
            "anomaly_score": "Anomaly score",
            "severity": "Severity",
            "anomaly_category": "Category",
        }
    )
    st.dataframe(
        table,
        width="stretch",
        hide_index=True,
        column_config={
            "Consumption (kWh)": st.column_config.NumberColumn(format="%.2f"),
            "Baseline (kWh)": st.column_config.NumberColumn(format="%.2f"),
            "Deviation (kWh)": st.column_config.NumberColumn(format="%.2f"),
            "Anomaly score": st.column_config.NumberColumn(format="%.2f"),
        },
    )

    labels = []
    for _, row in display_anomalies.iterrows():
        timestamp = row.get("timestamp")
        timestamp_text = (
            timestamp.strftime("%m/%d %H:%M")
            if isinstance(timestamp, pd.Timestamp) and pd.notna(timestamp)
            else "unknown time"
        )
        labels.append(
            f"{row.get('building_id', 'Unknown')} | {timestamp_text} | "
            f"{_format_kwh(row.get('consumption_kwh'))} | {row.get('anomaly_category', 'Anomaly')}"
        )

    default_index = 0
    if demo_mode and demo_index in display_anomalies.index:
        default_index = int(display_anomalies.index.get_loc(demo_index))
    else:
        after_hours = display_anomalies["anomaly_category"].eq("After-hours")
        if after_hours.any():
            default_index = int(after_hours.to_numpy().argmax())
    selected_label = st.selectbox(
        "Select an anomaly to inspect",
        labels,
        index=default_index,
    )
    selected_position = labels.index(selected_label)
    _render_anomaly_detail(display_anomalies.iloc[selected_position])


def _render_investigation_summary(selection: tuple[object, pd.Series]) -> None:
    """Give presenters a short, data-backed narrative for the selected record."""
    row = selection[1]
    building = row.get("building_id", "Unknown")
    actual = _format_kwh(row.get("consumption_kwh"))
    baseline = _format_kwh(row.get("baseline_kwh"))
    savings = _format_kwh(row.get("potential_savings_kwh"))
    with st.container(border=True):
        st.markdown("**Investigation summary**")
        st.caption(
            f"Real dataset demo · {building} · Follow the signal from normal baseline "
            "to actionable recommendation."
        )
        st.write(
            f"**Normal consumption:** expected baseline is {baseline}.  "
            f"**Anomaly detected:** actual consumption is {actual}."
        )
        st.write(
            f"**Context:** {_display_occupancy(row.get('occupancy_index'))} occupancy "
            f"at hour {_format_value(row.get('hour'))}.  "
            f"**Outcome:** potential savings are {savings}."
        )


def main() -> None:
    """Render the dashboard page."""
    st.title("Commercial Building Energy Copilot")
    st.caption("AI-powered energy anomaly detection and efficiency recommendations")

    try:
        data = load_pipeline(str(DATA_PATH))
    except Exception as exc:
        st.error(f"Unable to load the energy pipeline: {exc}")
        st.stop()

    filters = _build_filters(data)
    filtered = _apply_filters(data, filters)
    demo_selection = (
        _select_demo_anomaly(data) if filters["demo_mode"] else None
    )
    if not filters["presentation_mode"]:
        st.caption(
            f"Showing {len(filtered):,} of {len(data):,} observations · "
            "Use the sidebar to expand the analysis window."
        )
    if filters["demo_mode"] and demo_selection is not None:
        _render_investigation_summary(demo_selection)

    _render_kpis(filtered)

    if filtered.empty:
        st.info("No observations match the selected filters.")
        if filters["demo_mode"] and demo_selection is not None:
            _render_anomaly_table(
                filtered,
                demo_mode=True,
                demo_selection=demo_selection,
            )
        return

    _render_consumption_chart(filtered, filters["presentation_mode"])
    _render_anomaly_table(
        filtered,
        demo_mode=filters["demo_mode"],
        demo_selection=demo_selection,
    )


if __name__ == "__main__":
    main()