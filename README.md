# Commercial Building Energy Copilot

## Problem

Commercial buildings produce large volumes of meter and operating data, but unusual energy consumption can be difficult to identify and explain. Operators need to distinguish normal variation from waste, scheduling issues, HVAC problems, and other conditions that may require investigation.

## Solution

The project turns smart-building observations into an operator-facing investigation workflow:

Smart Building Data -> Data Preprocessing -> Anomaly Detection -> Contextual Interpretation -> Recommendation -> Potential Savings -> Streamlit Dashboard

The current dashboard uses the supplied Excel dataset. The loader adapts source-specific fields into a unified integration schema, preprocessing keeps rows usable, the recommendation engine applies deterministic contextual rules, and the impact layer estimates avoidable energy above baseline.

## Features

- Real-time-style energy monitoring
- Energy anomaly detection
- Building filtering
- Date and hour filtering
- Anomaly severity
- Contextual explanations
- Rule-based recommendations
- Potential savings estimation
- Interactive Plotly visualization
- Demo mode using real dataset records
- Offline operation

## Architecture

```text
energy-anomaly/
|-- data/
|-- src/
|   |-- data_loader.py
|   |-- preprocessing.py
|   |-- recommendations.py
|   `-- impact.py
|-- dashboard/
|   `-- app.py
`-- README.md
```

### Module responsibilities

- `src/data_loader.py`: Reads the supplied Excel or CSV data, validates source fields, adapts source names to the unified dashboard contract, and adds baseline, anomaly, and deviation fields.
- `src/preprocessing.py`: Normalizes timestamps and numeric fields, derives missing hours where possible, preserves rows, and creates reusable anomaly/deviation fields.
- `src/recommendations.py`: Applies deterministic rules for after-hours, low-occupancy, high-consumption, persistent, and fallback anomaly explanations.
- `src/impact.py`: Calculates event savings, aggregate impact metrics, and building-level impact without producing negative savings.
- `dashboard/app.py`: Runs the shared pipeline once with Streamlit caching, applies interactive filters, and presents KPIs, charts, anomaly details, recommendations, and savings.

## Installation

From Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install streamlit pandas numpy plotly scikit-learn openpyxl
```

The current dashboard directly requires Streamlit, pandas, NumPy, Plotly, and openpyxl. `scikit-learn` is included in the hackathon setup command for compatibility with future Team A model output; the current temporary dataset adapter does not import it.

## Running

From the project root:

```powershell
python -m streamlit run dashboard/app.py
```

Streamlit starts a local server and opens the dashboard in a browser. The application runs offline and reads `data/capstone_smartgrid_20000.xlsx` from the project directory.

## Dataset

The dashboard uses the supplied smart-building dataset at:

```text
data/capstone_smartgrid_20000.xlsx
```

The workbook contains meter readings, timestamps, building context, consumption, rolling baseline statistics, anomaly flags, sensor health, and outage risk fields. The illustrative `23:00 / 74 kWh / 24 kWh / 3% occupancy / 50 kWh savings` example is not present in the actual workbook and is not inserted or fabricated by the application.

## Demo

A concise two-minute presentation flow:

1. Open the dashboard.
2. Enable `Demo Mode - real dataset` in the sidebar.
3. Show the automatically selected real anomaly.
4. Explain actual consumption versus the expected baseline.
5. Point out occupancy and operating hour.
6. Show the severity.
7. Show the likely cause.
8. Show the deterministic recommendation.
9. Show potential savings.
10. Change the building or filters to demonstrate scalability.

The dashboard's Demo Mode selects a real high-severity anomaly, preferably after-hours. It does not add synthetic records.

## Reliability

The dashboard and supporting helpers were tested against:

- Normal observation
- Single anomaly
- After-hours anomaly
- Low-occupancy anomaly
- Persistent anomaly
- Different buildings
- Missing values
- Zero baseline
- Empty dataset
- Building with no anomalies

The dashboard also supports building, date, hour, anomaly status, and building-type filtering where that field is available. The expensive loading and rule-based pipeline is cached with Streamlit.

## Design principle

The core system does not depend on an LLM or external API.

The anomaly detector identifies unusual consumption. The recommendation engine uses deterministic rules and available building context. The impact layer estimates potential avoidable energy. The dashboard communicates the result clearly to the operator.
