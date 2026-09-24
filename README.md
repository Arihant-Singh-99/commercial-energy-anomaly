# ⚡ Commercial Building Energy Copilot

An AI-powered energy anomaly detection and investigation system for commercial buildings. The system detects unusual energy consumption, analyzes the surrounding operating conditions, estimates energy impact, and provides actionable recommendations through an interactive Streamlit dashboard.

## 🚨 Problem

Commercial buildings generate large amounts of energy and operational data, but unusual consumption can be difficult to identify manually.

Energy waste may be caused by:

- Inefficient HVAC operation
- After-hours energy consumption
- Low-occupancy operation
- Sudden consumption spikes
- Persistent deviations from normal usage

Building operators need a system that can identify these patterns and help prioritize areas for investigation.

## 💡 Solution

Commercial Building Energy Copilot converts smart-building data into an operator-focused workflow:

```text
Smart Building Data
        ↓
Data Preprocessing
        ↓
Feature Engineering
        ↓
Isolation Forest
        ↓
Anomaly Detection
        ↓
Contextual Analysis
        ↓
Impact & Savings
        ↓
Recommendations
        ↓
Streamlit Dashboard
```

The system uses an **unsupervised Isolation Forest model**, allowing unusual energy behavior to be detected without requiring manually labelled anomaly data.

## ✨ Key Features

- 🤖 Unsupervised ML-based anomaly detection
- 📊 Consumption vs. baseline visualization
- 🏢 Multi-building/meter analysis
- 🔎 Building, date, hour and severity filters
- 🚨 Anomaly severity classification
- 🧠 Contextual anomaly explanations
- 💡 Deterministic energy-efficiency recommendations
- ⚡ Excess energy estimation
- 💰 Potential savings estimation
- 📈 Interactive Plotly charts
- 🖥️ Streamlit dashboard
- 📴 Local/offline operation

## 🧠 Machine Learning

The project uses:

**Isolation Forest + RobustScaler**

The trained model analyzes **35 engineered features** covering:

- Current energy consumption
- Previous-hour and previous-day consumption
- Rolling consumption statistics
- Meter-specific historical behavior
- Consumption deviations
- Percentage changes
- Z-scores
- Temperature
- Humidity
- Occupancy
- Time of day
- Day of week
- Weekend/holiday information
- Cyclic time features

### Model Artifacts

```text
model/
├── final_isolation_forest_v2.joblib
├── final_isolation_forest_scaler_v2.joblib
└── final_model_features_v2.joblib
```

## 🏗️ Architecture

```text
commercial-energy-anomaly/
│
├── dashboard/
│   └── app.py
│
├── data/
│   └── capstone_smartgrid_20000.xlsx
│
├── model/
│   ├── final_isolation_forest_v2.joblib
│   ├── final_isolation_forest_scaler_v2.joblib
│   └── final_model_features_v2.joblib
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── recommendations.py
│   └── impact.py
│
├── isolation_forest_detector.py
├── pipeline.py
├── requirements.txt
└── README.md
```

### Main Components

**`dashboard/app.py`**  
Streamlit dashboard containing filters, KPIs, charts, anomaly analysis, recommendations and impact information.

**`isolation_forest_detector.py`**  
Feature engineering and Isolation Forest inference using the trained model and scaler.

**`src/data_loader.py`**  
Loads and normalizes the supplied energy dataset.

**`src/preprocessing.py`**  
Handles timestamps, numeric fields and preprocessing required by the pipeline.

**`src/recommendations.py`**  
Generates contextual recommendations using deterministic rules.

**`src/impact.py`**  
Calculates excess energy and potential savings.

## 📊 Dataset

The primary dataset is:

```text
data/capstone_smartgrid_20000.xlsx
```

It contains smart-building information including:

- Meter IDs
- Timestamps
- Building types
- Temperature
- Humidity
- Occupancy
- Energy consumption
- Historical consumption features
- Solar generation
- Sensor health
- Outage risk

The dataset contains **40 unique meters**.

## 🛠️ Tech Stack

- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **Isolation Forest**
- **RobustScaler**
- **Streamlit**
- **Plotly**
- **OpenPyXL**
- **Joblib**

## 🚀 Installation

Clone the repository and enter the project directory:

```powershell
git clone https://github.com/Arihant-Singh-99/commercial-energy-anomaly.git
cd commercial-energy-anomaly
```

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

## ▶️ Run the Dashboard

From the project root:

```powershell
python -m streamlit run dashboard/app.py
```

The dashboard will open in your browser.

## 🎯 Demo Flow

For a hackathon demonstration:

1. Launch the Streamlit dashboard.
2. Select a building using the sidebar.
3. View energy consumption against the expected baseline.
4. Apply severity/date/hour filters.
5. Open a detected anomaly.
6. Review the anomaly score and severity.
7. Examine the operating context.
8. View the likely cause.
9. Review the recommended action.
10. View estimated excess energy and potential savings.

## 🧪 Testing

The project can be tested using normal and synthetic anomaly scenarios, including:

- Normal operation
- After-hours consumption
- Low-occupancy consumption
- Sudden consumption spikes
- Persistent deviations
- Multiple buildings/meters
- Missing values
- Insufficient historical data

Test datasets can be used to demonstrate how the anomaly-detection pipeline responds to different levels of abnormal energy behavior.

## 🎯 Project Goal

The goal is to move beyond simply detecting an anomaly.

```text
Detect
  ↓
Explain
  ↓
Quantify
  ↓
Recommend
```

Commercial Building Energy Copilot transforms raw building energy data into actionable information that can help operators investigate potential energy waste and improve building efficiency.

---

## 👥 Hackathon Project

**Track:** Energy & Resource Efficiency

**Aligned SDGs:** SDG 7 · SDG 9 · SDG 12 · SDG 13