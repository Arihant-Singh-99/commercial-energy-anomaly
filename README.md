# PS-5B: Commercial Building Energy Anomaly Detection
## Contextual Recommendation & Impact Engine

Track: Energy & Resource Efficiency | Aligned SDGs: 7, 9, 12, 13

### Architecture
- **ML Anomaly Input**: Isolation Forest output with consumption, baseline, occupancy, and thermal metrics.
- **Contextual Interpretation (`src/recommendations.py`)**: Rule-based categorization across:
  - After-Hours Operation (unoccupied baseline creep)
  - Low-Occupancy Inefficiencies (static conditioning in low-density zones)
  - High-Temperature Thermal Stress (cooling equipment & setpoint strain)
  - Persistent Baseload Deviations (3+ consecutive hours of abnormal draw)
- **Potential Energy & Cost Savings (`src/impact.py`)**:
  - Calculates non-negative excess energy: `max(actual - baseline, 0)`
  - Projects potential dollar savings ($0.15/kWh assumed) and CO2 reduction (0.42 kg/kWh assumed)
  - Aggregation engine supporting rollups by `anomaly`, `day`, `week`, `month`, and `building`.

### Quick Run
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python pipeline.py