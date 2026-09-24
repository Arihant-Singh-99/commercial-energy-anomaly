import pandas as pd
from src.recommendations import process_building_stream
from src.impact import calculate_potential_savings

def run():
    print("Reading data/ml_output.csv...")
    df = pd.read_csv("data/ml_output.csv")
    
    print("Generating context recommendations and severity...")
    enriched = process_building_stream(df)
    
    print("Calculating potential energy, cost, and CO2 reductions...")
    final = calculate_potential_savings(
        enriched,
        price_per_kwh=0.15,
        grid_emission_factor_kg_co2_per_kwh=0.42
    )
    
    final.to_csv("data/dashboard_input.csv", index=False)
    print("Success! Created data/dashboard_input.csv\n")
    
    cols = ["timestamp", "severity", "category", "potential_savings_kwh", "estimated_potential_cost_savings"]
    print(final[cols].to_string(index=False))

if __name__ == "__main__":
    run()
