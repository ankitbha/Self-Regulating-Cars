import pandas as pd

# Load the FZP trajectory data (assume CSV format with semicolon delimiter)
fzp_file = r"D:\\vissim experiments\\nyc_nj\\Mainz20copy_040.fzp"
fzp_columns = ["Time", "VehicleID", "Speed_kmh"]
fzp_data = pd.read_csv(fzp_file, sep=";", usecols=[0, 1, 5], names=fzp_columns, skiprows=22)

# Load CO2 Emissions Lookup Table
lookup_table = pd.DataFrame({
    "Speed_Lower": [0, 5, 15, 20, 30, 50, 90],
    "Speed_Upper": [5, 15, 20, 30, 50, 90, 120],
    "CO2_g_km": [800, 400, 300, 250, 200, 150, 200]  # Example values
})

# Function to get CO2 emissions factor based on speed
def get_co2_factor(speed):
    row = lookup_table[(lookup_table["Speed_Lower"] <= speed) & (lookup_table["Speed_Upper"] > speed)]
    return row["CO2_g_km"].values[0] if not row.empty else 200  # Default value if outside range

# Create bin edges and labels
bin_edges = lookup_table["Speed_Lower"].tolist() + [lookup_table["Speed_Upper"].iloc[-1]]
bin_labels = lookup_table["CO2_g_km"].tolist()

# Assign CO2 emissions using pd.cut (vectorized, much faster)
fzp_data["CO2_g_km"] = pd.cut(fzp_data["Speed_kmh"], bins=bin_edges, labels=bin_labels, include_lowest=True, ordered=False).astype(float)

# Convert CO2 emissions to g/s
fzp_data.loc[fzp_data['Speed_kmh'] < 5, 'Speed_kmh'] = 5
fzp_data["CO2_g_s"] = (fzp_data["CO2_g_km"] * fzp_data["Speed_kmh"]) / 3600  # Convert to grams per second

# Aggregate emissions per vehicle
total_emissions_per_vehicle = fzp_data.groupby("VehicleID")["CO2_g_s"].sum()

# Compute total CO2 emissions for the network
total_co2_emissions = total_emissions_per_vehicle.sum()

# Print total emissions
print(f"Total CO₂ Emissions for the Simulation: {total_co2_emissions:.2f} grams")