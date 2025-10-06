import pandas as pd

# Load the FZP file (assuming CSV format after processing)
fzp_file_path = r"D:\\vissim experiments\\FZP Files\\eval_no_ctrl_Mainz_pattern1\\Mainz_787.fzp"

jam_speed_threshold = 5

fzp_columns = ["Time", "VehicleID", "LinkID", "Speed"]
fzp_data = pd.read_csv(fzp_file_path, sep=";", usecols=[0, 1, 2, 5], names=fzp_columns, skiprows=23)

rl_links = eval(open('D:\\vissim experiments\\ML\\rl_links_mainz.txt','r').read())

# Create a mapping from LinkID to Super-Link ID
link_to_superlink = {}
for i, superlink in enumerate(rl_links):
    for link in superlink:
        link_to_superlink[link] = i

# Add a Super-Link column to FZP data
fzp_data["SuperLinkID"] = fzp_data["LinkID"].map(link_to_superlink)

# Compute the average speed per super-link per time step
avg_speed_per_superlink = fzp_data.groupby(["Time", "SuperLinkID"])["Speed"].mean().reset_index()

# Find the first time each super-link becomes jammed
time_to_jam_per_superlink = avg_speed_per_superlink[avg_speed_per_superlink["Speed"] <= jam_speed_threshold] \
    .groupby("SuperLinkID")["Time"].min().reset_index()

# Find the earliest time any super-link becomes jammed
overall_time_to_jam = time_to_jam_per_superlink["Time"].min()


print(f"Average time to hit a jam: {overall_time_to_jam:.2f} seconds")
