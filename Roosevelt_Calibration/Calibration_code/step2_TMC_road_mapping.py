"""
This script performs a mapping between TMC (Traffic Message Channel) segments and SUMO (Simulation of Urban MObility) road segments.
It follows the steps below:
1. Load TMC data and SUMO road segment data.
2. Add direction information to both TMC and SUMO road segments based on longitude comparison.
3. For each TMC segment, find the matching SUMO segments based on direction and longitude range.
4. Store the matched TMC-SUMO segment pairs in a file for further use.

The resulting mapping can be used for traffic flow estimation or simulation purposes.
"""

import pandas as pd
import numpy as np

# === 1. Load the Data ===
# Load SUMO road segment data (contains segment_id, longitude, and travel time)
sumo_df = pd.read_csv("Roosevelt_Calibration/Row_data/road_segments_conversion.csv")  # Adjust the path to your project directory
# Load TMC data (contains tmc_id, longitude, and travel time)
tmc_df = pd.read_csv("Roosevelt_Calibration/Fine_data/TMC_Identification.csv")  # Adjust the path to your project directory

# === 2. Parameter Settings ===
# Longitude tolerance to avoid floating-point errors during comparison
LON_TOLERANCE = 0.0001

# === 3. Add Direction Column to TMC and SUMO Data ===
# Assign direction ('east' or 'west') to TMC segments based on the longitude comparison
tmc_df['direction'] = tmc_df.apply(
    lambda row: 'east' if row['end_longitude'] > row['start_longitude'] else 'west',
    axis=1
)

# Assign direction to SUMO segments based on the longitude comparison
sumo_df['direction'] = sumo_df.apply(
    lambda row: 'east' if row['lon_end'] > row['lon_start'] else 'west',
    axis=1
)

# === 4. Match TMC and SUMO Segments ===
# Initialize an empty list to store the matching records
mapping_records = []

# Iterate over each TMC segment
for _, tmc_row in tmc_df.iterrows():
    tmc_id = tmc_row['tmc']
    lon_start = tmc_row['start_longitude']
    lon_end = tmc_row['end_longitude']
    direction = tmc_row['direction']

    lon_min = min(lon_start, lon_end)
    lon_max = max(lon_start, lon_end)

    # Find SUMO road segments that match the TMC segment in terms of direction and longitude range
    matched_segments = sumo_df[(
        sumo_df['direction'] == direction) &
        (sumo_df['lon_start'] >= lon_min - LON_TOLERANCE) &
        (sumo_df['lon_end'] <= lon_max + LON_TOLERANCE)
    ]

    # Store the matched segment information
    for _, sumo_row in matched_segments.iterrows():
        mapping_records.append({
            'tmc_id': tmc_id,
            'segment_id': sumo_row['edge_id'],
            'segment_lon_start': sumo_row['lon_start'],
            'segment_lon_end': sumo_row['lon_end'],
            'segment_direction': sumo_row['direction'],
            'tmc_lon_start': lon_start,
            'tmc_lon_end': lon_end,
            'tmc_direction': direction
        })

# === 5. Create DataFrame for Mapped Data ===
# Convert the list of mapping records to a DataFrame
mapping_df = pd.DataFrame(mapping_records)

# === 6. Save the Mapped Data ===
# Save the mapping results to a CSV file for further analysis or use
mapping_df.to_excel("Roosevelt_Calibration/Fine_data/TMC_Road_mapping_raw.xlsx", index=False)  # Save the file to an output folder
print("TMC-SUMO matching completed. Results saved to TMC_Road_mapping_raw.xlsx")
