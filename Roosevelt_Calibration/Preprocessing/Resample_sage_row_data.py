"""
Resamples traffic data by aggregating values into 1-minute intervals based on a predefined time bin.
This script implements the following steps:
1. Load the original traffic data from a CSV file.
    The data includes time, direction, node, and traffic counts.
2. Clean the data by extracting the direction (left, right, or straight) and removing any null direction values.
3. Define the start time for time binning and convert the original data into 1-minute intervals by applying weighted resampling.
4. Aggregate the resampled data to compute the total traffic counts for each 1-minute interval by node and direction.
5. Save the resampled data to a new file.

This script is useful for aggregating traffic data into consistent time intervals for further analysis or simulation.
"""

import pandas as pd
import numpy as np
from datetime import timedelta

# Load the data from a CSV file and preprocess it
df = pd.read_csv("Roosevelt_Calibration/Row_data/Sage_node_Row_data.csv")  # Replace with your actual path
df['Time'] = pd.to_datetime(df['Time'])  # Convert the 'Time' column to datetime format

# Extract the 'Direction' from the 'Name' column and keep only the desired directions
df['Direction'] = df['Name'].str.extract(r'incoming\.(left|right|straight)')
df = df.dropna(subset=['Direction'])  # Remove rows where 'Direction' is NaN

# Define the start time as the earliest time in the dataset
start_time_base = df['Time'].min()


# Function to find the start of the 1-minute time bin for a given timestamp
def get_bin_start(t):
    delta = t - start_time_base
    num_bins = int(delta.total_seconds() // (1 * 60))  # Calculate the number of 1-minute bins
    return start_time_base + timedelta(minutes=1 * num_bins)


# Initialize an empty list to store the resampled data
resampled_data = []

# Iterate through the original data and resample it to 1-minute intervals
for _, row in df.iterrows():
    start_time = row['Time']
    end_time = start_time + timedelta(minutes=1)  # Define the 1-minute window for resampling
    node = row['Node']
    direction = row['Direction']
    count = row['Value']

    # Calculate the time bin that this data point falls into
    bin_start = get_bin_start(start_time)

    # While the bin start is within the range of the current data
    while bin_start < end_time:
        bin_end = bin_start + timedelta(minutes=1)

        # Calculate the overlap between the current data and the time bin
        overlap_start = max(start_time, bin_start)
        overlap_end = min(end_time, bin_end)
        overlap_seconds = (overlap_end - overlap_start).total_seconds()

        # If there is any overlap, calculate the weighted value for the current bin
        if overlap_seconds > 0:
            portion = overlap_seconds / 60  # Portion of the 1-minute bin the data covers
            weighted_value = round(count * portion)  # Round the weighted value to an integer

            # Append the resampled data to the list
            resampled_data.append({
                'Time': bin_start,
                'Node': node,
                'Direction': direction,
                'Value': weighted_value
            })

        # Move to the next time bin
        bin_start += timedelta(minutes=1)

# Create a DataFrame from the resampled data
resampled_df = pd.DataFrame(resampled_data)

# Aggregate the resampled data by summing the values for each (Time, Node, Direction)
final_df = (
    resampled_df
    .groupby(['Time', 'Node', 'Direction'])['Value']
    .sum()
    .reset_index()
)

# Print the first few rows of the aggregated data
print(final_df.head())

# Save the final resampled data to a new CSV file
final_df.to_excel("Roosevelt_Calibration/Fine_data/Sage_data.xlsx", index=False)  # Specify the output file path
