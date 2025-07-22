"""
This script extracts edge information from a SUMO network XML file, performs coordinate transformation
from SUMO local coordinate system to WGS84 (latitude, longitude), and then processes the data to create road segments.
The output is saved in a CSV file containing the road segments with their start and end coordinates.

Steps:
1. Parse SUMO network file to extract edge and lane data.
2. Transform edge coordinates from SUMO local coordinate system to WGS84 (latitude, longitude).
3. Save the transformed data in a CSV file.
4. Group edges by `edge_id` to create road segments.
5. Save the final road segments into a CSV file.
"""

import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from pyproj import Transformer
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

# === 1. Coordinate Transformation Setup ===
# Set file paths for the SUMO network and output files
net_file = 'Roosevelt_Calibration/Row_data/roosevelt.net.xml'  # Change this path to your SUMO network file
output_csv = 'Roosevelt_Calibration/Row_data/road_segments.csv'  # Output file for the road segments

# Set up the coordinate transformer: SUMO local coordinate system → WGS84 (latitude, longitude)
transformer = Transformer.from_crs("EPSG:32616", "EPSG:4326", always_xy=True)

# === 2. Extract and Transform SUMO Network Edge Coordinates ===
# Parse the SUMO network XML file
tree = ET.parse(net_file)
root = tree.getroot()

# List to store edge data
edge_data = []

print("Extracting and transforming edge coordinates...")

# Iterate through each edge in the network file
for edge in tqdm(root.findall('edge')):
    edge_id = edge.get('id')

    # Skip internal edges and walking areas
    if edge_id.startswith(':') or edge.get('function') == 'walkingarea':
        continue

    for lane in edge.findall('lane'):
        shape = lane.get('shape')
        if shape is None:
            continue

        points = shape.strip().split()
        if len(points) < 2:
            continue

        # Extract the first and last points from the shape
        x1, y1 = map(float, points[0].split(','))
        x2, y2 = map(float, points[-1].split(','))

        # Transform the coordinates from the local coordinate system to WGS84
        lon1, lat1 = transformer.transform(x1, y1)
        lon2, lat2 = transformer.transform(x2, y2)

        edge_data.append({
            'edge_id': edge_id,
            'x_start': x1, 'y_start': y1,
            'x_end': x2, 'y_end': y2,
            'lon_start': lon1, 'lat_start': lat1,
            'lon_end': lon2, 'lat_end': lat2
        })

# === 3. Linear Regression for Coordinate Transformation (x, y) → lon, lat ===
# Sample data for (x, y) to lon, lat conversion (for further transformation)
sample_data = [
    [1424.64, 975.80, -87.625022, 41.866520],
    [1460.42, 1074.47, -87.624599, 41.867411],
    [1468.13, 976.30, -87.624498, 41.866527],
    [1578.84, 1074.45, -87.623172, 41.867419],
]
sample_data = np.array(sample_data)
X = sample_data[:, :2]
lon = sample_data[:, 2]
lat = sample_data[:, 3]

# Linear regression models to convert (x, y) to lon, lat
lon_model = LinearRegression().fit(X, lon)
lat_model = LinearRegression().fit(X, lat)

# Function to convert (x, y) to lon, lat using the trained models
def xy_to_lonlat(x, y):
    return lon_model.predict([[x, y]])[0], lat_model.predict([[x, y]])[0]

# === 4. Extract Non-Internal Edges and Convert Coordinates ===
data = []

# Iterate over each edge to extract and convert coordinates
for edge in root.findall("edge"):
    if edge.get("function") == "internal":
        continue
    edge_id = edge.get("id")

    for lane in edge.findall("lane"):
        shape_str = lane.get("shape")
        if not shape_str:
            continue
        shape_coords = shape_str.strip().split(" ")

        for point in shape_coords:
            x_str, y_str = point.split(",")
            x, y = float(x_str), float(y_str)
            lon, lat = xy_to_lonlat(x, y)

            data.append({
                "edge_id": edge_id,
                "lane_id": lane.get("id"),
                "x": x,
                "y": y,
                "lon": lon,
                "lat": lat
            })

# === 5. Group by Edge ID and Save Road Segments ===
# Group by edge_id and extract start and end points for each road segment
road_segments = []

for edge_id, group in pd.DataFrame(data).groupby('edge_id'):
    # Get the start point (first point in the group)
    x_start = group.iloc[0]['x']
    y_start = group.iloc[0]['y']
    lon_start = group.iloc[0]['lon']
    lat_start = group.iloc[0]['lat']

    # Get the end point (last point in the group)
    x_end = group.iloc[-1]['x']
    y_end = group.iloc[-1]['y']
    lon_end = group.iloc[-1]['lon']
    lat_end = group.iloc[-1]['lat']

    road_segments.append({
        'edge_id': edge_id,
        'x_start': x_start,
        'y_start': y_start,
        'x_end': x_end,
        'y_end': y_end,
        'lon_start': lon_start,
        'lat_start': lat_start,
        'lon_end': lon_end,
        'lat_end': lat_end
    })

# Create a DataFrame for the road segments
road_segments_df = pd.DataFrame(road_segments)

# Save the road segments data to a CSV file
road_segments_df.to_csv(output_csv, index=False)

print(f" Road segments mapping completed, results saved to '{output_csv}'")
