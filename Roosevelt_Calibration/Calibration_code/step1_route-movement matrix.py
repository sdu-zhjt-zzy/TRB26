"""
Generates a Route-Movement Matrix from a SUMO network and routes.
This script performs the following steps:
1. Load the SUMO network file and parse the edges.
    - The network file is provided in .net.xml format.
2. Parse the `.rou.xml` file to extract the routes and their respective edges.
3. Construct the Route-Movement Matrix, where each entry indicates whether a movement is part of a route.
4. Output the resulting Route-Movement Matrix as a DataFrame and save it as a CSV file.

This script is useful for analyzing traffic flow and route assignments in SUMO-based simulations.
"""

import pandas as pd
import xml.etree.ElementTree as ET
from sumolib.net import readNet

# ======== Step 1: Load the SUMO network file ========
# Read the SUMO network XML file to get all edges
net = readNet("Roosevelt_Calibration/Fine_data/roosevelt.net.xml")  # Replace with your actual path

# ======== Step 2: Parse all movements (from_edge -> to_edge) ========
# Initialize lists to store the movements and their indices
movement_list = []
movement_index_map = {}

# Iterate through all edges in the network and store their connections
for edge in net.getEdges():
    from_edge = edge.getID()  # Get the starting edge
    for conn in edge.getOutgoing():
        to_edge = conn.getID()  # Get the outgoing edge
        movement_id = f"{from_edge}->{to_edge}"  # Create a movement ID
        if movement_id not in movement_index_map:
            movement_index_map[movement_id] = len(movement_list)
            movement_list.append((from_edge, to_edge))

# ======== Step 3: Parse the .rou.xml file for routes ========
# Function to parse the .rou.xml file and extract route information
def parse_routes_from_xml(xml_path):
    tree = ET.parse(xml_path)  # Parse the XML file
    root = tree.getroot()  # Get the root element of the XML
    route_dict = {}  # Dictionary to store route information
    for route in root.iter("route"):  # Iterate over all <route> elements
        route_id = route.attrib.get("id")  # Get the route ID
        edges = route.attrib.get("edges")  # Get the sequence of edges in the route
        if route_id and edges:
            edge_list = edges.strip().split()  # Split the edges by spaces
            route_dict[route_id] = edge_list  # Store the route ID and its edges
    return route_dict

# Parse routes from the provided .rou.xml file
route_dict = parse_routes_from_xml("Roosevelt_Calibration/Fine_data/roosevelt.rou.xml")  # Replace with your actual path
route_list = list(route_dict.keys())  # Get the list of route IDs

# ======== Step 4: Construct the Route-Movement Matrix ========
# Create an empty DataFrame to store the Route-Movement Matrix
A = pd.DataFrame(0, index=movement_index_map.keys(), columns=route_list)

# Iterate over the routes and assign values to the matrix based on the movements
for route_id, edge_seq in route_dict.items():
    for i in range(len(edge_seq) - 1):
        from_edge = edge_seq[i]  # Get the starting edge for this segment
        to_edge = edge_seq[i + 1]  # Get the ending edge for this segment
        movement_id = f"{from_edge}->{to_edge}"  # Create the movement ID
        if movement_id in A.index:  # Check if this movement exists in the matrix
            A.loc[movement_id, route_id] = 1  # Set the matrix value to 1 for the corresponding route

# ======== Step 5: Output the Result ========
# Print the first few rows of the Route-Movement Matrix
print("Route-Movement Matrix A[l, r]:")
print(A.head())

# Save the Route-Movement Matrix as a CSV file for easy viewing
A.to_excel("Roosevelt_Calibration/Fine_data/Route_Movement_matrix.xlsx")  # Specify the desired output directory
