"""
This script optimizes traffic flow prediction and delay estimation using a deep learning model in PyTorch.
It performs the following steps:
1. Loads and preprocesses the traffic flow and route data.
2. Computes BPR (Bureau of Public Roads) parameters for delay estimation.
3. Defines a model using a parameterized traffic flow matrix.
4. Defines a loss function combining MSE for predicted and observed traffic flows and delays.
5. Optimizes the model using the Adam optimizer.
6. Saves the optimized results (route time predictions) to an Excel file.

Steps:
1. **Data Loading and Preprocessing**:
   - Reads multiple CSV and Excel files containing traffic flow data, route movement matrix, and TMC (Traffic Message Channel) data.
   - Formats timestamps to align data.

2. **BPR Parameter Calculation**:
   - Uses geodesic distance calculations and traffic characteristics like free-flow time and capacity to compute the BPR delay model parameters.

3. **Model Definition**:
   - Implements a deep learning model using PyTorch to predict traffic flow and delay.

4. **Loss Function**:
   - The loss function incorporates MSE for traffic flow prediction and delay prediction using BPR.

5. **Optimization**:
   - The model is trained using the Adam optimizer to minimize the loss function.

6. **Saving Results**:
   - The final optimized route times are saved as an Excel file for further analysis.
"""

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from datetime import datetime
from tqdm import tqdm
from geopy.distance import geodesic

### === 1. Load and preprocess data === ###
# Load data from CSV and Excel files
a_df = pd.read_excel('Roosevelt_Calibration/Fine_data/Sage_data.excel')  # c_obs
b_df = pd.read_excel('Roosevelt_Calibration/Fine_data/Route_Movement_matrix.xlsx')  # A matrix
c_df = pd.read_excel('Roosevelt_Calibration/Fine_data/UP_TMC_Road_mapping.xlsx')  # TMC mapping and BPR params
d_df = pd.read_csv('Roosevelt_Calibration/Fine_data/TMC_data.csv')  # y_obsd
e_df = pd.read_csv('Roosevelt_Calibration/Fine_data/TMC_Identification.csv')

# Format timestamps
a_df['Time'] = pd.to_datetime(a_df['Time']).dt.strftime('%Y-%m-%d %H:%M')
d_df['measurement_tstamp'] = pd.to_datetime(d_df['measurement_tstamp']).dt.strftime('%Y-%m-%d %H:%M')

# Get route list from A matrix
route_cols = b_df.columns[3:]
routes = list(route_cols)

# Map A matrix using (Node, Direction)
a_index = a_df[['Node', 'Direction']].drop_duplicates()
b_index = b_df[['Node', 'Direction']]
A_matrix = b_df[route_cols].values  # shape: (l, r)

# Build a mapping from (Node, Direction) to row index in A
index_map = {(node, direction): i for i, (node, direction) in enumerate(zip(b_index['Node'], b_index['Direction']))}


def get_c_obs_tensor():
    """Prepare traffic flow observations as a tensor"""
    time_list = sorted(a_df['Time'].unique())
    l_list = list(index_map.keys())
    c_obs = torch.zeros((len(l_list), len(time_list)))
    for i, (node, direction) in enumerate(l_list):
        df = a_df[(a_df['Node'] == node) & (a_df['Direction'] == direction)]
        for j, t in enumerate(time_list):
            val = df[df['Time'] == t]['Value']
            if not val.empty:
                c_obs[i, j] = float(val.values[0])
    return c_obs, time_list, l_list


c_obs, time_list, l_list = get_c_obs_tensor()


### === 2. Compute BPR parameters === ###

# Function to compute distance using geodesic
def compute_distance(lon1, lon2, lat=41.867):
    """Calculate distance in meters between two longitude points at a fixed latitude"""
    p1 = (lat, lon1)
    p2 = (lat, lon2)
    return geodesic(p1, p2).meters


# Free-flow time computation
tmc_to_miles = dict(zip(e_df['tmc'], e_df['miles']))
c_df['miles'] = c_df['tmc_id'].map(tmc_to_miles)
c_df['t_free'] = (c_df['miles'] / c_df['reference_speed']) * 3600

# Capacity adjustment for lane count
c_df['cap'] = c_df['cap'] * c_df['lane']


# Function to calculate overlap of TMC and segment
def compute_overlap(row):
    """Calculate the overlap between TMC segment and road segment based on longitude"""
    if row['segment_direction'].lower() != row['tmc_direction'].lower():
        return 0.0

    tmc_lon_start, tmc_lon_end = sorted([row['tmc_lon_start'], row['tmc_lon_end']])
    seg_lon_start, seg_lon_end = sorted([row['segment_lon_start'], row['segment_lon_end']])

    # Calculation of overlapping longitudes
    overlap_start = max(tmc_lon_start, seg_lon_start)
    overlap_end = min(tmc_lon_end, seg_lon_end)
    if overlap_end <= overlap_start:
        return 0.0

    # Calculate distance
    overlap_distance = compute_distance(overlap_start, overlap_end)
    tmc_total_distance = compute_distance(tmc_lon_start, tmc_lon_end)

    if tmc_total_distance == 0:
        return 0.0
    return round(overlap_distance / tmc_total_distance, 6)


c_df = c_df.copy()
c_df['radio'] = c_df.apply(compute_overlap, axis=1)

# Aggregate travel time observations
y_obs_df = d_df.groupby('measurement_tstamp')['travel_time_seconds'].sum().reset_index()
y_obs_df.columns = ['Time', 'y_obs']

# Align with c_obs to prevent time inconsistencies
y_obs_df = y_obs_df[y_obs_df['Time'].isin(time_list)]
y_obs_series = y_obs_df.set_index('Time').reindex(time_list).fillna(0)['y_obs']
y_obs = torch.tensor(y_obs_series.values, dtype=torch.float32)


### === 3. Define model === ###
class TrafficModel(nn.Module):
    def __init__(self, n_routes, n_times, A_matrix, c_df, l_list):
        super().__init__()
        self.x = nn.Parameter(torch.rand(n_routes, n_times))
        self.A = torch.tensor(A_matrix, dtype=torch.float32)
        self.c_df = c_df.reset_index(drop=True)
        self.l_list = l_list

    def forward(self):
        x = torch.relu(self.x).clamp(min=0)  # shape: (r, t)
        c_pred = self.A @ x  # shape: (l, t)

        # === Build node to flow map ===
        node_flow_map = {}
        for node in self.c_df['Node']:
            indices = [index for (n, d), index in index_map.items() if n == node]
            flow = c_pred[indices, :].sum(dim=0) if indices else torch.zeros(c_pred.shape[1])
            node_flow_map[node] = flow

        # Estimate BPR delay for each segment
        y_l_t_hat = []
        for _, row in self.c_df.iterrows():
            node = row['Node']
            t0 = row['t_free']
            cap = row['cap']
            radio = row.get('radio', 1.0)
            alpha, beta = 0.15, 4

            if node in node_flow_map:
                flow = node_flow_map[node]
                delay_cong = t0 * alpha * radio * ((flow / cap).clamp(min=1e-6)) ** beta
            else:
                delay_cong = torch.zeros_like(flow)

            y_l_t_hat.append(t0 * radio + delay_cong)

        y_l_t_hat = torch.stack(y_l_t_hat, dim=0)  # shape: (l, t)
        y_l_t_hat = y_l_t_hat.sum(dim=0)  # shape: [1330]

        return x, c_pred, y_l_t_hat


### === 4. Define loss function === ###
def loss_fn(c_pred, c_obs, y_l_t_hat, y_obs, l_list, c_df, lambda_):
    """Loss function combining traffic flow and delay prediction"""
    # MSE for traffic flow prediction
    c_loss = ((c_pred - c_obs) ** 2).mean()
    # MSE for predicted BPR delay vs observed delay
    y_loss = ((y_l_t_hat - y_obs) ** 2).mean()

    # Scale losses to avoid dimensional inconsistencies
    c_loss_scaled = c_loss / (c_obs.var() + 1e-6)
    y_loss_scaled = y_loss / (y_obs.var() + 1e-6)

    total_loss = y_loss_scaled #c_loss_scaled + lambda_ * y_loss_scaled
    return total_loss


### === 5. Optimization === ###
model = TrafficModel(n_routes=len(routes), n_times=len(time_list), A_matrix=A_matrix, c_df=c_df, l_list=l_list)
optimizer = Adam(model.parameters(), lr=0.01)
epochs = 1000

for epoch in tqdm(range(epochs)):
    optimizer.zero_grad()
    x, c_pred, y_l_t_hat = model()
    loss = loss_fn(c_pred, c_obs, y_l_t_hat, y_obs, l_list, c_df, lambda_=1)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

### === 6. Save results === ###
x_result = model.x.detach().clamp(min=0).round().int().numpy().T
x_df = pd.DataFrame(x_result, index=[str(t) for t in time_list], columns=routes)
x_df.to_excel("Roosevelt_Calibration/results/step2_results_pure2.xlsx")
print("Optimization complete and results saved.")
