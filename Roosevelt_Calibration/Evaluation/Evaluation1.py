"""
Functionality and Steps:
This script processes observed and predicted traffic flow data to calculate error metrics (MAE, RMSE, MAPE) 
for each Node-Direction pair and generates heatmaps for both observed and predicted flow values.

Steps:
1. Load observed traffic flow data (resampled data).
2. Load predicted traffic flow data (route time flow matrix).
3. Calculate predicted movement flow using matrix multiplication.
4. Process observed data and merge it with the predicted values.
5. Calculate error metrics (MAE, RMSE, MAPE) for each Node-Direction pair.
6. Generate and save heatmaps for both observed and predicted flows.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error

# === Step 1: Load Data ===
# Load observed data (Time, Node, Direction, Value)
observed_data_path = 'Roosevelt_Calibration/Fine_data/Sage_data.xlsx'
a_df = pd.read_excel(observed_data_path)
a_df['Time'] = pd.to_datetime(a_df['Time']).dt.strftime('%Y-%m-%d %H:%M')

# Load route time flow data (T × R)
route_time_flow_data_path = 'Roosevelt_Calibration/Results/Step1_result.xlsx'
b_df = pd.read_excel(route_time_flow_data_path)
b_df['Time'] = pd.to_datetime(b_df['Time']).dt.strftime('%Y-%m-%d %H:%M')

# Extract the predicted route time flow matrix (T × R)
X_pred = b_df.iloc[:, 1:].to_numpy()  # shape: (T, R)

# Load route movement matrix (L × R)
route_movement_matrix_path = 'Roosevelt_Calibration/Results/Route_Movement_matrix.xlsx'
c_df = pd.read_excel(route_movement_matrix_path)
route_matrix = c_df.iloc[:, 3:].to_numpy()  # shape: (L, R)

# Create a list of node-direction pairs
node_dir_list = list(zip(c_df['Node'], c_df['Direction']))

# === Step 2: Predict Movement Flow ===
# Matrix multiplication to get predicted movement flow (L, T)
movement_pred = route_matrix @ X_pred.T  # (L, T)

# === Step 3: Construct Prediction DataFrame (Node, Direction, Time, Predicted) ===
time_list = b_df['Time']  # The first column is time
L, T = movement_pred.shape

# Create a list of records to store prediction data
records = []
for l in range(L):
    node, direction = node_dir_list[l]
    for t in range(T):
        records.append({
            'Node': node,
            'Direction': direction,
            'Time': time_list[t],
            'Predicted': movement_pred[l, t]
        })
# Convert the list of records into a DataFrame
pred_df = pd.DataFrame(records)

# === Step 4: Process Observed Data (a_df) ===
# Group observed data by Node, Direction, and Time, then sum the values
a_grouped = a_df.groupby(['Node', 'Direction', 'Time'])['Value'].sum().reset_index()
a_grouped.rename(columns={'Value': 'Observed'}, inplace=True)

# === Step 5: Merge Predicted and Observed Values ===
# Merge the predicted and observed values into a single DataFrame
merged = pd.merge(pred_df, a_grouped, on=['Node', 'Direction', 'Time'], how='inner')

# === Step 6: Error Metrics Calculation (for each Node-Direction pair) ===
# Calculate error metrics for each Node-Direction pair (MAE, RMSE, MAPE)
group_metrics = []
grouped = merged.groupby(['Node', 'Direction'])
for (node, direction), group in grouped:
    y_true = group['Observed'].values
    y_pred = group['Predicted'].values
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-5))) * 100  # Avoid division by zero
    group_metrics.append({
        'Node': node,
        'Direction': direction,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE(%)': mape
    })

# Save the error metrics to an Excel file
metrics_df = pd.DataFrame(group_metrics)
error_metrics_path = 'Roosevelt_Calibration/Results/error_evaluation1_step1.xlsx'
metrics_df.to_excel(error_metrics_path, index=False)

# === Step 7: Generate Heatmaps ===
# Create a new column combining Node and Direction
merged['Node_Dir'] = merged['Node'] + "_" + merged['Direction']

# Create pivot tables for predicted and observed data
pivot_pred = merged.pivot_table(index='Node_Dir', columns='Time', values='Predicted', aggfunc='sum')
pivot_obs = merged.pivot_table(index='Node_Dir', columns='Time', values='Observed', aggfunc='sum')

# === Step 8: Heatmap Plotting Function ===
def plot_heatmap(data, title, save_path):
    """
    Plots and saves a heatmap of the provided data.
    """
    plt.figure(figsize=(18, 10))
    ax = sns.heatmap(data, cmap='YlOrRd', linewidths=0.2, linecolor='white')
    plt.title(title, fontsize=16)
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Node_Direction", fontsize=12)
    ax.set_xticks(np.arange(0, data.shape[1], 50))
    ax.set_xticklabels(data.columns[::50], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# Save the observed and predicted flow heatmaps
observed_heatmap_path = 'Roosevelt_Calibration/Results/Evaluation1_step1_real.png'
predicted_heatmap_path = 'Roosevelt_Calibration/Results/Evaluation1_step1_predicted.png'
plot_heatmap(pivot_obs, "Observed Flow Heatmap (Node-Direction vs Time)", observed_heatmap_path)
plot_heatmap(pivot_pred, "Predicted Flow Heatmap (Node-Direction vs Time)", predicted_heatmap_path)

