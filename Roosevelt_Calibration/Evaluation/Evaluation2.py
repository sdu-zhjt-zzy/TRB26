"""
Functionality and Steps:
This script processes observed and predicted traffic flow data to compute error metrics (MAE, RMSE, MAPE)
and visualize the comparison between observed and predicted total vehicle flow in one minute.

Steps:
1. Load observed and predicted flow data.
2. Calculate predicted total flow using matrix multiplication.
3. Align time and merge observed and predicted data.
4. Calculate error metrics (MAE, RMSE, MAPE).
5. Save the predicted movement matrix and error comparison results.
6. Visualize the comparison between observed and predicted total flow.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# === Step 1: Load Observed Data ===
# Load observed total flow data (from the previous step)
observed_data_path = './data/observed_total_flow_per_time.xlsx'
observed_df = pd.read_excel(observed_data_path)

# === Step 2: Load Predicted Route Flow Data ===
# Load predicted route flow data (T × R)
predicted_data_path = './data/sag_time_result_pytorch_test2.xlsx'
pred_df = pd.read_excel(predicted_data_path)

# === Step 3: Calculate Predicted Total Flow ===
# Load route movement matrix (L × R)
route_matrix_path = './data/UP_route_movement_matrix.xlsx'
A_df = pd.read_excel(route_matrix_path)
routes = A_df.columns[3:]  # Assuming the first three columns are metadata
A_matrix = A_df[routes].to_numpy()

# Extract the predicted route flow values (T × R)
X_pred = pred_df[routes].to_numpy()

# Multiply the predicted values by the route matrix (T, L)
movement_pred = X_pred @ A_matrix.T  # (T, L)

# Sum over rows to get total flow for each time step (T,)
pred_total = movement_pred.sum(axis=1)

# === Step 4: Align Time and Merge Data ===
# Merge observed data and predicted data on the time column
merged_df = pd.merge(observed_df, pred_df[['Time']], on='Time')  # Ensure time alignment
merged_df['Predicted'] = pred_total
merged_df.rename(columns={'Value': 'Observed'}, inplace=True)

# === Step 5: Calculate Error Metrics ===
# Compute error metrics: MAE, RMSE, and MAPE
mae = mean_absolute_error(merged_df['Observed'], merged_df['Predicted'])
rmse = np.sqrt(mean_squared_error(merged_df['Observed'], merged_df['Predicted']))
mape = np.mean(np.abs((merged_df['Observed'] - merged_df['Predicted']) / (merged_df['Observed'] + 1e-5))) * 100  # Avoid division by zero

# Print error metrics
print(f"📊 MAE:  {mae:.2f}")
print(f"📉 RMSE: {rmse:.2f}")
print(f"📈 MAPE: {mape:.2f}%")

# === Step 6: Save the Predicted Movement Matrix ===
movement_pred_df = pd.DataFrame(movement_pred, columns=[f"Movement_{i+1}" for i in range(movement_pred.shape[1])])
movement_pred_df.insert(0, "Time", pred_df['Time'])
movement_pred_matrix_path = './results/restart/movement_pred_matrix_pytorch_test2.xlsx'
movement_pred_df.to_excel(movement_pred_matrix_path, index=False)

# === Step 7: Save the Error Comparison Results ===
error_comparison_path = './results/restart/error_pytorch_pytorch_test2.xlsx'
merged_df.to_excel(error_comparison_path, index=False)

# === Step 8: Visualize the Comparison of Observed vs Predicted Total Flow ===
plt.figure(figsize=(14, 6))
plt.plot(merged_df['Time'], merged_df['Observed'], label='Observed', linewidth=2)
plt.plot(merged_df['Time'], merged_df['Predicted'], label='Predicted', linestyle='--', linewidth=2)
plt.title('Total Flow: Observed vs Predicted')
plt.xlabel('Time')
plt.ylabel('Vehicle Count')

# Simplify the time format for better readability
time_labels = pd.to_datetime(merged_df['Time']).dt.strftime('%Y-%m-%d %H:%M')

# Set x-axis ticks interval (every 50th time point)
plt.xticks(ticks=np.arange(0, len(time_labels), step=50), labels=time_labels[::50], rotation=45)

plt.grid(True)
plt.legend()
plt.tight_layout()

# Save the plot
plot_path = './results/restart/sage_pytorch_test2.png'
plt.savefig(plot_path, dpi=300)
plt.show()

# === Step 9: Save Error Metrics to a Text File ===
metrics_txt_path = './results/restart/metrics_pytorch_test2.txt'
with open(metrics_txt_path, "w") as f:
    f.write(f" MAE:  {mae:.2f}\n")
    f.write(f" RMSE: {rmse:.2f}\n")
    f.write(f" MAPE: {mape:.2f}%\n")
