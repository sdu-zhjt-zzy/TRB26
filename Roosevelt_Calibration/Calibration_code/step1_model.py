"""
This script estimates the route flows based on route-movement matrix (A matrix) and observed data (c_lt) using optimization.
The following steps are executed:
1. Load the route-movement matrix (A matrix) from an Excel file.
2. Load the observed data (c_lt) from another Excel file.
3. For each time point, solve an optimization problem to estimate the route flows.
4. Post-process the results to remove small floating-point errors and round the values.
5. Save the estimated route flows to an Excel file.

This process allows for refining traffic flow predictions using optimization techniques.
"""

import pandas as pd
import numpy as np
import cvxpy as cp
from tqdm import tqdm

# === 1. Load the route-movement matrix (A matrix) ===
# Read the route-movement matrix from an Excel file
A_df = pd.read_excel("Roosevelt_Calibration/Fine_data/Route_Movement_matrix.xlsx")  # Adjust the path to your project directory
routes = A_df.columns[3:]  # Routes start from the 4th column
movements = A_df[['Node', 'Direction']].astype(str).agg('_'.join, axis=1)  # Combine Node and Direction into a single column for identification
print(movements)

# === 2. Load the observed data (c_lt) ===
# Read the observed data from an Excel file
c_df = pd.read_excel("Roosevelt_Calibration/Fine_data/Sage_data.xlsx")  # Adjust the path to your project directory
c_df['Node_Dir'] = c_df[['Node', 'Direction']].astype(str).agg('_'.join, axis=1)  # Combine Node and Direction

# === 3. Extract all unique time points ===
# Get all the unique time points from the observed data
time_points = sorted(c_df['Time'].unique())

results = []  # To store the results

# === 4. Loop through each time point and solve the optimization problem ===
# For each time point, estimate the route flows
for t in tqdm(time_points, desc="Estimating route flows"):
    c_t = c_df[c_df['Time'] == t]  # Filter the data for the current time point

    # Construct c vector (observed vehicle count)
    c_vec = A_df[['Node', 'Direction']].astype(str).agg('_'.join, axis=1).map(
        dict(zip(c_t['Node_Dir'], c_t['Value']))
    ).fillna(0).to_numpy()  # Map the observed values to the matrix, filling missing values with 0

    print(c_vec)

    # Construct A matrix (route-movement matrix)
    A_matrix = A_df[routes].to_numpy()

    # Define the optimization variable (non-negative continuous variable)
    x = cp.Variable(len(routes), nonneg=True)

    # Build the objective function (minimizing the squared error between A * x and observed values c_vec)
    objective = cp.Minimize(cp.sum_squares(A_matrix @ x - c_vec))

    # Define the problem
    prob = cp.Problem(objective)

    # Solve the problem
    prob.solve()

    # === Post-processing ===
    # Remove small floating point errors and round to the nearest integer
    x_val = np.array(x.value)  # Get the values of the variable
    x_val[np.abs(x_val) < 1e-5] = 0  # Consider values smaller than a threshold as 0
    x_val = np.round(x_val)  # Round the values to the nearest integer

    # Append the results for the current time point
    results.append([t] + x_val.tolist())

# === 5. Save the results ===
# Convert the results into a DataFrame
results_df = pd.DataFrame(results, columns=['Time'] + list(routes))

# Save the results to an Excel file
results_df.to_excel("Roosevelt_Calibration/results/Step1_results.xlsx", index=False)  # Save the results to a file in the output folder
