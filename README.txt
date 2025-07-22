# Roosevelt_Calibration Project Documentation

This project is designed to model and evaluate the traffic flow on Roosevelt Rd, Chicago, IL 60607. The entire process is divided into three main parts: Data Preprocessing, Calibration Model, and Model Evaluation. Below is a detailed description of each step:

## 1. Data Preprocessing

### 1.1 Construct SUMO Simulation Network

Firstly, a proportionally scaled SUMO simulation network was created for Roosevelt Rd, Chicago, IL 60607. The latitude and longitude of this network correspond to the real-world road coordinates and are saved as `roosevelt.net.xml`.

### 1.2 Extract Route Data

A program named `Get_routes.py` was written to extract all the routes from the SUMO simulation network, and the results are saved as `roosevelt.rou.xml`.

### 1.3 Extract Segment Coordinates

Using the constructed SUMO network, the program `Node_extraction.py` was utilized to extract the latitude and longitude coordinates of the start and end points of each road segment, with the results saved in `Road_segments_conversion.xlsx`.

### 1.4 Extract Sage Node Data

Data for each Sage node in the network was obtained from https://vto.sagecontinuum.org/nodes. Data from February 28th, 16:00 to March 1st, 16:00 was selected and saved in `Sage_node_Row_data.xlsx`. Due to inconsistent data collection intervals, the data was resampled to a 1-minute interval using the `Resample_sage_row_data.py` program, and the final dataset is saved as `Sage_data.xlsx`.

## 2. Calibration Model

### 2.1 Step 1: Route Flow Calibration Using Movement Counts Only

The goal of this step is to estimate time-varying route flows using the observed movement counts from Sage nodes. The steps are as follows:

- Using the `roosevelt.net.xml` and `roosevelt.rou.xml` files, a `Route-Movement Matrix` was created by the script `step1_route-movement_matrix.py`, and the matrix was saved as `Route_Movement_matrix.xlsx`. After this step, the relationship matrix was manually filtered to include only the routes that intersect with Sage nodes.

- The `Route_Movement_matrix.xlsx` and `Sage_data.xlsx` were used, and the `cvxpy` library was applied to obtain an initial prediction matrix, which records the number of vehicles for each route at each time step. This result is saved as `Step1_result.xlsx`.

### 2.2 Step 2: Hybrid Calibration Using Additional TMC Travel Time

Additional data from HERE Technologies was introduced, which provides per-minute vehicle speed and observation time, saved in `TMC_data.csv`. As the TMC road segments differ from the SUMO network, the segments needed to be matched based on latitude and longitude. The steps are as follows:

- The script `step2_TMC_road_mapping.py` was used to map the real network to TMC segments based on latitude and longitude. The initial mapping was saved in `TMC_Road_mapping.xlsx`.

- Due to errors in latitude and longitude matching, some road segments were incorrectly matched. Therefore, we manually selected the correct matches and added parameters for each Sage node, including lane count (`lane`), basic capacity (`cap`), free-flow speed (`reference_speed`), and node ID (`Node`). This data was saved in `UP_TMC_Road_mapping.xlsx`.

- Finally, the script `step2_model_pytorch.py` was used to perform modeling with gradient descent using Pytorch. The files used were `Sage_data.xlsx`, `Route_Movement_matrix.xlsx`, `UP_TMC_Road_mapping.xlsx`, `TMC_data.csv`, and `TMC_Identification.csv`. The results are saved in `step2_result.xlsx`.

## 3. Model Evaluation

### 3.1 Step 1 Evaluation: Error Analysis per Node-Direction

The `Evaluation1.py` program analyzes the error for each node and direction, and generates a heatmap of predicted vehicle counts for each minute.

### 3.2 Step 2 Evaluation: Overall Error Analysis

The `Evaluation2.py` program analyzes the overall error in predicted vehicle counts for each minute and generates a line chart comparing predicted and observed values.
