# 💳 Project Theta:

# Real-Time NBFC Fraud Detection System .

# Project Overview 
  Project Theta is a production-simulated Machine Learning system designed for Non-Banking Financial Companies (NBFCs) to perform real-time fraud detection. It delivers a transaction risk score in under $\mathbf{50ms}$, allowing for instant, three-tiered decision-making.The project addresses the critical issue of fraud loss by moving beyond traditional batch processing to a high-speed, continuous monitoring system.Key Value PropositionOur system is a Real-Time Security Guard that uses a "Traffic Light" system to manage risk, ensuring maximum fraud capture (high Recall) while minimizing false customer blocks (high Precision).

# 2. Architecture and Components: 
  The system is split into two primary environments: the Offline Training Pipeline and the Online Inference Service.
  <div  align="center" > 
 <img width="476" height="189" alt="image"src="https://github.com/user-attachments/assets/494631a3-743a-408b-80db-083027ed4648" /> </div>
<br>
<br> 
# 3. Setup and Execution Guide
# 3.1. Prerequisites
Ensure you have Python 3.9+ installed.

# 3.2. Installation
Install all required dependencies using pip:

```bash
pip install pandas numpy xgboost scikit-learn imblearn fastapi uvicorn pydantic joblib
```

# 3.3. Execution Steps
You must run these steps in order:

# Step 1: Train the Model (Initialize Assets)
This script trains the XGBoost model and saves the necessary files (fraud_detection_xgb_model.joblib and scaler.joblib) into the newly created models/ directory.

```Bash

python fraud_system.py
```
# Step 2 (Recommended for Demo): Run the GUI Simulator 💻
This is the best tool for demonstrating the "Traffic Light" logic in your presentation.

```Bash

python nbfc_fraud_gui.py
```
# Step 3 (Optional: Production Demo): Run the Real-Time API Service
To demonstrate the system's deployment capability in a production environment:

```Bash

uvicorn api_service:app --reload --port 8000
```
API Test: Access http://127.0.0.1:8000/ for a health check or post a transaction to http://127.0.0.1:8000/api/v1/predict_fraud.
<br>
# 4. The 3-Tier Decision Engine (Traffic Light System)
The core of Project Theta is the decision logic, which uses two thresholds to categorize risk instead of a simple binary classification.
<div align="center">
<img width="844" height="254" alt="image" src="https://github.com/user-attachments/assets/959f7041-7b87-47b6-b5f6-a1162f97454f" /> </div>
<br><br>

# Demo Test Cases (Use in the GUI)
<div align="center">
  <img width="735" height="168" alt="image" src="https://github.com/user-attachments/assets/1243e6b1-d19e-42e8-9798-fff10d3fe826" /> 
</div>

