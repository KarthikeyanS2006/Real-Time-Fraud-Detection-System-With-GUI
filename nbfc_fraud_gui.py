import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import pandas as pd
import numpy as np
import os

# --- Configuration (Matching your other files) ---
MODEL_DIR = 'models'
MODEL_FILENAME = os.path.join(MODEL_DIR, 'fraud_detection_xgb_model.joblib')
SCALER_FILENAME = os.path.join(MODEL_DIR, 'scaler.joblib')
NUMERICAL_FEATURES = ['amount', 'hour_of_day', 'distance_from_last_tx', 'count_tx_1h']
PREDICTION_THRESHOLD = 0.80

# --- Global Assets ---
try:
    # Load assets once when the application starts
    MODEL = joblib.load(MODEL_FILENAME)
    SCALER = joblib.load(SCALER_FILENAME)
except FileNotFoundError:
    # Handle the case where the model hasn't been trained yet
    messagebox.showerror("Error", 
                         f"Model or Scaler files not found in '{MODEL_DIR}'. \n"
                         "Please run 'fraud_system.py' first to train and save the assets.")
    exit()

# --- Prediction Logic Function ---
def predict_fraud():
    """Fetches user input, runs the prediction logic, and displays the result."""
    
    entry_map = {
        'amount': entry_amount,
        'distance_from_last_tx': entry_distance,
        'hour_of_day': entry_hour,
        'count_tx_1h': entry_count_tx
    }
    
    try:
        # 1. Get User Input and Validation
        input_values = {}
        for feature, entry in entry_map.items():
            value_str = entry.get()
            if not value_str:
                raise ValueError(f"'{feature.replace('_', ' ').title()}' cannot be empty.")
            
            if feature == 'hour_of_day' or feature == 'count_tx_1h':
                input_values[feature] = int(value_str)
            else:
                input_values[feature] = float(value_str)

        # 2. Prepare Data for Model
        input_list = [input_values[k] for k in NUMERICAL_FEATURES]
        X_predict = pd.DataFrame([input_list], columns=NUMERICAL_FEATURES)
        
        # 3. Scale Features
        X_scaled = SCALER.transform(X_predict)
        
        # 4. Model Inference
        score = MODEL.predict_proba(X_scaled)[:, 1][0]
        
        # 5. Decision Engine Logic (Traffic Light Logic)
        decision = 'ACCEPT'
        color = "green"
        if score >= PREDICTION_THRESHOLD:
            decision = 'BLOCK'
            color = "red"
        elif score >= (PREDICTION_THRESHOLD * 0.5):
            decision = '⚠️ ALERT: Analyst Review' # YELLOW LIGHT Message
            color = "orange" # Sets the color to Orange/Yellow

        # 6. Display Result
        result_text = (
            f"FRAUD SCORE: {score:.4f}\n"
            f"RISK DECISION: **{decision}**"
        )
        result_label.config(text=result_text, 
                            foreground=color)
        
    except ValueError as e:
        messagebox.showerror("Invalid Input", f"Please enter valid numbers: {e}")
    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred: {e}")

# --- Tkinter GUI Setup ---
root = tk.Tk()
root.title("Project Theta: NBFC Fraud Detection Simulator")

input_frame = ttk.Frame(root, padding="20")
input_frame.pack(padx=10, pady=10, fill="x")

# Create and place Labels and Entry Fields
# Transaction Amount
ttk.Label(input_frame, text="Transaction Amount:").grid(row=0, column=0, sticky="w", pady=5)
entry_amount = ttk.Entry(input_frame, width=30)
entry_amount.grid(row=0, column=1, pady=5)
entry_amount.insert(0, "5500.0") # Set default to YELLOW ALERT test case for demo

# Distance from Last Transaction
ttk.Label(input_frame, text="Distance (km from last TX):").grid(row=1, column=0, sticky="w", pady=5)
entry_distance = ttk.Entry(input_frame, width=30)
entry_distance.grid(row=1, column=1, pady=5)
entry_distance.insert(0, "0.5") 

# Hour of Day (0-23)
ttk.Label(input_frame, text="Hour of Day (0-23):").grid(row=2, column=0, sticky="w", pady=5)
entry_hour = ttk.Entry(input_frame, width=30)
entry_hour.grid(row=2, column=1, pady=5)
entry_hour.insert(0, "14") 

# TX Count in Last 1 Hour (Velocity)
ttk.Label(input_frame, text="TX Count (last 1 hour):").grid(row=3, column=0, sticky="w", pady=5)
entry_count_tx = ttk.Entry(input_frame, width=30)
entry_count_tx.grid(row=3, column=1, pady=5)
entry_count_tx.insert(0, "8") 

# Prediction Button
predict_button = ttk.Button(root, text="🚀 Check Fraud Risk (Predict)", command=predict_fraud)
predict_button.pack(pady=15, fill="x", padx=20)

# Result Label
result_label = ttk.Label(root, 
                         text="Enter transaction details and click 'Check Fraud Risk'.", 
                         font=('Arial', 14, 'bold'))
result_label.pack(pady=10)

# Start the main loop
root.mainloop()