# deployment/api_service.py
"""
Conceptual FastAPI service to serve the real-time fraud model.
To run:
1. Make sure 'fraud_system.py' has been run to create the 'models/' directory and files.
2. pip install fastapi uvicorn pandas joblib pydantic
3. uvicorn api_service:app --reload --port 8000
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import time
import os
import uvicorn 

# --- Configuration (Matching fraud_system.py) ---
MODEL_DIR = 'models'
MODEL_FILENAME = os.path.join(MODEL_DIR, 'fraud_detection_xgb_model.joblib')
SCALER_FILENAME = os.path.join(MODEL_DIR, 'scaler.joblib')
NUMERICAL_FEATURES = ['amount', 'hour_of_day', 'distance_from_last_tx', 'count_tx_1h']
PREDICTION_THRESHOLD = 0.80

# --- Pydantic Schema for Input Data ---
class TransactionRequest(BaseModel):
    user_id: str
    amount: float
    distance_from_last_tx: float
    hour_of_day: int
    count_tx_1h: int # This feature is required in the input payload

# --- API Initialization ---
app = FastAPI(title="Project Theta: Real-Time Fraud Scorer (NBFC)")

model_scorer = None
scaler_preprocessor = None

# --- Startup Event: Load Model ---
@app.on_event("startup")
def load_assets():
    """Load model and scaler into memory once at startup."""
    global model_scorer, scaler_preprocessor
    try:
        model_scorer = joblib.load(MODEL_FILENAME)
        scaler_preprocessor = joblib.load(SCALER_FILENAME)
        print("Model loaded successfully. Ready for inference.")
    except FileNotFoundError:
        print(f"ERROR: Model files not found in {MODEL_DIR}/. Run fraud_system.py first.")
        
# --- Health Check Endpoint ---
@app.get("/")
def read_root():
    return {"service_status": "Fraud Scorer is ONLINE", "model_loaded": model_scorer is not None}

# --- Prediction Endpoint ---
@app.post("/api/v1/predict_fraud")
def predict(tx_data: TransactionRequest):
    start_time = time.perf_counter()
    
    # 1. Prepare Data for Model
    input_data = tx_data.dict()
    
    # Filter features to match the model's expected input order
    X_predict = pd.DataFrame([[input_data[k] for k in NUMERICAL_FEATURES]], 
                             columns=NUMERICAL_FEATURES)
    
    # 2. Scale Features
    if scaler_preprocessor is None:
        raise HTTPException(status_code=503, detail="Model assets not loaded.")
        
    X_scaled = scaler_preprocessor.transform(X_predict)
    
    # 3. Model Inference
    try:
        score = model_scorer.predict_proba(X_scaled)[:, 1][0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    # 4. Decision Engine Logic (Traffic Light Logic)
    decision = 'ACCEPT' # GREEN LIGHT
    if score >= PREDICTION_THRESHOLD:
        decision = 'BLOCK' # RED LIGHT
    elif score >= (PREDICTION_THRESHOLD * 0.5):
        decision = '⚠️ ALERT: Analyst Review' # YELLOW LIGHT

    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000
    
    return {
        "transaction_id": tx_data.user_id,
        "fraud_score": round(score, 4),
        "decision": decision,
        "inference_latency_ms": round(latency_ms, 2)
    }

# Optional: Allows direct running via 'python api_service.py'
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)