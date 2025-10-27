import pandas as pd
import numpy as np
import time
import joblib
import random
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, precision_score, roc_auc_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE 
import os

# --- Configuration ---
MODEL_DIR = 'models'
MODEL_FILENAME = os.path.join(MODEL_DIR, 'fraud_detection_xgb_model.joblib')
SCALER_FILENAME = os.path.join(MODEL_DIR, 'scaler.joblib')
RANDOM_STATE = 42
TARGET_COLUMN = 'is_fraud'
# Features that are normalized/scaled
NUMERICAL_FEATURES = ['amount', 'hour_of_day', 'distance_from_last_tx', 'count_tx_1h'] 
PREDICTION_THRESHOLD = 0.80 # Decision threshold for blocking/flagging (RED LIGHT)

# --- Mock Feature Store (Simulating Redis/In-Memory Cache) ---
FEATURE_STORE = {
    'user_A101': {'count_tx_1h': 2, 'avg_amount': 950.0},
    'user_B202': {'count_tx_1h': 8, 'avg_amount': 12000.0},
    'user_C303': {'count_tx_1h': 1, 'avg_amount': 50.0},
}

# ----------------------------------------------------
# 1. DATA SIMULATION FUNCTION
# ----------------------------------------------------

def generate_simulated_data(n_samples=10000, fraud_rate=0.01):
    """Generates synthetic, imbalanced transaction data."""
    np.random.seed(RANDOM_STATE)
    
    data = {}
    
    # 1. Core Transaction Features
    data['amount'] = np.exp(np.random.normal(7, 1.5, n_samples)) 
    data['hour_of_day'] = np.random.randint(0, 24, n_samples)
    data['distance_from_last_tx'] = np.random.exponential(1, n_samples)
    
    # 2. Velocity Feature 
    data['count_tx_1h'] = np.random.poisson(3, n_samples)
    
    df = pd.DataFrame(data)
    
    # 3. Introduce Fraud Labels 
    n_fraud = int(n_samples * fraud_rate)
    df[TARGET_COLUMN] = 0
    
    # Logic to select fraud samples (high amount + high velocity)
    high_amount_idx = df['amount'].nlargest(int(n_samples * 0.05)).index
    high_velocity_idx = df[df['count_tx_1h'] > 6].index
    
    potential_fraud_idx = high_amount_idx.intersection(high_velocity_idx).values
    
    if len(potential_fraud_idx) > n_fraud:
        fraud_indices = np.random.choice(potential_fraud_idx, size=n_fraud, replace=False)
    else:
        fraud_indices = potential_fraud_idx 
        remaining = n_fraud - len(fraud_indices)
        if remaining > 0:
            safe_indices = df.index.difference(fraud_indices)
            fraud_indices = np.concatenate([fraud_indices, np.random.choice(safe_indices, size=remaining, replace=False)])
            
    df.loc[fraud_indices, TARGET_COLUMN] = 1
    
    # Enhance the fraud samples characteristics
    df.loc[df[TARGET_COLUMN] == 1, 'amount'] = df.loc[df[TARGET_COLUMN] == 1, 'amount'] * np.random.uniform(2, 5, size=sum(df[TARGET_COLUMN]))
    df.loc[df[TARGET_COLUMN] == 1, 'distance_from_last_tx'] = df.loc[df[TARGET_COLUMN] == 1, 'distance_from_last_tx'] * np.random.uniform(5, 10, size=sum(df[TARGET_COLUMN]))
    
    return df

# ----------------------------------------------------
# 2. MODEL TRAINING AND EVALUATION
# ----------------------------------------------------

def train_and_evaluate_model(df):
    """Trains the XGBoost model, saves assets, and prints performance metrics."""
    print("--- 1. Model Training Pipeline ---")
    
    # 1. Split Data
    X = df[NUMERICAL_FEATURES]
    y = df[TARGET_COLUMN]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    # 2. Preprocessing: Fit and Transform Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=NUMERICAL_FEATURES)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=NUMERICAL_FEATURES)

    # 3. Handle Imbalance 
    n_pos = sum(y_train == 1)
    n_neg = sum(y_train == 0)
    scale_pos_weight = n_neg / n_pos
    
    # 4. Train Model (XGBoost)
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=100,
        random_state=RANDOM_STATE,
        scale_pos_weight=scale_pos_weight, 
        eval_metric='auc'
    )
    model.fit(X_train_scaled, y_train)
    
    # 5. Evaluate Model
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_pred_proba >= PREDICTION_THRESHOLD).astype(int)
    
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    
    print("\n[Model Performance on Test Set]")
    print(f"AUC-ROC Score: {auc_roc:.4f}")
    print(f"Recall (Fraud Captured): {recall:.4f}")
    print(f"Precision (Low False Positives): {precision:.4f}")
    print(f"\nClassification Report (Threshold: {PREDICTION_THRESHOLD}):\n", classification_report(y_test, y_pred))
    
    # 6. Save Model Assets (for deployment/use)
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_FILENAME)
    joblib.dump(scaler, SCALER_FILENAME)
    print(f"\nModel and Scaler saved to: {MODEL_DIR}/")
    
    return model, scaler

# ----------------------------------------------------
# 3. REAL-TIME INFERENCE (MOCK)
# ----------------------------------------------------

def load_deployed_assets():
    """Loads the model and scaler from disk."""
    model = joblib.load(MODEL_FILENAME)
    scaler = joblib.load(SCALER_FILENAME)
    return model, scaler

def mock_feature_lookup(user_id):
    """Mocks looking up real-time features from a store (e.g., Redis)."""
    return FEATURE_STORE.get(user_id, {'count_tx_1h': 3})['count_tx_1h']

def get_fraud_score_and_decision(transaction_data, model, scaler):
    """The core real-time prediction function."""
    start_time = time.perf_counter()
    
    user_id = transaction_data.get('user_id')
    count_tx_1h = mock_feature_lookup(user_id) 
    
    # Prepare Data Structure
    features_input = {
        'amount': [transaction_data['amount']],
        'hour_of_day': [transaction_data['hour_of_day']],
        'distance_from_last_tx': [transaction_data['distance_from_last_tx']],
        'count_tx_1h': [count_tx_1h] 
    }
    X_predict = pd.DataFrame(features_input, columns=NUMERICAL_FEATURES)
    
    # Scale Features
    X_scaled = scaler.transform(X_predict)
    
    # Model Inference
    score = model.predict_proba(X_scaled)[:, 1][0]
    
    # Decision Engine (Traffic Light Logic)
    decision = 'ACCEPT' # GREEN LIGHT
    if score >= PREDICTION_THRESHOLD:
        decision = 'BLOCK' # RED LIGHT
    elif score >= (PREDICTION_THRESHOLD * 0.5):
        decision = '⚠️ ALERT: Analyst Review' # YELLOW LIGHT

    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000
    
    return {
        'score': round(score, 4),
        'decision': decision,
        'latency_ms': round(latency_ms, 2)
    }

# ----------------------------------------------------
# MAIN EXECUTION
# ----------------------------------------------------

if __name__ == '__main__':
    # 1. Run the Training Pipeline
    df_simulated = generate_simulated_data()
    print(f"Generated {len(df_simulated)} samples. Fraud count: {sum(df_simulated[TARGET_COLUMN])}")
    
    trained_model, trained_scaler = train_and_evaluate_model(df_simulated)
    
    # 2. Simulate Real-Time Use with the Deployed Assets
    model_service, scaler_service = load_deployed_assets()

    print("\n--- 2. Real-Time Transaction Simulation (Traffic Light Cases) ---")

    # --- GREEN LIGHT: ACCEPT (Low Risk) ---
    tx_normal = {
        'user_id': 'user_A101', 
        'amount': 300.0, 
        'distance_from_last_tx': 0.1, 
        'hour_of_day': 10
    }
    result_normal = get_fraud_score_and_decision(tx_normal, model_service, scaler_service)
    print(f"\n[TX A - GREEN/ACCEPT] User: {tx_normal['user_id']} | Amount: {tx_normal['amount']}")
    print(f"  Result: Score={result_normal['score']}, Decision='{result_normal['decision']}'")

    # --- YELLOW LIGHT: ALERT (Medium Risk) ---
    tx_alert = {
        'user_id': 'user_B202', 
        'amount': 5500.0, 
        'distance_from_last_tx': 0.5, 
        'hour_of_day': 14
    }
    FEATURE_STORE['user_B202']['count_tx_1h'] = 8 # Ensure a high velocity for this case
    result_alert = get_fraud_score_and_decision(tx_alert, model_service, scaler_service)
    print(f"\n[TX B - YELLOW/ALERT] User: {tx_alert['user_id']} | Amount: {tx_alert['amount']}")
    print(f"  Result: Score={result_alert['score']}, Decision='{result_alert['decision']}'")
    
    # --- RED LIGHT: BLOCK (High Risk) ---
    tx_fraud = {
        'user_id': 'user_C303', 
        'amount': 95000.0, 
        'distance_from_last_tx': 10.0, 
        'hour_of_day': 3
    }
    FEATURE_STORE['user_C303']['count_tx_1h'] = 15 # Ensure max velocity for this case
    
    result_fraud = get_fraud_score_and_decision(tx_fraud, model_service, scaler_service)
    print(f"\n[TX C - RED/BLOCK] User: {tx_fraud['user_id']} | Amount: {tx_fraud['amount']}")
    print(f"  Result: Score={result_fraud['score']}, Decision='{result_fraud['decision']}'")