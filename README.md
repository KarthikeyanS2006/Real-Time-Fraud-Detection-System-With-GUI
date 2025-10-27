💳 Project Theta: Real-Time NBFC Fraud Detection System1. Project OverviewProject Theta is a production-ready Machine Learning system designed to detect high-risk and fraudulent transactions for Non-Banking Financial Companies (NBFCs) in real-time. It addresses the critical need for low-latency, accurate fraud scoring by implementing a robust XGBoost classification model and deploying it via a high-speed FastAPI service.The project demonstrates a complete MLOps lifecycle, from data simulation and model training to API deployment and a dedicated GUI for analyst review.The ChallengeFraud losses require immediate action. Traditional systems often rely on batch processing or simple rules. Project Theta's solution provides a real-time risk score (under 50ms), allowing for instant decision-making.2. Core Features & ArchitectureKey ComponentsML Training (fraud_system.py): Trains an XGBoost Classifier on synthetic, highly imbalanced data. Handles class imbalance using scale_pos_weight.Real-Time API (api_service.py): A low-latency FastAPI endpoint for integrating the model into core payment processing systems.Hackathon Demo GUI (nbfc_fraud_gui.py): A simple Tkinter desktop application for manual testing and live demonstration of the risk scoring.Architecture FlowThe system operates in two main phases: Offline Training and Online Inference.Code snippetgraph TD
    A[Offline Training: fraud_system.py] --> B(Generate Synthetic Data & Features);
    B --> C(Train XGBoost Model & Fit Scaler);
    C --> D[Save Model & Scaler (models/)];
    
    subgraph Online Inference (Real-Time)
        D --> E[API Service: api_service.py]
        D --> F[GUI Demo: nbfc_fraud_gui.py]
        
        E --> G{Incoming Transaction Request (JSON)}
        F --> H{Manual GUI Input}
        
        G & H --> I[Load Real-Time Features (Mock Feature Store)]
        I --> J[Preprocess & Scale Input]
        J --> K[XGBoost Inference]
        K --> L[3-Tier Decision Engine]
        L --> M(Return Decision: BLOCK / ALERT / ACCEPT)
    end
3. Setup and InstallationPrerequisitesYou must have Python 3.9+ installed.DependenciesInstall all required libraries using the following pip command:Bashpip install pandas numpy xgboost scikit-learn imblearn fastapi uvicorn pydantic joblib

4. Usage & Execution GuideFollow these three steps to run the complete system.Step 1: Train the Model (Required First)This script creates the models/ directory and saves the trained machine learning assets (.joblib files).Bashpython fraud_system.py

Expected Output:
         Displays model performance (AUC, Recall, Precision) and confirms: Model and Scaler saved to: models/.Step 2 (Option A): Run the Hackathon Demo GUI 💻Use this for your live demonstration to quickly show how the system makes a decision based on four input features.Bashpython nbfc_fraud_gui.py
Demo CaseInput ValuesExpected DecisionNormal (ACCEPT)Amount: 300.0, 
TX Count: 3ACCEPT (Green)
Warning (ALERT)
Amount: 5500.0,
 TX Count: 7⚠️ 
 ALERT: Analyst Review (Orange) High Risk (BLOCK)Amount: 95000.0, TX Count: 15BLOCK (Red)Step 2 (Option B): Deploy the Real-Time API Service (Production Track)This demonstrates the system's low-latency performance as an HTTP API service.Bashuvicorn api_service:app --reload --port 8000
Test Endpoint: Send a POST request to http://127.0.0.1:8000/api/v1/predict_fraud.Bash# Example cURL Request for a HIGH-RISK transaction
curl -X POST "http://127.0.0.1:8000/api/v1/predict_fraud" \
     -H "Content-Type: application/json" \
     -d '{
       "user_id": "hackathon_test",
       "amount": 95000.0,
       "distance_from_last_tx": 10.0,
       "hour_of_day": 3,
       "count_tx_1h": 15
     }'
5. Model and Decision LogicFeatures UsedThe model uses four key features, simulating real-time inputs:amounthour_of_day (Time-based feature)distance_from_last_tx (Geospatial/Behavioral feature)count_tx_1h (Velocity Feature from Mock Feature Store)3-Tier Decision EngineThe system uses a two-level threshold approach to move beyond simple binary (fraud/not-fraud) classification, providing nuanced operational decisions: