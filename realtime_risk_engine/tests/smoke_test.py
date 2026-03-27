import pandas as pd
import os
import sys

# Ensure the parent directory is in the path so we can import our new modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference import VectorPredictor

def run_smoke_test():
    print("🚀 Initializing VECTOR Risk Engine Smoke Test...")
    
    try:
        predictor = VectorPredictor()
        print("✅ Model loaded successfully from models/behavioral_engine_v2.pkl")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # 1. Create Mock MoneyVis Data (UK Schema)
    # 180+ days needed for a "Ref Date" of today
    dates = pd.date_range(end=pd.Timestamp.now(), periods=50, freq='4D')
    
    mock_data = pd.DataFrame({
        "Transaction Date": dates.strftime("%d/%m/%Y"),
        "Transaction Description": ["Monthly Salary", "O2 Mobile Bill", "Grocery Store"] * 16 + ["Salary", "Rent"],
        "Transaction Type": ["BGC", "DD", "POS"] * 16 + ["FPI", "DD"],
        "Credit Amount": [2500.0, 0.0, 0.0] * 16 + [2500.0, 0.0],
        "Debit Amount": [0.0, 45.0, 30.0] * 16 + [0.0, 850.0],
        "Balance": [2500.0, 2455.0, 2425.0] * 16 + [4925.0, 4075.0]
    })
    
    print(f"✅ Generated {len(mock_data)} mock UK transactions.")

    # 2. Run Inference
    print("🔍 Running risk engine prediction...")
    results = predictor.predict_risk(mock_data)
    
    if "error" in results:
        print(f"❌ Prediction failed: {results['error']}")
    else:
        print("\n" + "="*40)
        print("RISK PREDICTION RESULT")
        print("="*40)
        print(f"Account ID: {results['account_id']}")
        print(f"Score:      {results['probability']*100:.2f}%")
        print(f"Status:     {'🚨 DISTRESSED' if results['is_distressed'] else '✅ SAFE'}")
        print("-"*40)
        print("BEHAVIORAL SIGNALS (Top 3):")
        for key in ["liquidity_momentum_v", "income_erosion_v", "overdraft_v"]:
            print(f"- {key}: {results['signals'][key]:.4f}")
        print("="*40)

if __name__ == "__main__":
    run_smoke_test()
