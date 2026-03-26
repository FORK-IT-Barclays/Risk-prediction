import joblib
import os

# Define workspace and model path
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "Output_Artifacts", "behavioral_engine_v2.pkl")

# Load model data
if os.path.exists(model_path):
    m_data = joblib.load(model_path)
    model = m_data['model']
    features = m_data['features']
    
    # Get importances
    importances = dict(zip(features, model.feature_importances_))
    
    print("--- Feature Importances ---")
    for feat, imp in sorted(importances.items(), key=lambda x: x[1], reverse=True):
        print(f"{feat:20}: {imp:.4f}")
else:
    print(f"Error: Model not found at {model_path}")
