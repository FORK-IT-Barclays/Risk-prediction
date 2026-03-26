import os

# Universal Risk Constants
ECONOMIC_PPP_SCALER = 35.0  # Mapping metric from Berka (CZK) to UK (GBP) contexts
F2_OPTIMAL_THRESHOLD = 0.46  # Established via Optuna sweep for maximum recall

# Path Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "behavioral_engine_v2.pkl")
DATA_TEMPLATE = os.path.join(BASE_DIR, "data", "MoneyVis.csv")

# Feature Configuration
VELOCITY_WINDOW_DAYS = 90
TOTAL_LOOKBACK_DAYS = 180
