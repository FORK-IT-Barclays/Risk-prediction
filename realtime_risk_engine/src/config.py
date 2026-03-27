import os

# Universal Risk Constants
ECONOMIC_PPP_SCALER = 35.0
F2_OPTIMAL_THRESHOLD = 0.46
HISTORIAN_SENTINEL_ANNUAL_INC = 62000.0
HISTORIAN_SENTINEL_DTI = 17.0
EPS = 1e-6

# Path Configuration
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = os.path.dirname(SRC_DIR)
MODELS_DIR = os.path.join(ENGINE_DIR, "models")
DATA_DIR = os.path.join(ENGINE_DIR, "data")

BEHAVIORAL_MODEL_PATH = os.path.join(MODELS_DIR, "behavioral_engine_v2.pkl")
HISTORIAN_MODEL_PATH = os.path.join(MODELS_DIR, "universal_historian_v1.pkl")
HISTORIAN_FEATURE_MAP_PATH = os.path.join(MODELS_DIR, "universal_features_map.pkl")
DATA_TEMPLATE = os.path.join(DATA_DIR, "MoneyVis.csv")

# Feature Configuration
VELOCITY_WINDOW_DAYS = 90
TOTAL_LOOKBACK_DAYS = 180

# Fusion Configuration
FUSION_BEHAVIORAL_CENTER = F2_OPTIMAL_THRESHOLD
FUSION_BEHAVIORAL_WEIGHT = 0.30
