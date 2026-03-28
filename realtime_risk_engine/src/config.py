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
    # (Removed MoneyViz placeholder)

# Feature Configuration
VELOCITY_WINDOW_DAYS = 90
TOTAL_LOOKBACK_DAYS = 180

# Fusion Configuration
FUSION_BEHAVIORAL_CENTER = F2_OPTIMAL_THRESHOLD
FUSION_BEHAVIORAL_WEIGHT = 0.30

# Meta-Physics Engine (Layer 3) Configuration
META_WINDOW_DAYS = 14           # WLS regression window (the "Goldilocks Zone")
META_ACCEL_WINDOW = 7           # Sub-window for acceleration calculation
META_R2_HIGH = 0.85             # R² above this = high-confidence crisis spiral
META_R2_LOW = 0.40              # R2 below this = noisy, suppress projection
META_DEFAULT_THRESHOLD = 0.75   # Risk score threshold for default projection
META_RECENCY_WEIGHT = 10.0      # Today's data has 10x weight of oldest data
META_VELOCITY_EPSILON = 0.001   # |beta| below this = STABLE (no meaningful trend)
META_MIN_SPAN_HOURS = 1.0       # Minimum time span required for regression
META_VELOCITY_CAP = 1.0         # Cap velocity at 100% risk change per day

# 9-Zone Trajectory Grid (3x3 Level/Trend)
# Level x [IMPROVING, STABLE, WORSENING]
META_ZONE_GRID = {
    ("LOW", "IMPROVING"):  "SAFE_RECOVERING",
    ("LOW", "STABLE"):     "SAFE_STABLE",
    ("LOW", "WORSENING"):  "SAFE_WORSENING",
    
    ("MID", "IMPROVING"):  "WATCH_RECOVERING",
    ("MID", "STABLE"):     "WATCH_STABLE",
    ("MID", "WORSENING"):  "WATCH_WORSENING",
    
    ("HIGH", "IMPROVING"): "CRITICAL_RECOVERING",
    ("HIGH", "STABLE"):    "CRITICAL_STABLE",
    ("HIGH", "WORSENING"): "CRITICAL_SPIRAL",
}
