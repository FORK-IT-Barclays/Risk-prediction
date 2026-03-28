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

# 9-Zone Strategic Matrix (Velocity [Rising, Stable, Falling] x Accel [Positive, Zero, Negative])
META_ZONE_GRID = {
    # Rising (v > 0)
    ("WORSENING", "POSITIVE"): "EXPONENTIAL_CRASH",
    ("WORSENING", "ZERO"):     "LINEAR_WORSENING",
    ("WORSENING", "NEGATIVE"): "REACHING_PEAK",
    
    # Stable (v ~= 0)
    ("STABLE", "POSITIVE"):    "EARLY_VOLATILITY",
    ("STABLE", "ZERO"):        "SOLID_STABILITY",
    ("STABLE", "NEGATIVE"):    "LATE_STABILITY",
    
    # Falling (v < 0)
    ("IMPROVING", "POSITIVE"): "RECOVERY_STALL",
    ("IMPROVING", "ZERO"):     "LINEAR_RECOVERY",
    ("IMPROVING", "NEGATIVE"): "TOTAL_REHAB",
}

# Dynamics Intervention Tiers (Action Prioritization)
# RED (1) = High Danger, YELLOW (2) = Monitor/Nudge, GREEN (3) = Healthy/Growth
META_TIER_MAPPING = {
    # Tier 1: RED (Immediate Action)
    "EXPONENTIAL_CRASH":  "TIER_1_RED",
    "LINEAR_WORSENING":   "TIER_1_RED",
    "EARLY_VOLATILITY":   "TIER_1_RED",
    
    # Tier 2: YELLOW (Intervention Needed)
    "REACHING_PEAK":      "TIER_2_YELLOW",
    "RECOVERY_STALL":     "TIER_2_YELLOW",
    "LATE_STABILITY":     "TIER_2_YELLOW",
    
    # Tier 3: GREEN (Healthy/Growth)
    "SOLID_STABILITY":    "TIER_3_GREEN",
    "LINEAR_RECOVERY":    "TIER_3_GREEN",
    "TOTAL_REHAB":        "TIER_3_GREEN",
}
