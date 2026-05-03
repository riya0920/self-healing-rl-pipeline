"""
Self-Healing RL Recommendation Agent — Configuration
Fill in your Reddit API credentials before running.
Get them at: https://www.reddit.com/prefs/apps (create a "script" type app)
"""

import os

# ============================================================
# REDDIT API CREDENTIALS — FILL THESE IN
# ============================================================
REDDIT_CLIENT_ID = os.environ.get("REDDIT_CLIENT_ID", "YOUR_CLIENT_ID_HERE")
REDDIT_CLIENT_SECRET = os.environ.get("REDDIT_CLIENT_SECRET", "YOUR_CLIENT_SECRET_HERE")
REDDIT_USER_AGENT = os.environ.get(
    "REDDIT_USER_AGENT", "SelfHealingRL/1.0 by YOUR_USERNAME"
)

# ============================================================
# SUBREDDIT CONFIGURATION
# ============================================================
# Phase 1: Training subreddits (agent learns these)
TRAINING_SUBREDDITS = ["technology", "sports", "politics", "science"]

# Phase 2: Drift subreddits (agent has never seen these)
DRIFT_SUBREDDITS = ["cooking", "fitness", "legaladvice", "medicine"]

# Phase 3: Heavy drift (completely different content)
HEAVY_DRIFT_SUBREDDITS = ["philosophy", "art", "gardening", "astronomy"]

# All categories the RL agent can recommend
CATEGORIES = TRAINING_SUBREDDITS + DRIFT_SUBREDDITS + HEAVY_DRIFT_SUBREDDITS

# ============================================================
# RL AGENT SETTINGS
# ============================================================
STATE_DIM = 64  # Embedding dimension for post features
ACTION_DIM = len(TRAINING_SUBREDDITS)  # Number of recommendation categories
HIDDEN_DIM = 128  # Hidden layer size
LEARNING_RATE = 1e-3
GAMMA = 0.99  # Discount factor
EPSILON_START = 1.0  # Exploration rate start
EPSILON_END = 0.05  # Exploration rate end
EPSILON_DECAY = 0.995  # Exploration decay per episode
BATCH_SIZE = 32
MEMORY_SIZE = 10000
TARGET_UPDATE = 10  # Update target network every N episodes

# ============================================================
# SERVER SETTINGS
# ============================================================
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000
DB_NAME = "rl_predictions.db"

# ============================================================
# MONITORING THRESHOLDS
# ============================================================
REWARD_DROP_THRESHOLD = 0.25  # Alert if mean reward drops by this much
ENGAGEMENT_THRESHOLD = 0.30  # Alert if engagement rate drops below this
LOW_REWARD_PCT_THRESHOLD = 0.40  # Alert if >40% of recommendations have low reward

# ============================================================
# PIPELINE DIRECTORY (auto-detected)
# ============================================================
PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(PIPELINE_DIR, DB_NAME)
