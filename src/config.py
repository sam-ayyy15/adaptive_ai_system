"""
Central configuration file for Adaptive AI System.
Contains all constants, paths, and toggleable settings.
"""
import os
from pathlib import Path
from typing import Dict, Any

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Training configuration
RANDOM_SEED: int = 42
TEST_SIZE: float = 0.2
VALIDATION_SIZE: float = 0.2
FEEDBACK_THRESHOLD: int = 100  # Minimum rows to trigger retraining

# Model selection thresholds
SMALL_DATASET_ROWS: int = 20000
SMALL_DATASET_FEATURES: int = 75

# Auto-detection parameters
CATEGORICAL_THRESHOLD: int = 20  # Max unique values to consider categorical
MISSING_VALUE_THRESHOLD: float = 0.5  # Drop columns with >50% missing

# Training parameters
MAX_ITER: int = 1000
N_JOBS: int = -1
CV_FOLDS: int = 5
TRAIN_TOP_K: int = 5  # Number of top recommended models to train/evaluate

# Dashboard settings
DASHBOARD_PORT: int = 8501
REFRESH_INTERVAL: int = 5  # seconds

# File paths
METADATA_FILE: str = "metadata.json"
FEEDBACK_FILE: str = "feedback.csv"
TRAINING_LOG_FILE: str = "training_log.json"

# Model configurations
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "elastic_net": {
        "param_grid": {
            "alpha": [0.01, 0.1, 1.0, 10.0],
            "l1_ratio": [0.1, 0.5, 0.9]
        }
    },
    "linear_regression": {
        "param_grid": {
            # LinearRegression has few hyperparameters; include normalize-like behavior
            # via fit_intercept toggle for completeness
            "fit_intercept": [True, False]
        }
    },
    "random_forest": {
        "param_grid": {
            "n_estimators": [100, 200, 500],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5, 10]
        }
    },
    "gradient_boosting": {
        "param_grid": {
            "n_estimators": [100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7]
        }
    },
    "xgboost": {
        "param_grid": {
            "n_estimators": [100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7],
            "subsample": [0.8, 1.0]
        }
    }
}