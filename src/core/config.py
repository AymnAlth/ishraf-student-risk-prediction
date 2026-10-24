"""Core application configuration and path settings."""

from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Paths
ARTIFACTS_DIR = PROJECT_ROOT / "ml" / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "training-runs" / "latest" / "model.joblib"
REPORT_PATH = ARTIFACTS_DIR / "training-runs" / "latest" / "report.json"
DATASET_PATH = ARTIFACTS_DIR / "student-risk-synthetic" / "student-risk-synthetic-training.csv"
DATASET_METADATA_PATH = ARTIFACTS_DIR / "student-risk-synthetic" / "student-risk-synthetic-training.metadata.json"

# App settings
APP_TITLE = "Ishraf Student Attendance Risk Early Warning System"
APP_VERSION = "2.0.0"
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# Model threshold defaults
DEFAULT_RISK_THRESHOLD = 0.3854  # Calibrated on Validation F1/PR-AUC
CRITICAL_RISK_THRESHOLD = 0.70
HIGH_RISK_THRESHOLD = 0.50
