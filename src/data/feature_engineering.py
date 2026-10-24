"""Domain-driven feature engineering module for Ishraf Student Risk model.

Enforces zero-leakage rules and calculates 22 high-impact interaction and ratio features.
Works consistently across both batch training datasets and real-time single-record inference.
"""

from __future__ import annotations
from typing import Any, Dict, List, Union
import numpy as np
import pandas as pd
from src.core.exceptions import InvalidFeatureError

# Contract columns
IDENTIFIER_COLUMNS = ["student_key", "snapshot_date"]
LABEL_COLUMNS = [
    "label_at_risk_next_14d",
    "label_observation_count",
    "label_future_attendance_percent",
]
FORBIDDEN_COLUMNS = set(IDENTIFIER_COLUMNS + LABEL_COLUMNS)

RAW_FEATURE_COLUMNS = [
    "grade_level_id",
    "days_since_enrollment",
    "lookback_attendance_records",
    "lookback_present_count",
    "lookback_absent_count",
    "lookback_late_count",
    "lookback_excused_count",
    "lookback_attendance_percent",
    "lookback_assessment_count",
    "lookback_assessment_average_percent",
    "lookback_assessment_minimum_percent",
    "lookback_homework_due_count",
    "lookback_homework_completed_count",
    "lookback_homework_late_count",
    "lookback_homework_missing_count",
    "lookback_negative_behavior_count",
    "lookback_positive_behavior_count",
    "lookback_negative_behavior_severity_sum",
    "transport_assigned_at_snapshot",
    "lookback_transport_boarded_count",
    "lookback_transport_absent_count",
]


def engineer_features(data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]) -> pd.DataFrame:
    """Takes raw input (DataFrame or dicts), engineers interaction features, and returns clean feature DataFrame."""
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, list):
        df = pd.DataFrame(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise InvalidFeatureError(f"Unsupported data type for feature engineering: {type(data)}")

    # Ensure required raw features exist
    for col in RAW_FEATURE_COLUMNS:
        if col not in df.columns:
            if col in ["lookback_assessment_average_percent", "lookback_assessment_minimum_percent"]:
                df[col] = np.nan
            elif col == "transport_assigned_at_snapshot":
                df[col] = False
            else:
                df[col] = 0

    # Convert booleans to int
    if "transport_assigned_at_snapshot" in df.columns:
        df["transport_assigned_at_snapshot"] = df["transport_assigned_at_snapshot"].astype(int)

    rec = np.maximum(df["lookback_attendance_records"].to_numpy(dtype=float), 1.0)
    pres = np.maximum(df["lookback_present_count"].to_numpy(dtype=float), 0.0)

    # 1. Attendance ratios & momentum
    df["feat_absent_ratio"] = df["lookback_absent_count"] / rec
    df["feat_late_ratio"] = df["lookback_late_count"] / rec
    df["feat_excused_ratio"] = df["lookback_excused_count"] / rec
    unexcused = np.maximum(0.0, df["lookback_absent_count"] - df["lookback_excused_count"])
    df["feat_unexcused_absent_ratio"] = unexcused / rec
    df["feat_attendance_odds"] = df["lookback_absent_count"] / (pres + 1.0)
    df["feat_attendance_disruption_index"] = (
        df["lookback_absent_count"] * 2.0 + df["lookback_late_count"] * 0.5
    ) / rec

    # 2. Homework discipline
    due = np.maximum(df["lookback_homework_due_count"].to_numpy(dtype=float), 1.0)
    completed = np.maximum(df["lookback_homework_completed_count"].to_numpy(dtype=float), 1.0)
    df["feat_homework_due_flag"] = (df["lookback_homework_due_count"] > 0).astype(int)
    df["feat_homework_missing_ratio"] = df["lookback_homework_missing_count"] / due
    df["feat_homework_completion_ratio"] = df["lookback_homework_completed_count"] / due
    df["feat_homework_late_ratio"] = df["lookback_homework_late_count"] / completed

    # 3. Behavior compound
    df["feat_total_behavior_events"] = (
        df["lookback_positive_behavior_count"] + df["lookback_negative_behavior_count"]
    )
    df["feat_behavior_net_score"] = (
        df["lookback_positive_behavior_count"] - df["lookback_negative_behavior_count"]
    )
    neg_count = np.maximum(df["lookback_negative_behavior_count"].to_numpy(dtype=float), 1.0)
    df["feat_neg_severity_per_event"] = df["lookback_negative_behavior_severity_sum"] / neg_count
    df["feat_behavior_risk_ratio"] = (
        df["lookback_negative_behavior_severity_sum"] + 1.0
    ) / (df["lookback_positive_behavior_count"] + 1.0)

    # 4. Academic health & volatility
    has_score = (df["lookback_assessment_count"] > 0).astype(int)
    df["feat_has_assessment"] = has_score
    avg_score = df["lookback_assessment_average_percent"].fillna(75.0)
    min_score = df["lookback_assessment_minimum_percent"].fillna(avg_score)
    df["feat_assessment_spread"] = np.where(has_score == 1, avg_score - min_score, 0.0)
    df["feat_assessment_failing"] = np.where((has_score == 1) & (avg_score < 60.0), 1, 0)
    df["feat_score_deficit"] = (100.0 - avg_score) / 100.0
    df["feat_academic_attendance_health"] = (
        df["lookback_attendance_percent"] / 100.0
    ) * (avg_score / 100.0)

    # 5. Domain interactions
    df["feat_interact_absent_score"] = df["feat_absent_ratio"] * df["feat_score_deficit"]
    df["feat_interact_hw_score"] = df["feat_homework_missing_ratio"] * df["feat_score_deficit"]
    df["feat_interact_absent_hw"] = df["feat_absent_ratio"] * df["feat_homework_missing_ratio"]

    # 6. Transport reliability
    uses_bus = df["transport_assigned_at_snapshot"].astype(int)
    bus_events = np.maximum(
        df["lookback_transport_boarded_count"] + df["lookback_transport_absent_count"], 1.0
    )
    df["feat_transport_absence_ratio"] = np.where(
        uses_bus == 1, df["lookback_transport_absent_count"] / bus_events, 0.0
    )
    df["feat_transport_boarded_ratio"] = np.where(
        uses_bus == 1, df["lookback_transport_boarded_count"] / bus_events, 0.0
    )

    # 7. Composite latent risk proxy
    df["feat_latent_risk_proxy"] = (
        df["feat_absent_ratio"] * 0.35
        + df["feat_score_deficit"] * 0.40
        + df["feat_homework_missing_ratio"] * 0.25
        + (df["lookback_negative_behavior_count"] * 0.08)
        - (df["lookback_positive_behavior_count"] * 0.05)
    )

    # 8. Context & Tenure
    df["feat_tenure_years"] = df["days_since_enrollment"] / 365.25
    df["feat_tenure_per_grade"] = df["days_since_enrollment"] / np.maximum(
        df["grade_level_id"], 1.0
    )

    # Exclude forbidden columns
    final_cols = [c for c in df.columns if c not in FORBIDDEN_COLUMNS]
    return df[final_cols]
