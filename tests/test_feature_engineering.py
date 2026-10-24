"""Unit tests for feature engineering and zero-leakage enforcement."""

import numpy as np
import pandas as pd
import pytest
from src.data.feature_engineering import (
    engineer_features,
    FORBIDDEN_COLUMNS,
    RAW_FEATURE_COLUMNS,
)

def test_feature_engineering_output_shape():
    sample_input = {col: 1 for col in RAW_FEATURE_COLUMNS}
    sample_input["lookback_assessment_average_percent"] = 75.0
    sample_input["lookback_assessment_minimum_percent"] = 60.0
    sample_input["transport_assigned_at_snapshot"] = True

    df = engineer_features(sample_input)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    # Check that forbidden columns are strictly absent
    for col in FORBIDDEN_COLUMNS:
        assert col not in df.columns, f"Forbidden column {col} leaked into feature matrix!"

def test_zero_leakage_when_future_labels_present_in_input():
    sample_input = {
        "student_key": "STUDENT_LEAK_TEST",
        "snapshot_date": "2025-10-15",
        "label_at_risk_next_14d": True,
        "label_observation_count": 10,
        "label_future_attendance_percent": 60.0,
        "grade_level_id": 3,
        "days_since_enrollment": 120,
        "lookback_attendance_records": 15,
        "lookback_present_count": 12,
        "lookback_absent_count": 3,
        "lookback_late_count": 0,
        "lookback_excused_count": 0,
        "lookback_attendance_percent": 80.0,
        "lookback_assessment_count": 2,
        "lookback_assessment_average_percent": 70.0,
        "lookback_assessment_minimum_percent": 65.0,
        "lookback_homework_due_count": 4,
        "lookback_homework_completed_count": 4,
        "lookback_homework_late_count": 0,
        "lookback_homework_missing_count": 0,
        "lookback_negative_behavior_count": 0,
        "lookback_positive_behavior_count": 1,
        "lookback_negative_behavior_severity_sum": 0,
        "transport_assigned_at_snapshot": False,
        "lookback_transport_boarded_count": 0,
        "lookback_transport_absent_count": 0,
    }

    df = engineer_features(sample_input)
    assert "label_at_risk_next_14d" not in df.columns
    assert "label_observation_count" not in df.columns
    assert "label_future_attendance_percent" not in df.columns
    assert "student_key" not in df.columns
    assert "snapshot_date" not in df.columns

def test_edge_case_zero_assessments_and_zero_homework():
    sample_input = {
        "grade_level_id": 1,
        "days_since_enrollment": 50,
        "lookback_attendance_records": 10,
        "lookback_present_count": 10,
        "lookback_absent_count": 0,
        "lookback_late_count": 0,
        "lookback_excused_count": 0,
        "lookback_attendance_percent": 100.0,
        "lookback_assessment_count": 0,
        "lookback_assessment_average_percent": np.nan,
        "lookback_assessment_minimum_percent": np.nan,
        "lookback_homework_due_count": 0,
        "lookback_homework_completed_count": 0,
        "lookback_homework_late_count": 0,
        "lookback_homework_missing_count": 0,
        "lookback_negative_behavior_count": 0,
        "lookback_positive_behavior_count": 0,
        "lookback_negative_behavior_severity_sum": 0,
        "transport_assigned_at_snapshot": False,
        "lookback_transport_boarded_count": 0,
        "lookback_transport_absent_count": 0,
    }

    df = engineer_features(sample_input)
    assert len(df.columns) == 48
    assert df["feat_homework_missing_ratio"].iloc[0] == 0.0
    assert df["feat_has_assessment"].iloc[0] == 0
    assert df["feat_attendance_disruption_index"].iloc[0] == 0.0
