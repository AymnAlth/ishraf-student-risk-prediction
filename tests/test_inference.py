"""Unit tests for production predictor and inference engine."""

import json
from pathlib import Path
import pandas as pd
import pytest

from src.models.predictor import StudentRiskPredictor
from src.core.config import MODEL_PATH

def test_predictor_singleton_initialization():
    predictor1 = StudentRiskPredictor.get_instance()
    predictor2 = StudentRiskPredictor.get_instance()
    assert predictor1 is predictor2
    assert predictor1.threshold > 0.0
    assert len(predictor1.feature_columns) > 20

def test_single_prediction_stable_student():
    predictor = StudentRiskPredictor.get_instance()
    stable_student = {
        "student_key": "TEST_STUDENT_STABLE",
        "grade_level_id": 4,
        "days_since_enrollment": 300,
        "lookback_attendance_records": 22,
        "lookback_present_count": 22,
        "lookback_absent_count": 0,
        "lookback_late_count": 0,
        "lookback_excused_count": 0,
        "lookback_attendance_percent": 100.0,
        "lookback_assessment_count": 4,
        "lookback_assessment_average_percent": 92.0,
        "lookback_assessment_minimum_percent": 88.0,
        "lookback_homework_due_count": 6,
        "lookback_homework_completed_count": 6,
        "lookback_homework_late_count": 0,
        "lookback_homework_missing_count": 0,
        "lookback_negative_behavior_count": 0,
        "lookback_positive_behavior_count": 3,
        "lookback_negative_behavior_severity_sum": 0,
        "transport_assigned_at_snapshot": False,
        "lookback_transport_boarded_count": 0,
        "lookback_transport_absent_count": 0,
    }

    result = predictor.predict_single(stable_student)
    assert result.is_at_risk is False
    assert result.risk_level == "LOW"
    assert result.risk_probability < predictor.threshold
    assert len(result.primary_risk_factors) > 0
    assert len(result.recommended_interventions) > 0

def test_single_prediction_at_risk_student():
    predictor = StudentRiskPredictor.get_instance()
    at_risk_student = {
        "student_key": "TEST_STUDENT_RISK",
        "grade_level_id": 5,
        "days_since_enrollment": 180,
        "lookback_attendance_records": 18,
        "lookback_present_count": 10,
        "lookback_absent_count": 7,
        "lookback_late_count": 1,
        "lookback_excused_count": 0,
        "lookback_attendance_percent": 55.56,
        "lookback_assessment_count": 2,
        "lookback_assessment_average_percent": 45.0,
        "lookback_assessment_minimum_percent": 38.0,
        "lookback_homework_due_count": 6,
        "lookback_homework_completed_count": 1,
        "lookback_homework_late_count": 0,
        "lookback_homework_missing_count": 5,
        "lookback_negative_behavior_count": 3,
        "lookback_positive_behavior_count": 0,
        "lookback_negative_behavior_severity_sum": 8,
        "transport_assigned_at_snapshot": True,
        "lookback_transport_boarded_count": 4,
        "lookback_transport_absent_count": 7,
    }

    result = predictor.predict_single(at_risk_student)
    assert result.is_at_risk is True
    assert result.risk_level in ["HIGH", "CRITICAL"]
    assert result.risk_probability >= predictor.threshold
    assert any("حضور" in f or "غياب" in f for f in result.primary_risk_factors)
