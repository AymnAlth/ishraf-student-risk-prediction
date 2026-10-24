"""Integration tests for FastAPI endpoints."""

from fastapi.testclient import TestClient
import pytest

from app.api import app

client = TestClient(app)

def test_api_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_version" in data
    assert "calibrated_threshold" in data

def test_api_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "test_results" in data
    assert "best_model" in data

def test_api_predict_single_student():
    payload = {
        "student_key": "API_TEST_001",
        "grade_level_id": 3,
        "days_since_enrollment": 150,
        "lookback_attendance_records": 20,
        "lookback_present_count": 19,
        "lookback_absent_count": 1,
        "lookback_late_count": 0,
        "lookback_excused_count": 0,
        "lookback_attendance_percent": 95.0,
        "lookback_assessment_count": 3,
        "lookback_assessment_average_percent": 85.0,
        "lookback_assessment_minimum_percent": 80.0,
        "lookback_homework_due_count": 5,
        "lookback_homework_completed_count": 5,
        "lookback_homework_late_count": 0,
        "lookback_homework_missing_count": 0,
        "lookback_negative_behavior_count": 0,
        "lookback_positive_behavior_count": 2,
        "lookback_negative_behavior_severity_sum": 0,
        "transport_assigned_at_snapshot": False,
        "lookback_transport_boarded_count": 0,
        "lookback_transport_absent_count": 0,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["student_key"] == "API_TEST_001"
    assert "is_at_risk" in data
    assert "risk_probability" in data
    assert "risk_level" in data
    assert isinstance(data["primary_risk_factors"], list)
    assert isinstance(data["recommended_interventions"], list)

def test_api_predict_batch():
    payload = {
        "students": [
            {
                "student_key": "STUDENT_A",
                "grade_level_id": 2,
                "lookback_attendance_records": 20,
                "lookback_present_count": 19,
                "lookback_absent_count": 1,
                "lookback_attendance_percent": 95.0,
                "lookback_assessment_count": 2,
                "lookback_assessment_average_percent": 88.0,
                "lookback_homework_due_count": 4,
                "lookback_homework_completed_count": 4,
                "lookback_homework_missing_count": 0,
                "transport_assigned_at_snapshot": False
            },
            {
                "student_key": "STUDENT_B",
                "grade_level_id": 5,
                "lookback_attendance_records": 18,
                "lookback_present_count": 10,
                "lookback_absent_count": 8,
                "lookback_attendance_percent": 55.0,
                "lookback_assessment_count": 2,
                "lookback_assessment_average_percent": 45.0,
                "lookback_homework_due_count": 5,
                "lookback_homework_completed_count": 1,
                "lookback_homework_missing_count": 4,
                "transport_assigned_at_snapshot": True,
                "lookback_transport_boarded_count": 4,
                "lookback_transport_absent_count": 6
            }
        ]
    }

    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "summary" in data
    assert data["summary"]["total_students"] == 2
    assert len(data["predictions"]) == 2
