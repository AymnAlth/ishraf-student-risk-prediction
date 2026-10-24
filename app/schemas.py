"""Pydantic schemas for Ishraf ML Early Warning API."""

from typing import List, Optional
from pydantic import BaseModel, Field


class StudentRawInput(BaseModel):
    """Raw student snapshot input for inference."""
    student_key: Optional[str] = Field(default="STUDENT_NEW_001", description="Pseudonymous student identifier")
    grade_level_id: int = Field(default=3, ge=1, le=12, description="Student grade level (1 to 12)")
    days_since_enrollment: int = Field(default=240, ge=0, description="Total days since student enrollment")
    lookback_attendance_records: int = Field(default=20, ge=1, description="Total attendance check days in 28d window")
    lookback_present_count: int = Field(default=16, ge=0, description="Number of days recorded present")
    lookback_absent_count: int = Field(default=3, ge=0, description="Number of days recorded absent")
    lookback_late_count: int = Field(default=1, ge=0, description="Number of days recorded late")
    lookback_excused_count: int = Field(default=0, ge=0, description="Number of excused absences")
    lookback_attendance_percent: float = Field(default=80.0, ge=0.0, le=100.0, description="Present attendance percentage")
    lookback_assessment_count: int = Field(default=2, ge=0, description="Number of graded assessments")
    lookback_assessment_average_percent: Optional[float] = Field(default=68.5, ge=0.0, le=100.0, description="Average assessment score")
    lookback_assessment_minimum_percent: Optional[float] = Field(default=55.0, ge=0.0, le=100.0, description="Lowest assessment score")
    lookback_homework_due_count: int = Field(default=4, ge=0, description="Number of homeworks assigned")
    lookback_homework_completed_count: int = Field(default=2, ge=0, description="Number of completed homeworks")
    lookback_homework_late_count: int = Field(default=0, ge=0, description="Number of late homework submissions")
    lookback_homework_missing_count: int = Field(default=2, ge=0, description="Number of missing homeworks")
    lookback_negative_behavior_count: int = Field(default=1, ge=0, description="Count of negative behavior incidents")
    lookback_positive_behavior_count: int = Field(default=1, ge=0, description="Count of positive behavior records")
    lookback_negative_behavior_severity_sum: int = Field(default=2, ge=0, description="Cumulative severity of negative behavior")
    transport_assigned_at_snapshot: bool = Field(default=True, description="Whether student is assigned to school bus")
    lookback_transport_boarded_count: int = Field(default=12, ge=0, description="Count of successful bus boardings")
    lookback_transport_absent_count: int = Field(default=3, ge=0, description="Count of bus absences")


class SinglePredictionResponse(BaseModel):
    """Prediction output for a single student."""
    student_key: Optional[str]
    grade_level_id: Optional[int]
    is_at_risk: bool
    risk_probability: float
    risk_level: str
    decision_threshold: float
    primary_risk_factors: List[str]
    recommended_interventions: List[str]


class BatchPredictionRequest(BaseModel):
    """Batch prediction request payload."""
    students: List[StudentRawInput]


class BatchPredictionSummary(BaseModel):
    total_students: int
    at_risk_count: int
    at_risk_percentage: float
    critical_risk_count: int
    high_risk_count: int
    moderate_risk_count: int
    low_risk_count: int


class BatchPredictionResponse(BaseModel):
    summary: BatchPredictionSummary
    predictions: List[SinglePredictionResponse]


class HealthResponse(BaseModel):
    status: str
    model_version: str
    calibrated_threshold: float
    features_count: int
    timestamp: str
