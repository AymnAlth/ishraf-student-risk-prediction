"""Production Prediction Engine for Ishraf Student Risk Model.

Loads trained model artifact, executes feature transformations, generates probabilities,
calibrates decision thresholds, and provides local explainability and intervention guidance.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import joblib
import numpy as np
import pandas as pd

from src.core.config import (
    MODEL_PATH,
    DEFAULT_RISK_THRESHOLD,
    CRITICAL_RISK_THRESHOLD,
    HIGH_RISK_THRESHOLD,
)
from src.core.exceptions import ModelNotFoundError, InferenceError
from src.data.feature_engineering import engineer_features


@dataclass
class SinglePredictionResult:
    is_at_risk: bool
    risk_probability: float
    risk_level: str
    decision_threshold: float
    primary_risk_factors: List[str] = field(default_factory=list)
    recommended_interventions: List[str] = field(default_factory=list)
    student_key: Optional[str] = None
    grade_level_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "student_key": self.student_key,
            "grade_level_id": self.grade_level_id,
            "is_at_risk": self.is_at_risk,
            "risk_probability": round(self.risk_probability, 4),
            "risk_level": self.risk_level,
            "decision_threshold": round(self.decision_threshold, 4),
            "primary_risk_factors": self.primary_risk_factors,
            "recommended_interventions": self.recommended_interventions,
        }


class StudentRiskPredictor:
    """Production predictor singleton for Ishraf early warning system."""

    _instance: Optional[StudentRiskPredictor] = None

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or MODEL_PATH
        self.artifact = self._load_artifact()
        self.threshold = float(self.artifact.get("threshold", DEFAULT_RISK_THRESHOLD))
        self.feature_columns = self.artifact.get("feature_columns", [])
        self.pipeline_info = self.artifact.get("pipeline", {})

        # Extract model and imputer
        if isinstance(self.pipeline_info, dict) and self.pipeline_info.get("type") == "single":
            self.model = self.pipeline_info.get("model")
            self.imputer = self.pipeline_info.get("imputer")
        elif "model" in self.artifact:
            self.model = self.artifact["model"]
            self.imputer = None
        else:
            raise InferenceError("Invalid model artifact structure.")

    @classmethod
    def get_instance(cls, model_path: Optional[str] = None) -> StudentRiskPredictor:
        if cls._instance is None:
            cls._instance = cls(model_path)
        return cls._instance

    def _load_artifact(self) -> Dict[str, Any]:
        if not self.model_path.is_file():
            raise ModelNotFoundError(f"Trained model not found at: {self.model_path}")
        return joblib.load(self.model_path)

    def _determine_risk_level(self, probability: float) -> str:
        if probability >= CRITICAL_RISK_THRESHOLD:
            return "CRITICAL"
        elif probability >= HIGH_RISK_THRESHOLD:
            return "HIGH"
        elif probability >= self.threshold:
            return "MODERATE"
        else:
            return "LOW"

    def _generate_explanations(self, row: pd.Series) -> tuple[List[str], List[str]]:
        factors = []
        actions = []

        # Attendance check
        att_pct = row.get("lookback_attendance_percent", 100.0)
        absent_cnt = row.get("lookback_absent_count", 0)
        if att_pct < 85.0 or absent_cnt >= 3:
            factors.append(f"نسبة الحضور السابقة متدنية ({att_pct:.1f}%) مع تسجيل {int(absent_cnt)} غيابات")
            actions.append("التواصل العاجل مع ولي الأمر واستيضاح مبررات الغياب المتكرر")

        # Academic check
        avg_score = row.get("lookback_assessment_average_percent")
        if pd.notna(avg_score) and avg_score < 65.0:
            factors.append(f"تدني ملحوظ في متوسط الدرجات والتقييمات ({avg_score:.1f}%)")
            actions.append("إحالة الطالب لبرنامج التقوية الأكاديمية والمتابعة الصفية المباشرة")

        # Homework check
        hw_missing = row.get("lookback_homework_missing_count", 0)
        if hw_missing >= 2:
            factors.append(f"تراكم {int(hw_missing)} واجبات مدرسية مفقودة لم تُسلّم")
            actions.append("جلسة إرشاد مع المعلم المعني لتنظيم جدول الواجبات والمتابعة اليومية")

        # Behavior check
        neg_beh = row.get("lookback_negative_behavior_count", 0)
        neg_sev = row.get("lookback_negative_behavior_severity_sum", 0)
        if neg_beh >= 1 or neg_sev >= 3:
            factors.append(f"تسجيل مخالفات سلوكية بحدة تراكمية ({int(neg_sev)}) نقاط")
            actions.append("جلسة دعم وتوجيه نفسي فردي مع الأخصائي الاجتماعي بالمدرسة")

        # Transport check
        bus_absent = row.get("lookback_transport_absent_count", 0)
        uses_bus = row.get("transport_assigned_at_snapshot", False)
        if uses_bus and bus_absent >= 2:
            factors.append(f"تكرار التخلف عن ركوب الحافلة المدرسية ({int(bus_absent)} مرات)")
            actions.append("التنسيق مع مشرف النقل المدرسي لمراجعة توقيت وموقع محطة الحافلة")

        if not factors:
            factors.append("المؤشرات العامة مستقرة ضمن الحدود الطبيعية")
            actions.append("مواصلة التحفيز الإيجابي ومراقبة المؤشرات الدورية")

        return factors, actions

    def predict_single(self, raw_input: Dict[str, Any]) -> SinglePredictionResult:
        try:
            student_key = raw_input.get("student_key", "UNKNOWN_STUDENT")
            grade_level = raw_input.get("grade_level_id", 1)

            # Feature Engineering
            df_feat = engineer_features(raw_input)

            # Match exact feature columns
            missing_cols = [c for c in self.feature_columns if c not in df_feat.columns]
            for col in missing_cols:
                df_feat[col] = 0.0

            X_input = df_feat[self.feature_columns].copy()

            # Apply imputer if required
            if self.imputer is not None:
                X_mat = self.imputer.transform(X_input)
            else:
                X_mat = X_input.to_numpy()

            # Predict probability
            prob = float(self.model.predict_proba(X_mat)[:, 1][0])
            is_at_risk = bool(prob >= self.threshold)
            risk_level = self._determine_risk_level(prob)

            factors, actions = self._generate_explanations(df_feat.iloc[0])

            return SinglePredictionResult(
                student_key=student_key,
                grade_level_id=grade_level,
                is_at_risk=is_at_risk,
                risk_probability=prob,
                risk_level=risk_level,
                decision_threshold=self.threshold,
                primary_risk_factors=factors,
                recommended_interventions=actions,
            )
        except Exception as exc:
            raise InferenceError(f"Prediction failed: {str(exc)}") from exc

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            result_df = df.copy()
            df_feat = engineer_features(df)

            missing_cols = [c for c in self.feature_columns if c not in df_feat.columns]
            for col in missing_cols:
                df_feat[col] = 0.0

            X_input = df_feat[self.feature_columns].copy()

            if self.imputer is not None:
                X_mat = self.imputer.transform(X_input)
            else:
                X_mat = X_input.to_numpy()

            probabilities = self.model.predict_proba(X_mat)[:, 1]
            predictions = (probabilities >= self.threshold).astype(int)

            result_df["predicted_probability"] = np.round(probabilities, 4)
            result_df["is_at_risk"] = predictions.astype(bool)
            result_df["risk_level"] = [self._determine_risk_level(p) for p in probabilities]

            return result_df
        except Exception as exc:
            raise InferenceError(f"Batch prediction failed: {str(exc)}") from exc
