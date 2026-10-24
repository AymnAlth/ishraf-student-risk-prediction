"""FastAPI Application for Ishraf Student Risk Early Warning System."""

from datetime import datetime, timezone
import io
import json
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd

from app.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    BatchPredictionSummary,
    HealthResponse,
    SinglePredictionResponse,
    StudentRawInput,
)
from src.core.config import APP_TITLE, APP_VERSION, REPORT_PATH
from src.models.predictor import StudentRiskPredictor

app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description="Production-grade AI Early Warning System for Student Attendance Risk & Drop-out Prevention.",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serves the interactive web interface."""
    index_file = STATIC_DIR / "index.html"
    if index_file.is_file():
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h2>Ishraf Student Risk API is active. Visit /docs for Swagger UI.</h2>")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """System health check and loaded model diagnostics."""
    try:
        predictor = StudentRiskPredictor.get_instance()
        return HealthResponse(
            status="healthy",
            model_version=APP_VERSION,
            calibrated_threshold=predictor.threshold,
            features_count=len(predictor.feature_columns),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(exc)}")


@app.post("/predict", response_model=SinglePredictionResponse)
async def predict_single_student(student_input: StudentRawInput):
    """Predict attendance risk for an individual student record with tailored explanations."""
    try:
        predictor = StudentRiskPredictor.get_instance()
        result = predictor.predict_single(student_input.model_dump())
        return SinglePredictionResponse(**result.to_dict())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Inference error: {str(exc)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch_students(batch_request: BatchPredictionRequest):
    """Run batch risk prediction for multiple students and return aggregated analytics."""
    try:
        predictor = StudentRiskPredictor.get_instance()
        predictions = []
        at_risk_count = 0
        critical_count = 0
        high_count = 0
        moderate_count = 0
        low_count = 0

        for student in batch_request.students:
            res = predictor.predict_single(student.model_dump())
            predictions.append(SinglePredictionResponse(**res.to_dict()))
            if res.is_at_risk:
                at_risk_count += 1
            if res.risk_level == "CRITICAL":
                critical_count += 1
            elif res.risk_level == "HIGH":
                high_count += 1
            elif res.risk_level == "MODERATE":
                moderate_count += 1
            else:
                low_count += 1

        total = len(predictions)
        summary = BatchPredictionSummary(
            total_students=total,
            at_risk_count=at_risk_count,
            at_risk_percentage=round((at_risk_count / total) * 100, 2) if total > 0 else 0.0,
            critical_risk_count=critical_count,
            high_risk_count=high_count,
            moderate_risk_count=moderate_count,
            low_risk_count=low_count,
        )

        return BatchPredictionResponse(summary=summary, predictions=predictions)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Batch inference error: {str(exc)}")


@app.post("/predict/csv")
async def predict_csv_file(file: UploadFile = File(...)):
    """Upload a CSV of student records and download the evaluated CSV with risk probabilities."""
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))

        predictor = StudentRiskPredictor.get_instance()
        evaluated_df = predictor.predict_batch(df)

        output_stream = io.StringIO()
        evaluated_df.to_csv(output_stream, index=False)
        output_stream.seek(0)

        return StreamingResponse(
            io.BytesIO(output_stream.getvalue().encode("utf-8")),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=evaluated_{file.filename}"},
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to process CSV: {str(exc)}")


@app.get("/metrics")
async def get_model_metrics() -> Dict[str, Any]:
    """Retrieve verified evaluation metrics, fairness audit, and benchmark report."""
    if not REPORT_PATH.is_file():
        raise HTTPException(status_code=404, detail="Benchmark report not found.")
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))
