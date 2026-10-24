"""Command Line Interface (CLI) for Ishraf Student Risk AI System."""

import argparse
import json
import sys
from pathlib import Path
import pandas as pd

from src.models.predictor import StudentRiskPredictor
from src.core.config import REPORT_PATH

def cmd_predict_single(args):
    predictor = StudentRiskPredictor.get_instance()
    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"Error: input file '{args.input}' not found.", file=sys.stderr)
        sys.exit(1)
        
    data = json.loads(input_path.read_text(encoding="utf-8"))
    res = predictor.predict_single(data)
    print("=" * 60)
    print("ISHRAF STUDENT RISK PREDICTION RESULT")
    print("=" * 60)
    print(f"Student Key:        {res.student_key}")
    print(f"Grade Level:        {res.grade_level_id}")
    print(f"Risk Probability:   {res.risk_probability:.2%}")
    print(f"Risk Level:         {res.risk_level}")
    print(f"Is at Risk?         {'YES' if res.is_at_risk else 'NO'}")
    print(f"Decision Threshold: {res.decision_threshold:.4f}")
    print("\nPrimary Risk Factors:")
    for f in res.primary_risk_factors:
        print(f"  * {f}")
    print("\nRecommended Interventions:")
    for a in res.recommended_interventions:
        print(f"  -> {a}")
    print("=" * 60)

def cmd_predict_batch(args):
    predictor = StudentRiskPredictor.get_instance()
    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"Error: input file '{args.input}' not found.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(input_path)
    print(f"Loaded {len(df):,} student records from '{args.input}'...")
    evaluated = predictor.predict_batch(df)

    out_path = Path(args.output)
    evaluated.to_csv(out_path, index=False)
    print(f"Evaluated predictions saved successfully to '{out_path}'.")
    
    at_risk_cnt = evaluated['is_at_risk'].sum()
    print(f"Summary: {at_risk_cnt:,} / {len(df):,} ({at_risk_cnt / len(df):.1%}) identified as At-Risk.")

def cmd_health(args):
    predictor = StudentRiskPredictor.get_instance()
    print("=" * 50)
    print("ISHRAF SYSTEM HEALTH CHECK")
    print("=" * 50)
    print(f"Status:               HEALTHY")
    print(f"Model Artifact:       {predictor.model_path}")
    print(f"Features Count:       {len(predictor.feature_columns)}")
    print(f"Calibrated Threshold: {predictor.threshold:.4f}")
    print("=" * 50)

def cmd_metrics(args):
    if not REPORT_PATH.is_file():
        print(f"Error: Report file '{REPORT_PATH}' not found.", file=sys.stderr)
        sys.exit(1)
    rep = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    print("=" * 60)
    print("ISHRAF MODEL BENCHMARK & TEST METRICS")
    print("=" * 60)
    print(f"Best Model Architecture: {rep.get('best_model')}")
    print(f"Optimal Threshold:       {rep.get('best_threshold'):.4f}")
    print("\nTest Evaluation Summary:")
    for m in rep.get("test_results", []):
        print(f"  * {m['model']:22s} | ROC-AUC: {m['roc_auc']:.4f} | PR-AUC: {m['pr_auc']:.4f} | F1: {m['f1']:.4f} | Recall: {m['recall']:.2%}")
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="Ishraf Student Attendance Risk CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # predict
    p_single = subparsers.add_parser("predict", help="Predict single student from JSON file")
    p_single.add_argument("--input", "-i", required=True, help="Path to input student JSON file")
    p_single.set_defaults(func=cmd_predict_single)

    # predict-batch
    p_batch = subparsers.add_parser("predict-batch", help="Batch predict student records from CSV")
    p_batch.add_argument("--input", "-i", required=True, help="Input CSV file")
    p_batch.add_argument("--output", "-o", default="evaluated_students.csv", help="Output evaluated CSV path")
    p_batch.set_defaults(func=cmd_predict_batch)

    # health
    p_health = subparsers.add_parser("health", help="Check system and model health")
    p_health.set_defaults(func=cmd_health)

    # metrics
    p_metrics = subparsers.add_parser("metrics", help="Display verified model benchmark metrics")
    p_metrics.set_defaults(func=cmd_metrics)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
