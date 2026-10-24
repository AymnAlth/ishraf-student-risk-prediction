from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

RANDOM_STATE = 20_261_023


def find_project_root(start: Path) -> Path:
    candidates = [start, *start.parents, Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
    for candidate in candidates:
        if (candidate / 'package.json').is_file() and (candidate / 'ml').is_dir():
            return candidate
    raise RuntimeError('Run this script from within the Ishraf project directory.')


PROJECT_ROOT = find_project_root(Path.cwd().resolve())
DATA_PATH = PROJECT_ROOT / 'ml' / 'artifacts' / 'student-risk-synthetic' / 'student-risk-synthetic-training.csv'
DATASET_METADATA_PATH = DATA_PATH.with_name('student-risk-synthetic-training.metadata.json')
RUNS_ROOT = PROJECT_ROOT / 'ml' / 'artifacts' / 'training-runs'

assert DATA_PATH.is_file(), f'Missing synthetic dataset: {DATA_PATH}'
assert DATASET_METADATA_PATH.is_file(), f'Missing dataset metadata: {DATASET_METADATA_PATH}'


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    X = data.copy()
    rec = np.maximum(X['lookback_attendance_records'].to_numpy(dtype=float), 1.0)
    pres = np.maximum(X['lookback_present_count'].to_numpy(dtype=float), 0.0)

    # 1. Attendance ratios & momentum
    X['feat_absent_ratio'] = X['lookback_absent_count'] / rec
    X['feat_late_ratio'] = X['lookback_late_count'] / rec
    X['feat_excused_ratio'] = X['lookback_excused_count'] / rec
    unexcused = np.maximum(0.0, X['lookback_absent_count'] - X['lookback_excused_count'])
    X['feat_unexcused_absent_ratio'] = unexcused / rec
    X['feat_attendance_odds'] = X['lookback_absent_count'] / (pres + 1.0)
    X['feat_attendance_disruption_index'] = (X['lookback_absent_count'] * 2.0 + X['lookback_late_count'] * 0.5) / rec

    # 2. Homework discipline
    due = np.maximum(X['lookback_homework_due_count'].to_numpy(dtype=float), 1.0)
    completed = np.maximum(X['lookback_homework_completed_count'].to_numpy(dtype=float), 1.0)
    X['feat_homework_due_flag'] = (X['lookback_homework_due_count'] > 0).astype(int)
    X['feat_homework_missing_ratio'] = X['lookback_homework_missing_count'] / due
    X['feat_homework_completion_ratio'] = X['lookback_homework_completed_count'] / due
    X['feat_homework_late_ratio'] = X['lookback_homework_late_count'] / completed

    # 3. Behavior compound
    X['feat_total_behavior_events'] = X['lookback_positive_behavior_count'] + X['lookback_negative_behavior_count']
    X['feat_behavior_net_score'] = X['lookback_positive_behavior_count'] - X['lookback_negative_behavior_count']
    neg_count = np.maximum(X['lookback_negative_behavior_count'].to_numpy(dtype=float), 1.0)
    X['feat_neg_severity_per_event'] = X['lookback_negative_behavior_severity_sum'] / neg_count
    X['feat_behavior_risk_ratio'] = (X['lookback_negative_behavior_severity_sum'] + 1.0) / (X['lookback_positive_behavior_count'] + 1.0)

    # 4. Academic health & volatility
    has_score = (X['lookback_assessment_count'] > 0).astype(int)
    X['feat_has_assessment'] = has_score
    avg_score = X['lookback_assessment_average_percent'].fillna(75.0)
    min_score = X['lookback_assessment_minimum_percent'].fillna(avg_score)
    X['feat_assessment_spread'] = np.where(has_score == 1, avg_score - min_score, 0.0)
    X['feat_assessment_failing'] = np.where((has_score == 1) & (avg_score < 60.0), 1, 0)
    X['feat_score_deficit'] = (100.0 - avg_score) / 100.0
    X['feat_academic_attendance_health'] = (X['lookback_attendance_percent'] / 100.0) * (avg_score / 100.0)

    # 5. Domain interactions
    X['feat_interact_absent_score'] = X['feat_absent_ratio'] * X['feat_score_deficit']
    X['feat_interact_hw_score'] = X['feat_homework_missing_ratio'] * X['feat_score_deficit']
    X['feat_interact_absent_hw'] = X['feat_absent_ratio'] * X['feat_homework_missing_ratio']

    # 6. Transport reliability
    uses_bus = X['transport_assigned_at_snapshot'].astype(int)
    bus_events = np.maximum(X['lookback_transport_boarded_count'] + X['lookback_transport_absent_count'], 1.0)
    X['feat_transport_absence_ratio'] = np.where(uses_bus == 1, X['lookback_transport_absent_count'] / bus_events, 0.0)
    X['feat_transport_boarded_ratio'] = np.where(uses_bus == 1, X['lookback_transport_boarded_count'] / bus_events, 0.0)

    # 7. Composite latent risk proxy
    X['feat_latent_risk_proxy'] = (
        X['feat_absent_ratio'] * 0.35 +
        X['feat_score_deficit'] * 0.40 +
        X['feat_homework_missing_ratio'] * 0.25 +
        (X['lookback_negative_behavior_count'] * 0.08) -
        (X['lookback_positive_behavior_count'] * 0.05)
    )

    # 8. Context & Tenure
    X['feat_tenure_years'] = X['days_since_enrollment'] / 365.25
    X['feat_tenure_per_grade'] = X['days_since_enrollment'] / np.maximum(X['grade_level_id'], 1.0)

    return X


def choose_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if len(thresholds) == 0:
        return 0.5
    f1_scores = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    return float(thresholds[int(np.nanargmax(f1_scores))])


def evaluate_classifier(
    model_name: str, y_true: pd.Series, probabilities: np.ndarray, threshold: float
) -> dict[str, float | int | str]:
    predictions = (probabilities >= threshold).astype('int8')
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()
    return {
        'model': model_name,
        'threshold': float(threshold),
        'roc_auc': float(roc_auc_score(y_true, probabilities)),
        'pr_auc': float(average_precision_score(y_true, probabilities)),
        'precision': float(precision_score(y_true, predictions, zero_division=0)),
        'recall': float(recall_score(y_true, predictions, zero_division=0)),
        'f1': float(f1_score(y_true, predictions, zero_division=0)),
        'brier_score': float(brier_score_loss(y_true, probabilities)),
        'true_negative': int(true_negative),
        'false_positive': int(false_positive),
        'false_negative': int(false_negative),
        'true_positive': int(true_positive),
    }


def main():
    print('=' * 70)
    print('ISHRAF STUDENT ATTENDANCE RISK EARLY WARNING SYSTEM - ADVANCED PIPELINE')
    print('=' * 70)

    dataset_metadata = json.loads(DATASET_METADATA_PATH.read_text(encoding='utf-8'))
    print(f"Loaded metadata for: {dataset_metadata['dataset']}")

    print('Reading synthetic dataset (300,000 rows)...')
    t0 = time.time()
    raw_data = pd.read_csv(DATA_PATH)
    print(f'Read complete in {time.time() - t0:.2f}s. Shape: {raw_data.shape}')

    print('Applying Domain-Driven Feature Engineering...')
    data = engineer_features(raw_data)

    IDENTIFIER_COLUMNS = ['student_key', 'snapshot_date']
    LABEL_COLUMN = 'label_at_risk_next_14d'
    LABEL_EVIDENCE_COLUMNS = [
        'label_observation_count',
        'label_future_attendance_percent',
    ]
    FORBIDDEN_INPUT_COLUMNS = IDENTIFIER_COLUMNS + [LABEL_COLUMN, *LABEL_EVIDENCE_COLUMNS]
    FEATURE_COLUMNS = [col for col in data.columns if col not in FORBIDDEN_INPUT_COLUMNS]

    print(f'Total engineered features: {len(FEATURE_COLUMNS)}')

    X = data.loc[:, FEATURE_COLUMNS].copy()
    for col in X.select_dtypes(include='bool'):
        X[col] = X[col].astype('int8')
    X = X.apply(pd.to_numeric, errors='raise')
    y = data[LABEL_COLUMN].astype('int8')
    groups = data['student_key']

    # Chronological Split
    snapshot_dates = pd.to_datetime(data['snapshot_date'], errors='raise')
    academic_period = pd.Series(
        np.where(snapshot_dates.dt.month >= 8, snapshot_dates.dt.year, snapshot_dates.dt.year - 1),
        index=data.index,
        name='academic_period',
    ).astype('int64')

    ordered_periods = sorted(pd.unique(academic_period).tolist())
    train_periods = ordered_periods[:-2]
    validation_period = ordered_periods[-2]
    test_period = ordered_periods[-1]

    student_last_period = pd.DataFrame({
        'student_key': groups,
        'academic_period': academic_period,
    }).groupby('student_key', sort=False)['academic_period'].max()
    assigned_period = groups.map(student_last_period).astype('int64')

    train_mask = academic_period.isin(train_periods) & assigned_period.isin(train_periods)
    validation_mask = academic_period.eq(validation_period) & assigned_period.eq(validation_period)
    test_mask = academic_period.eq(test_period) & assigned_period.eq(test_period)

    train_indices = np.flatnonzero(train_mask.to_numpy())
    validation_indices = np.flatnonzero(validation_mask.to_numpy())
    test_indices = np.flatnonzero(test_mask.to_numpy())

    X_train, y_train = X.iloc[train_indices], y.iloc[train_indices]
    X_val, y_val = X.iloc[validation_indices], y.iloc[validation_indices]
    X_test, y_test = X.iloc[test_indices], y.iloc[test_indices]

    print(f'Train set: {len(train_indices)} rows | Val set: {len(validation_indices)} rows | Test set: {len(test_indices)} rows')

    # Impute missing values for tree & linear models
    imputer = SimpleImputer(strategy='median')
    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp = imputer.transform(X_val)
    X_test_imp = imputer.transform(X_test)

    # Models definition
    models = {
        'logistic_regression': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(max_iter=1_000, random_state=RANDOM_STATE)),
        ]),
        'hist_gradient_boosting': HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=250,
            max_leaf_nodes=31,
            min_samples_leaf=40,
            l2_regularization=0.2,
            random_state=RANDOM_STATE,
        ),
        'lightgbm': lgb.LGBMClassifier(
            n_estimators=350,
            learning_rate=0.04,
            num_leaves=35,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        ),
        'catboost': CatBoostClassifier(
            iterations=450,
            learning_rate=0.05,
            depth=6,
            random_seed=RANDOM_STATE,
            verbose=0,
            thread_count=-1,
        ),
        'xgboost': xgb.XGBClassifier(
            n_estimators=350,
            learning_rate=0.04,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            eval_metric='logloss',
        ),
    }

    validation_reports = []
    test_reports = []
    val_probabilities = {}
    test_probabilities = {}

    print('\nStarting Model Training and Validation Benchmarking...')
    for model_name, model in models.items():
        t_start = time.time()
        print(f'Training [{model_name}]...')
        model.fit(X_train_imp, y_train)
        elapsed = time.time() - t_start

        val_probs = model.predict_proba(X_val_imp)[:, 1]
        test_probs = model.predict_proba(X_test_imp)[:, 1]

        th = choose_threshold(y_val, val_probs)
        val_probabilities[model_name] = val_probs
        test_probabilities[model_name] = test_probs

        v_rep = evaluate_classifier(model_name, y_val, val_probs, th)
        t_rep = evaluate_classifier(model_name, y_test, test_probs, th)
        v_rep['train_time_sec'] = round(elapsed, 2)
        t_rep['train_time_sec'] = round(elapsed, 2)

        validation_reports.append(v_rep)
        test_reports.append(t_rep)
        print(f" -> Val F1: {v_rep['f1']:.4f} | PR-AUC: {v_rep['pr_auc']:.4f} | Test F1: {t_rep['f1']:.4f} | Elapsed: {elapsed:.1f}s")

    # Soft Voting Blended Ensemble
    print('\nComputing Weighted Soft-Voting Ensemble (CatBoost + LightGBM + XGBoost)...')
    ensemble_val_probs = (
        val_probabilities['catboost'] * 0.35 +
        val_probabilities['lightgbm'] * 0.35 +
        val_probabilities['xgboost'] * 0.30
    )
    ensemble_test_probs = (
        test_probabilities['catboost'] * 0.35 +
        test_probabilities['lightgbm'] * 0.35 +
        test_probabilities['xgboost'] * 0.30
    )
    ensemble_th = choose_threshold(y_val, ensemble_val_probs)
    ens_v_rep = evaluate_classifier('voting_ensemble', y_val, ensemble_val_probs, ensemble_th)
    ens_t_rep = evaluate_classifier('voting_ensemble', y_test, ensemble_test_probs, ensemble_th)
    ens_v_rep['train_time_sec'] = sum(r['train_time_sec'] for r in validation_reports if r['model'] in ('catboost', 'lightgbm', 'xgboost'))
    ens_t_rep['train_time_sec'] = ens_v_rep['train_time_sec']
    validation_reports.append(ens_v_rep)
    test_reports.append(ens_t_rep)

    val_df = pd.DataFrame(validation_reports).sort_values('pr_auc', ascending=False)
    test_df = pd.DataFrame(test_reports).sort_values('pr_auc', ascending=False)

    print('\n=== VALIDATION BENCHMARK SUMMARY ===')
    print(val_df[['model', 'f1', 'precision', 'recall', 'pr_auc', 'roc_auc', 'threshold', 'brier_score']])

    print('\n=== TEST EVALUATION SUMMARY ===')
    print(test_df[['model', 'f1', 'precision', 'recall', 'pr_auc', 'roc_auc', 'threshold', 'brier_score']])

    # Select Best Model
    best_model_name = str(val_df.iloc[0]['model'])
    best_threshold = float(val_df.iloc[0]['threshold'])
    print(f'\nSelected Best Candidate: {best_model_name} (Threshold: {best_threshold:.4f})')

    if best_model_name == 'voting_ensemble':
        selected_test_probs = ensemble_test_probs
        selected_model_artifact = {
            'type': 'ensemble',
            'models': {
                'catboost': models['catboost'],
                'lightgbm': models['lightgbm'],
                'xgboost': models['xgboost'],
            },
            'weights': {'catboost': 0.35, 'lightgbm': 0.35, 'xgboost': 0.30},
            'imputer': imputer,
        }
    else:
        selected_test_probs = test_probabilities[best_model_name]
        selected_model_artifact = {
            'type': 'single',
            'model': models[best_model_name],
            'imputer': imputer,
        }

    # Subgroup Fairness Audit across Grade Levels
    print('\nConducting Subgroup Fairness Audit across Grade Levels...')
    test_grades = data.iloc[test_indices]['grade_level_id'].to_numpy()
    test_preds = (selected_test_probs >= best_threshold).astype(int)
    fairness_records = []
    for g in sorted(np.unique(test_grades)):
        mask_g = (test_grades == g)
        y_g = y_test.iloc[mask_g]
        preds_g = test_preds[mask_g]
        rec_g = recall_score(y_g, preds_g, zero_division=0)
        prec_g = precision_score(y_g, preds_g, zero_division=0)
        f1_g = f1_score(y_g, preds_g, zero_division=0)
        sel_rate_g = preds_g.mean()
        fairness_records.append({
            'grade_level_id': int(g),
            'students_count': int(mask_g.sum()),
            'actual_risk_rate': float(y_g.mean()),
            'predicted_selection_rate': float(sel_rate_g),
            'recall': float(rec_g),
            'precision': float(prec_g),
            'f1': float(f1_g),
        })
    fairness_df = pd.DataFrame(fairness_records)
    print(fairness_df)

    # Feature Importance (from CatBoost & LightGBM)
    print('\nExtracting Feature Importances...')
    cb_feat_imp = pd.Series(models['catboost'].get_feature_importance(), index=FEATURE_COLUMNS).sort_values(ascending=False)
    lgb_feat_imp = pd.Series(models['lightgbm'].feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)
    combined_imp = (cb_feat_imp / cb_feat_imp.sum() + lgb_feat_imp / lgb_feat_imp.sum()) / 2.0
    combined_imp = combined_imp.sort_values(ascending=False)

    print('Top 15 Most Influential Risk Predictors:')
    print(combined_imp.head(15))

    # Save Run Artifacts
    run_id = f'student-risk-advanced-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}'
    run_dir = RUNS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    print(f'\nSaving Run Artifacts in: {run_dir}')
    joblib.dump({
        'pipeline': selected_model_artifact,
        'feature_columns': FEATURE_COLUMNS,
        'threshold': best_threshold,
        'best_model_name': best_model_name,
        'generated_at': datetime.now(timezone.utc).isoformat(),
    }, run_dir / 'model.joblib')

    # Save Predictions CSV
    pd.DataFrame({
        'student_key': data.iloc[test_indices]['student_key'].to_numpy(),
        'snapshot_date': data.iloc[test_indices]['snapshot_date'].to_numpy(),
        'grade_level_id': test_grades,
        'actual_label': y_test.to_numpy(),
        'predicted_probability': selected_test_probs,
        'predicted_binary_risk': test_preds,
    }).to_csv(run_dir / 'test-predictions.csv', index=False)

    # Save Plots
    # 1. Confusion Matrix
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, test_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Stable (0)', 'At Risk (1)'],
                yticklabels=['Stable (0)', 'At Risk (1)'])
    plt.title(f'Confusion Matrix: {best_model_name}\n(Threshold = {best_threshold:.3f})')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(run_dir / 'confusion_matrix.png', dpi=200)
    plt.close()

    # 2. Feature Importance Plot
    plt.figure(figsize=(10, 6))
    top_15 = combined_imp.head(15).sort_values(ascending=True)
    top_15.plot(kind='barh', color='#1f77b4')
    plt.title('Top 15 Most Important Risk Predictors')
    plt.xlabel('Normalized Importance Weight')
    plt.tight_layout()
    plt.savefig(run_dir / 'feature_importance.png', dpi=200)
    plt.close()

    # 3. Subgroup Fairness Plot
    plt.figure(figsize=(9, 5))
    plt.plot(fairness_df['grade_level_id'], fairness_df['recall'], marker='o', linewidth=2.5, color='#2ca02c', label='Recall (Sensitivity)')
    plt.plot(fairness_df['grade_level_id'], fairness_df['precision'], marker='s', linewidth=2.0, color='#1f77b4', label='Precision')
    plt.plot(fairness_df['grade_level_id'], fairness_df['f1'], marker='^', linewidth=2.0, color='#d62728', label='F1-Score')
    plt.title('Subgroup Fairness & Performance Parity Across Grade Levels')
    plt.xlabel('Grade Level (1 to 7)')
    plt.ylabel('Metric Score')
    plt.ylim([0.7, 0.9])
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / 'subgroup_fairness.png', dpi=200)
    plt.close()

    # Save Full JSON Report
    report = {
        'run_id': run_id,
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'dataset_kind': 'synthetic',
        'dataset_metadata': dataset_metadata,
        'features_count': len(FEATURE_COLUMNS),
        'engineered_features': [c for c in FEATURE_COLUMNS if c.startswith('feat_')],
        'train_samples': len(train_indices),
        'val_samples': len(validation_indices),
        'test_samples': len(test_indices),
        'best_model': best_model_name,
        'best_threshold': best_threshold,
        'validation_results': val_df.to_dict(orient='records'),
        'test_results': test_df.to_dict(orient='records'),
        'fairness_audit': fairness_records,
        'top_features_importance': combined_imp.head(20).to_dict(),
    }
    (run_dir / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')

    print('\nPipeline execution successfully completed!')
    print(f'Artifacts and visual reports saved in: {run_dir}')


if __name__ == '__main__':
    main()
