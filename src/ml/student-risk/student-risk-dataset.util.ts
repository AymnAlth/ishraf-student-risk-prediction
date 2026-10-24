import { createHmac } from "node:crypto";

export const STUDENT_RISK_LOOKBACK_DAYS = 28;
export const STUDENT_RISK_HORIZON_DAYS = 14;
export const STUDENT_RISK_MIN_LABEL_OBSERVATIONS = 3;
export const STUDENT_RISK_ATTENDANCE_THRESHOLD_PERCENT = 85;

export const STUDENT_RISK_DATASET_COLUMNS = [
  "student_key",
  "snapshot_date",
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
  "label_observation_count",
  "label_future_attendance_percent",
  "label_at_risk_next_14d"
] as const;

export type StudentRiskDatasetColumn =
  (typeof STUDENT_RISK_DATASET_COLUMNS)[number];

export interface StudentRiskDatabaseRow {
  student_id: string;
  cutoff_date: string;
  grade_level_id: string | null;
  days_since_enrollment: string | null;
  lookback_attendance_records: string;
  lookback_present_count: string;
  lookback_absent_count: string;
  lookback_late_count: string;
  lookback_excused_count: string;
  lookback_attendance_percent: string | null;
  lookback_assessment_count: string;
  lookback_assessment_average_percent: string | null;
  lookback_assessment_minimum_percent: string | null;
  lookback_homework_due_count: string;
  lookback_homework_completed_count: string;
  lookback_homework_late_count: string;
  lookback_homework_missing_count: string;
  lookback_negative_behavior_count: string;
  lookback_positive_behavior_count: string;
  lookback_negative_behavior_severity_sum: string;
  transport_assigned_at_snapshot: boolean;
  lookback_transport_boarded_count: string;
  lookback_transport_absent_count: string;
  label_observation_count: string;
  label_future_attendance_percent: string;
  label_at_risk_next_14d: boolean;
}

export type StudentRiskDatasetRow = Record<
  StudentRiskDatasetColumn,
  boolean | number | string | null
>;

const parseNullableNumber = (value: string | null): number | null => {
  if (value === null) {
    return null;
  }

  const parsed = Number(value);

  if (!Number.isFinite(parsed)) {
    throw new Error(`Expected numeric database value but received ${value}`);
  }

  return parsed;
};

const parseRequiredNumber = (value: string): number => {
  const parsed = parseNullableNumber(value);

  if (parsed === null) {
    throw new Error("Expected required numeric database value");
  }

  return parsed;
};

export const createStudentKey = (
  studentId: string,
  pseudonymizationSalt: string
): string => {
  if (!/^[0-9]+$/.test(studentId)) {
    throw new Error("Student ID must be a positive integer string");
  }

  if (pseudonymizationSalt.length < 32) {
    throw new Error(
      "ML_EXPORT_PSEUDONYMIZATION_SALT must contain at least 32 characters"
    );
  }

  return createHmac("sha256", pseudonymizationSalt)
    .update(`student-risk-v1:${studentId}`)
    .digest("hex");
};

export const toStudentRiskDatasetRow = (
  row: StudentRiskDatabaseRow,
  pseudonymizationSalt: string
): StudentRiskDatasetRow => ({
  student_key: createStudentKey(row.student_id, pseudonymizationSalt),
  snapshot_date: row.cutoff_date,
  grade_level_id: parseNullableNumber(row.grade_level_id),
  days_since_enrollment: parseNullableNumber(row.days_since_enrollment),
  lookback_attendance_records: parseRequiredNumber(
    row.lookback_attendance_records
  ),
  lookback_present_count: parseRequiredNumber(row.lookback_present_count),
  lookback_absent_count: parseRequiredNumber(row.lookback_absent_count),
  lookback_late_count: parseRequiredNumber(row.lookback_late_count),
  lookback_excused_count: parseRequiredNumber(row.lookback_excused_count),
  lookback_attendance_percent: parseNullableNumber(
    row.lookback_attendance_percent
  ),
  lookback_assessment_count: parseRequiredNumber(row.lookback_assessment_count),
  lookback_assessment_average_percent: parseNullableNumber(
    row.lookback_assessment_average_percent
  ),
  lookback_assessment_minimum_percent: parseNullableNumber(
    row.lookback_assessment_minimum_percent
  ),
  lookback_homework_due_count: parseRequiredNumber(
    row.lookback_homework_due_count
  ),
  lookback_homework_completed_count: parseRequiredNumber(
    row.lookback_homework_completed_count
  ),
  lookback_homework_late_count: parseRequiredNumber(
    row.lookback_homework_late_count
  ),
  lookback_homework_missing_count: parseRequiredNumber(
    row.lookback_homework_missing_count
  ),
  lookback_negative_behavior_count: parseRequiredNumber(
    row.lookback_negative_behavior_count
  ),
  lookback_positive_behavior_count: parseRequiredNumber(
    row.lookback_positive_behavior_count
  ),
  lookback_negative_behavior_severity_sum: parseRequiredNumber(
    row.lookback_negative_behavior_severity_sum
  ),
  transport_assigned_at_snapshot: row.transport_assigned_at_snapshot,
  lookback_transport_boarded_count: parseRequiredNumber(
    row.lookback_transport_boarded_count
  ),
  lookback_transport_absent_count: parseRequiredNumber(
    row.lookback_transport_absent_count
  ),
  label_observation_count: parseRequiredNumber(row.label_observation_count),
  label_future_attendance_percent: parseRequiredNumber(
    row.label_future_attendance_percent
  ),
  label_at_risk_next_14d: row.label_at_risk_next_14d
});

export const escapeCsvValue = (
  value: StudentRiskDatasetRow[StudentRiskDatasetColumn]
): string => {
  if (value === null) {
    return "";
  }

  const serialized = String(value);

  return /[",\n\r]/.test(serialized)
    ? `"${serialized.replace(/"/g, '""')}"`
    : serialized;
};

export const toStudentRiskCsv = (rows: StudentRiskDatasetRow[]): string => {
  const header = STUDENT_RISK_DATASET_COLUMNS.join(",");
  const dataLines = rows.map((row) =>
    STUDENT_RISK_DATASET_COLUMNS.map((column) =>
      escapeCsvValue(row[column])
    ).join(",")
  );

  return [header, ...dataLines].join("\n");
};
