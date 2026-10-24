import { describe, expect, it } from "vitest";

import {
  STUDENT_RISK_DATASET_COLUMNS,
  createStudentKey,
  toStudentRiskCsv,
  toStudentRiskDatasetRow,
  type StudentRiskDatabaseRow
} from "../../src/ml/student-risk/student-risk-dataset.util";

const pseudonymizationSalt = "this-is-a-test-only-pseudonymization-salt-value";

const databaseRow: StudentRiskDatabaseRow = {
  student_id: "42",
  cutoff_date: "2026-04-01",
  grade_level_id: "3",
  days_since_enrollment: "180",
  lookback_attendance_records: "12",
  lookback_present_count: "10",
  lookback_absent_count: "1",
  lookback_late_count: "1",
  lookback_excused_count: "0",
  lookback_attendance_percent: "83.33",
  lookback_assessment_count: "2",
  lookback_assessment_average_percent: "72.5",
  lookback_assessment_minimum_percent: "60",
  lookback_homework_due_count: "4",
  lookback_homework_completed_count: "3",
  lookback_homework_late_count: "1",
  lookback_homework_missing_count: "1",
  lookback_negative_behavior_count: "1",
  lookback_positive_behavior_count: "0",
  lookback_negative_behavior_severity_sum: "3",
  transport_assigned_at_snapshot: true,
  lookback_transport_boarded_count: "5",
  lookback_transport_absent_count: "1",
  label_observation_count: "4",
  label_future_attendance_percent: "75",
  label_at_risk_next_14d: true
};

describe("student risk dataset utilities", () => {
  it("creates deterministic pseudonymous keys without returning the source ID", () => {
    const firstKey = createStudentKey("42", pseudonymizationSalt);

    expect(firstKey).toBe(createStudentKey("42", pseudonymizationSalt));
    expect(firstKey).not.toBe("42");
    expect(firstKey).not.toBe(createStudentKey("43", pseudonymizationSalt));
  });

  it("converts an operational row to the approved non-PII dataset contract", () => {
    const row = toStudentRiskDatasetRow(databaseRow, pseudonymizationSalt);

    expect(Object.keys(row)).toEqual(STUDENT_RISK_DATASET_COLUMNS);
    expect(row.student_key).toHaveLength(64);
    expect(row.lookback_attendance_percent).toBe(83.33);
    expect(row.label_at_risk_next_14d).toBe(true);
    expect(row).not.toHaveProperty("student_id");
  });

  it("serializes a header and dataset values into CSV", () => {
    const csv = toStudentRiskCsv([
      toStudentRiskDatasetRow(databaseRow, pseudonymizationSalt)
    ]);

    expect(csv.split("\n")).toHaveLength(2);
    expect(csv.startsWith("student_key,snapshot_date,grade_level_id")).toBe(
      true
    );
  });
});
