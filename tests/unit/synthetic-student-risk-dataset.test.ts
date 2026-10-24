import { describe, expect, it } from "vitest";

import { generateSyntheticStudentRiskRows } from "../../src/ml/student-risk/synthetic-student-risk-dataset";
import { STUDENT_RISK_DATASET_COLUMNS } from "../../src/ml/student-risk/student-risk-dataset.util";

const options = {
  studentCount: 3,
  snapshotsPerStudent: 8,
  seed: "synthetic-student-risk-unit-test"
};

describe("synthetic student risk dataset", () => {
  it("is deterministic for a configured seed", () => {
    const firstRun = [...generateSyntheticStudentRiskRows(options)];
    const secondRun = [...generateSyntheticStudentRiskRows(options)];

    expect(firstRun).toEqual(secondRun);
  });

  it("creates the configured number of non-identifying rows", () => {
    const rows = [...generateSyntheticStudentRiskRows(options)];

    expect(rows).toHaveLength(
      options.studentCount * options.snapshotsPerStudent
    );
    expect(new Set(rows.map((row) => row.student_key)).size).toBe(
      options.studentCount
    );
    expect(Object.keys(rows[0] ?? {})).toEqual(STUDENT_RISK_DATASET_COLUMNS);
    expect(
      rows.every((row) => row.student_key.startsWith("synthetic_student_"))
    ).toBe(true);
  });

  it("keeps every feature internally consistent", () => {
    const rows = [...generateSyntheticStudentRiskRows(options)];

    for (const row of rows) {
      expect(
        row.lookback_present_count +
          row.lookback_absent_count +
          row.lookback_late_count +
          row.lookback_excused_count
      ).toBe(row.lookback_attendance_records);
      expect(
        row.lookback_homework_completed_count +
          row.lookback_homework_missing_count
      ).toBe(row.lookback_homework_due_count);
      expect(row.label_at_risk_next_14d).toBe(
        row.label_future_attendance_percent < 85
      );
    }
  });
});
