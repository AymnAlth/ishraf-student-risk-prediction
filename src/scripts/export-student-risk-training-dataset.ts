import "dotenv/config";

import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

import { Pool } from "pg";
import { z } from "zod";

import {
  STUDENT_RISK_ATTENDANCE_THRESHOLD_PERCENT,
  STUDENT_RISK_HORIZON_DAYS,
  STUDENT_RISK_LOOKBACK_DAYS,
  STUDENT_RISK_MIN_LABEL_OBSERVATIONS,
  toStudentRiskCsv,
  toStudentRiskDatasetRow,
  type StudentRiskDatabaseRow
} from "../ml/student-risk/student-risk-dataset.util";
import { STUDENT_RISK_SNAPSHOT_QUERY } from "../ml/student-risk/student-risk-dataset.sql";

const scriptEnvSchema = z.object({
  DATABASE_URL: z.string().min(1),
  DATABASE_URL_MIGRATIONS: z.string().min(1).optional(),
  ML_EXPORT_PSEUDONYMIZATION_SALT: z.string().min(32)
});

const outputDirectory = path.resolve(
  process.cwd(),
  "ml",
  "artifacts",
  "student-risk"
);

const writeJson = async (filePath: string, value: unknown): Promise<void> => {
  await writeFile(filePath, `${JSON.stringify(value, null, 2)}\n`, "utf8");
};

const main = async (): Promise<void> => {
  const env = scriptEnvSchema.parse(process.env);
  const pool = new Pool({
    connectionString: env.DATABASE_URL_MIGRATIONS ?? env.DATABASE_URL,
    max: 1
  });

  try {
    const client = await pool.connect();

    try {
      await client.query(
        "BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY"
      );
      await client.query("SET LOCAL TIME ZONE 'UTC'");
      await client.query("SET LOCAL statement_timeout = '60s'");

      const result = await client.query<StudentRiskDatabaseRow>(
        STUDENT_RISK_SNAPSHOT_QUERY,
        [
          STUDENT_RISK_LOOKBACK_DAYS,
          STUDENT_RISK_HORIZON_DAYS,
          STUDENT_RISK_MIN_LABEL_OBSERVATIONS,
          STUDENT_RISK_ATTENDANCE_THRESHOLD_PERCENT
        ]
      );

      await client.query("COMMIT");

      const datasetRows = result.rows.map((row) =>
        toStudentRiskDatasetRow(row, env.ML_EXPORT_PSEUDONYMIZATION_SALT)
      );
      const positiveLabels = datasetRows.filter(
        (row) => row.label_at_risk_next_14d
      ).length;
      const uniqueStudents = new Set(datasetRows.map((row) => row.student_key))
        .size;

      await mkdir(outputDirectory, { recursive: true });
      await writeFile(
        path.join(outputDirectory, "student-risk-training.csv"),
        `${toStudentRiskCsv(datasetRows)}\n`,
        "utf8"
      );
      await writeJson(
        path.join(outputDirectory, "student-risk-training.metadata.json"),
        {
          generatedAt: new Date().toISOString(),
          dataset: "student-risk-training",
          privacy: {
            containsDirectIdentifiers: false,
            studentKey: "HMAC-SHA-256 using ML_EXPORT_PSEUDONYMIZATION_SALT",
            excluded: [
              "student_id",
              "names",
              "academic numbers",
              "email addresses",
              "phone numbers",
              "birth dates",
              "addresses",
              "free-text notes",
              "GPS coordinates"
            ]
          },
          temporalContract: {
            lookbackDays: STUDENT_RISK_LOOKBACK_DAYS,
            labelHorizonDays: STUDENT_RISK_HORIZON_DAYS,
            minimumLabelAttendanceRecords: STUDENT_RISK_MIN_LABEL_OBSERVATIONS,
            riskThreshold: `future present attendance < ${STUDENT_RISK_ATTENDANCE_THRESHOLD_PERCENT}%`
          },
          rows: datasetRows.length,
          uniqueStudents,
          positiveLabels,
          negativeLabels: datasetRows.length - positiveLabels,
          trainingReadiness: "requires_statistical_validation"
        }
      );

      console.log(
        `Created ${datasetRows.length} pseudonymized student-risk snapshots for ${uniqueStudents} students in ${outputDirectory}.`
      );
    } catch (error) {
      await client.query("ROLLBACK").catch(() => undefined);
      throw error;
    } finally {
      client.release();
    }
  } finally {
    await pool.end();
  }
};

if (require.main === module) {
  void main().catch((error: unknown) => {
    console.error(error);
    process.exit(1);
  });
}
