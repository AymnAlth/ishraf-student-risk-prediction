import "dotenv/config";

import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

import { Pool, type PoolClient } from "pg";
import { z } from "zod";

import {
  STUDENT_RISK_ATTENDANCE_THRESHOLD_PERCENT,
  STUDENT_RISK_HORIZON_DAYS,
  STUDENT_RISK_LOOKBACK_DAYS,
  STUDENT_RISK_MIN_LABEL_OBSERVATIONS
} from "../ml/student-risk/student-risk-dataset.util";
import { STUDENT_RISK_SNAPSHOT_QUERY } from "../ml/student-risk/student-risk-dataset.sql";

const scriptEnvSchema = z.object({
  DATABASE_URL: z.string().min(1),
  DATABASE_URL_MIGRATIONS: z.string().min(1).optional(),
  DATABASE_SCHEMA: z
    .string()
    .regex(/^[A-Za-z_][A-Za-z0-9_]*$/)
    .default("public")
});

interface TableEstimateRow {
  table_name: string;
  estimated_row_count: string;
}

interface ViewRow {
  table_name: string;
}

interface ColumnRow {
  table_name: string;
  column_name: string;
  data_type: string;
  is_nullable: "YES" | "NO";
}

interface ForeignKeyRow {
  source_table: string;
  source_column: string;
  target_table: string;
  target_column: string;
}

interface AcademicYearRow {
  id: string;
  start_date: string;
  end_date: string;
  is_active: boolean;
}

interface StudentRiskReadinessRow {
  snapshot_count: string;
  unique_student_count: string;
  positive_label_count: string;
  negative_label_count: string;
}

interface OperationalCoverageRow {
  active_students: string;
  attendance_records: string;
  attendance_sessions: string;
  student_assessment_records: string;
  homework_submissions: string;
  behavior_records: string;
  transport_events: string;
  attendance_first_date: string | null;
  attendance_last_date: string | null;
  assessment_first_date: string | null;
  assessment_last_date: string | null;
  behavior_first_date: string | null;
  behavior_last_date: string | null;
  transport_first_date: string | null;
  transport_last_date: string | null;
  expired_active_academic_years: string;
}

const outputDirectory = path.resolve(process.cwd(), "ml", "artifacts");

const candidateSummaryQuery = `
WITH student_risk_snapshots AS (
  ${STUDENT_RISK_SNAPSHOT_QUERY}
)
SELECT
  COUNT(*)::integer AS snapshot_count,
  COUNT(DISTINCT student_id)::integer AS unique_student_count,
  COUNT(*) FILTER (WHERE label_at_risk_next_14d)::integer AS positive_label_count,
  COUNT(*) FILTER (WHERE NOT label_at_risk_next_14d)::integer AS negative_label_count
FROM student_risk_snapshots
`;

const operationalCoverageQuery = `
SELECT
  (SELECT COUNT(*)::integer FROM students WHERE status = 'active') AS active_students,
  (SELECT COUNT(*)::integer FROM attendance) AS attendance_records,
  (SELECT COUNT(*)::integer FROM attendance_sessions) AS attendance_sessions,
  (SELECT COUNT(*)::integer FROM student_assessments) AS student_assessment_records,
  (SELECT COUNT(*)::integer FROM homework_submissions) AS homework_submissions,
  (SELECT COUNT(*)::integer FROM behavior_records) AS behavior_records,
  (SELECT COUNT(*)::integer FROM trip_student_events) AS transport_events,
  (SELECT MIN(session_date)::text FROM attendance_sessions) AS attendance_first_date,
  (SELECT MAX(session_date)::text FROM attendance_sessions) AS attendance_last_date,
  (SELECT MIN(assessment_date)::text FROM assessments) AS assessment_first_date,
  (SELECT MAX(assessment_date)::text FROM assessments) AS assessment_last_date,
  (SELECT MIN(behavior_date)::text FROM behavior_records) AS behavior_first_date,
  (SELECT MAX(behavior_date)::text FROM behavior_records) AS behavior_last_date,
  (SELECT MIN(event_time)::date::text FROM trip_student_events) AS transport_first_date,
  (SELECT MAX(event_time)::date::text FROM trip_student_events) AS transport_last_date,
  (
    SELECT COUNT(*)::integer
    FROM academic_years
    WHERE is_active = TRUE
      AND (CURRENT_DATE < start_date OR CURRENT_DATE > end_date)
  ) AS expired_active_academic_years
`;

const isSensitiveColumn = (columnName: string): boolean =>
  /(^|_)(academic_no|name|email|phone|mobile|address|password|token|date_of_birth|birth|dob|latitude|longitude|location|notes|remarks|description|device|fcm)(_|$)/i.test(
    columnName
  );

const writeJson = async (filePath: string, value: unknown): Promise<void> => {
  await writeFile(filePath, `${JSON.stringify(value, null, 2)}\n`, "utf8");
};

const profileSchema = async (client: PoolClient, schema: string) => {
  const tableEstimates = await client.query<TableEstimateRow>(
    `
      SELECT
        class.relname AS table_name,
        GREATEST(class.reltuples, 0)::bigint::text AS estimated_row_count
      FROM pg_class AS class
      JOIN pg_namespace AS namespace
        ON namespace.oid = class.relnamespace
      WHERE namespace.nspname = $1
        AND class.relkind = 'r'
      ORDER BY class.relname
    `,
    [schema]
  );
  const views = await client.query<ViewRow>(
    `
      SELECT table_name
      FROM information_schema.views
      WHERE table_schema = $1
      ORDER BY table_name
    `,
    [schema]
  );
  const columns = await client.query<ColumnRow>(
    `
      SELECT table_name, column_name, data_type, is_nullable
      FROM information_schema.columns
      WHERE table_schema = $1
      ORDER BY table_name, ordinal_position
    `,
    [schema]
  );
  const foreignKeys = await client.query<ForeignKeyRow>(
    `
      SELECT
        constraint_table.table_name AS source_table,
        source_column.column_name AS source_column,
        target_table.table_name AS target_table,
        target_column.column_name AS target_column
      FROM information_schema.table_constraints AS constraint_table
      JOIN information_schema.key_column_usage AS source_column
        ON source_column.constraint_name = constraint_table.constraint_name
        AND source_column.constraint_schema = constraint_table.constraint_schema
      JOIN information_schema.constraint_column_usage AS target_column
        ON target_column.constraint_name = constraint_table.constraint_name
        AND target_column.constraint_schema = constraint_table.constraint_schema
      JOIN information_schema.tables AS target_table
        ON target_table.table_schema = target_column.table_schema
        AND target_table.table_name = target_column.table_name
      WHERE constraint_table.table_schema = $1
        AND constraint_table.constraint_type = 'FOREIGN KEY'
      ORDER BY source_table, source_column
    `,
    [schema]
  );

  return {
    tables: tableEstimates.rows,
    views: views.rows,
    columns: columns.rows.map((column) => ({
      ...column,
      sensitiveOrFreeText: isSensitiveColumn(column.column_name)
    })),
    foreignKeys: foreignKeys.rows
  };
};

const buildMarkdown = (profile: {
  generatedAt: string;
  schema: string;
  operationalCoverage: OperationalCoverageRow;
  studentRiskReadiness: StudentRiskReadinessRow;
  tableCount: number;
  viewCount: number;
  sensitiveColumnCount: number;
}): string => {
  const coverage = profile.operationalCoverage;
  const readiness = profile.studentRiskReadiness;

  return `# ML Data Profile

Generated: ${profile.generatedAt}

## Schema Inventory

- Schema: \`${profile.schema}\`
- Base tables: ${profile.tableCount}
- Views: ${profile.viewCount}
- Columns requiring exclusion or review: ${profile.sensitiveColumnCount}

## Operational Coverage

- Active students: ${coverage.active_students}
- Attendance: ${coverage.attendance_records} records across ${coverage.attendance_sessions} sessions (${coverage.attendance_first_date ?? "n/a"} to ${coverage.attendance_last_date ?? "n/a"})
- Assessments: ${coverage.student_assessment_records} student records (${coverage.assessment_first_date ?? "n/a"} to ${coverage.assessment_last_date ?? "n/a"})
- Homework submissions: ${coverage.homework_submissions}
- Behavior records: ${coverage.behavior_records} (${coverage.behavior_first_date ?? "n/a"} to ${coverage.behavior_last_date ?? "n/a"})
- Transport events: ${coverage.transport_events} (${coverage.transport_first_date ?? "n/a"} to ${coverage.transport_last_date ?? "n/a"})
- Active academic years outside their configured date range: ${coverage.expired_active_academic_years}

## Student Risk Snapshot Readiness

- Snapshot rows: ${readiness.snapshot_count}
- Unique students: ${readiness.unique_student_count}
- Positive labels: ${readiness.positive_label_count}
- Negative labels: ${readiness.negative_label_count}
- Contract: ${STUDENT_RISK_LOOKBACK_DAYS}-day lookback, ${STUDENT_RISK_HORIZON_DAYS}-day future horizon, at least ${STUDENT_RISK_MIN_LABEL_OBSERVATIONS} future observations, at-risk when future present attendance is below ${STUDENT_RISK_ATTENDANCE_THRESHOLD_PERCENT}%.

This profile is descriptive only. It does not establish that the dataset is sufficient for production training. Review class balance, temporal coverage, data freshness, and student-level train/test separation before model experimentation.
`;
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

      const schemaInventory = await profileSchema(client, env.DATABASE_SCHEMA);
      const academicYears = await client.query<AcademicYearRow>(
        `
          SELECT id::text, start_date::text, end_date::text, is_active
          FROM academic_years
          ORDER BY start_date, id
        `
      );
      const operationalCoverage = await client.query<OperationalCoverageRow>(
        operationalCoverageQuery
      );
      const studentRiskReadiness = await client.query<StudentRiskReadinessRow>(
        candidateSummaryQuery,
        [
          STUDENT_RISK_LOOKBACK_DAYS,
          STUDENT_RISK_HORIZON_DAYS,
          STUDENT_RISK_MIN_LABEL_OBSERVATIONS,
          STUDENT_RISK_ATTENDANCE_THRESHOLD_PERCENT
        ]
      );

      await client.query("COMMIT");

      const coverage = operationalCoverage.rows[0];
      const readiness = studentRiskReadiness.rows[0];

      if (!coverage || !readiness) {
        throw new Error("ML profile queries returned no aggregate rows");
      }

      const sensitiveColumnCount = schemaInventory.columns.filter(
        (column) => column.sensitiveOrFreeText
      ).length;
      const profile = {
        generatedAt: new Date().toISOString(),
        schema: env.DATABASE_SCHEMA,
        academicYears: academicYears.rows,
        operationalCoverage: coverage,
        studentRiskReadiness: readiness,
        schemaInventory,
        notes: [
          "Table row counts are PostgreSQL planner estimates to avoid full scans of every operational table.",
          "The profile contains table metadata and aggregate metrics only; it contains no operational records or direct identifiers.",
          "Columns marked sensitiveOrFreeText must not be included in ML feature exports without a separately approved privacy review."
        ]
      };

      await mkdir(outputDirectory, { recursive: true });
      await writeJson(
        path.join(outputDirectory, "database-profile.json"),
        profile
      );
      await writeFile(
        path.join(outputDirectory, "database-profile.md"),
        buildMarkdown({
          generatedAt: profile.generatedAt,
          schema: profile.schema,
          operationalCoverage: coverage,
          studentRiskReadiness: readiness,
          tableCount: schemaInventory.tables.length,
          viewCount: schemaInventory.views.length,
          sensitiveColumnCount
        }),
        "utf8"
      );

      console.log(
        `Created a read-only ML profile for ${schemaInventory.tables.length} tables and ${schemaInventory.views.length} views in ${outputDirectory}.`
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
