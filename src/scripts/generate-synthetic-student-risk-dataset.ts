import "dotenv/config";

import { appendFile, mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

import { z } from "zod";

import {
  generateSyntheticStudentRiskRows,
  syntheticStudentRiskDatasetHeader
} from "../ml/student-risk/synthetic-student-risk-dataset";
import {
  escapeCsvValue,
  STUDENT_RISK_DATASET_COLUMNS,
  type StudentRiskDatasetRow
} from "../ml/student-risk/student-risk-dataset.util";

const scriptEnvSchema = z.object({
  ML_SYNTHETIC_STUDENT_COUNT: z.coerce
    .number()
    .int()
    .min(100)
    .max(100_000)
    .default(10_000),
  ML_SYNTHETIC_SNAPSHOTS_PER_STUDENT: z.coerce
    .number()
    .int()
    .min(8)
    .max(100)
    .default(30),
  ML_SYNTHETIC_SEED: z
    .string()
    .min(8)
    .max(128)
    .default("ishraf-student-risk-synthetic-v1")
});

const outputDirectory = path.resolve(
  process.cwd(),
  "ml",
  "artifacts",
  "student-risk-synthetic"
);
const rowsPerWrite = 5_000;

const toCsvLine = (row: StudentRiskDatasetRow): string =>
  STUDENT_RISK_DATASET_COLUMNS.map((column) =>
    escapeCsvValue(row[column])
  ).join(",");

const main = async (): Promise<void> => {
  const env = scriptEnvSchema.parse(process.env);
  const outputPath = path.join(
    outputDirectory,
    "student-risk-synthetic-training.csv"
  );
  let rows = 0;
  let positiveLabels = 0;
  let bufferedLines: string[] = [];

  await mkdir(outputDirectory, { recursive: true });
  await writeFile(outputPath, `${syntheticStudentRiskDatasetHeader}\n`, "utf8");

  for (const row of generateSyntheticStudentRiskRows({
    studentCount: env.ML_SYNTHETIC_STUDENT_COUNT,
    snapshotsPerStudent: env.ML_SYNTHETIC_SNAPSHOTS_PER_STUDENT,
    seed: env.ML_SYNTHETIC_SEED
  })) {
    bufferedLines.push(toCsvLine(row));
    rows += 1;

    if (row.label_at_risk_next_14d) {
      positiveLabels += 1;
    }

    if (bufferedLines.length === rowsPerWrite) {
      await appendFile(outputPath, `${bufferedLines.join("\n")}\n`, "utf8");
      bufferedLines = [];
    }
  }

  if (bufferedLines.length > 0) {
    await appendFile(outputPath, `${bufferedLines.join("\n")}\n`, "utf8");
  }

  const negativeLabels = rows - positiveLabels;

  if (positiveLabels === 0 || negativeLabels === 0) {
    throw new Error(
      "Synthetic generator produced a single-class dataset; choose a different seed."
    );
  }

  await writeFile(
    path.join(outputDirectory, "student-risk-synthetic-training.metadata.json"),
    `${JSON.stringify(
      {
        generatedAt: new Date().toISOString(),
        dataset: "student-risk-synthetic-training",
        datasetKind: "synthetic",
        realOperationalRecordsIncluded: false,
        rows,
        uniqueStudents: env.ML_SYNTHETIC_STUDENT_COUNT,
        positiveLabels,
        negativeLabels,
        snapshotsPerStudent: env.ML_SYNTHETIC_SNAPSHOTS_PER_STUDENT,
        academicYearsCovered: [2023, 2024, 2025],
        seed: env.ML_SYNTHETIC_SEED,
        contract:
          "Matches the student-risk CSV schema, including its future-attendance label.",
        restrictions: [
          "Use only for ML coursework, pipeline validation, and controlled experiments.",
          "Do not merge with real operational records for production evaluation.",
          "Do not use model metrics trained on this file to claim production performance."
        ]
      },
      null,
      2
    )}\n`,
    "utf8"
  );

  console.log(
    `Created ${rows} synthetic student-risk snapshots for ${env.ML_SYNTHETIC_STUDENT_COUNT} students in ${outputDirectory}.`
  );
};

if (require.main === module) {
  void main().catch((error: unknown) => {
    console.error(error);
    process.exit(1);
  });
}
