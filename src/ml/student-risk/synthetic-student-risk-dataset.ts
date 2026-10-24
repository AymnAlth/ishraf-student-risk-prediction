import {
  type StudentRiskDatasetRow,
  STUDENT_RISK_DATASET_COLUMNS
} from "./student-risk-dataset.util";

export interface SyntheticStudentRiskDatasetOptions {
  studentCount: number;
  snapshotsPerStudent: number;
  seed: string;
}

interface SeededRandom {
  next(): number;
  integer(minimum: number, maximum: number): number;
  normal(): number;
}

interface SyntheticStudentProfile {
  studentKey: string;
  gradeLevelId: number;
  daysSinceEnrollmentAtFirstSnapshot: number;
  riskPropensity: number;
  usesTransport: boolean;
  cohortYear: number;
}

const clamp = (value: number, minimum: number, maximum: number): number =>
  Math.min(Math.max(value, minimum), maximum);

const hashSeed = (seed: string): number => {
  let hash = 2_166_136_261;

  for (const character of seed) {
    hash ^= character.charCodeAt(0);
    hash = Math.imul(hash, 16_777_619);
  }

  return hash >>> 0;
};

export const createSeededRandom = (seed: string): SeededRandom => {
  let state = hashSeed(seed);
  let spareNormal: number | null = null;

  const next = (): number => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);

    return ((value ^ (value >>> 14)) >>> 0) / 4_294_967_296;
  };

  return {
    next,
    integer: (minimum, maximum) =>
      Math.floor(next() * (maximum - minimum + 1)) + minimum,
    normal: () => {
      if (spareNormal !== null) {
        const value = spareNormal;

        spareNormal = null;
        return value;
      }

      const radius = Math.sqrt(
        -2 * Math.log(Math.max(next(), Number.MIN_VALUE))
      );
      const angle = 2 * Math.PI * next();

      spareNormal = radius * Math.sin(angle);
      return radius * Math.cos(angle);
    }
  };
};

const countEvents = (
  count: number,
  probability: number,
  random: SeededRandom
): number => {
  let total = 0;

  for (let index = 0; index < count; index += 1) {
    if (random.next() < probability) {
      total += 1;
    }
  }

  return total;
};

const toIsoDate = (year: number, weekOffset: number): string => {
  const date = new Date(Date.UTC(year, 8, 8 + weekOffset * 7));

  return date.toISOString().slice(0, 10);
};

const createStudentProfile = (
  studentIndex: number,
  random: SeededRandom
): SyntheticStudentProfile => ({
  studentKey: `synthetic_student_${String(studentIndex).padStart(6, "0")}`,
  gradeLevelId: random.integer(1, 7),
  daysSinceEnrollmentAtFirstSnapshot: random.integer(45, 1_250),
  // This latent variable creates realistic correlation without exposing it in the dataset.
  riskPropensity: Math.pow(random.next(), 1.45),
  usesTransport: random.next() < 0.62,
  cohortYear: 2023 + ((studentIndex - 1) % 3)
});

const createDatasetRow = (
  profile: SyntheticStudentProfile,
  snapshotIndex: number,
  random: SeededRandom
): StudentRiskDatasetRow => {
  const weeklyShock = clamp(random.normal() * 0.1, -0.22, 0.22);
  const historicRecords = random.integer(9, 24);
  const historicAbsentRate = clamp(
    0.015 + profile.riskPropensity * 0.2 + weeklyShock * 0.1,
    0.005,
    0.38
  );
  const historicLateRate = clamp(
    0.015 + profile.riskPropensity * 0.08 + weeklyShock * 0.04,
    0.005,
    0.18
  );
  const historicExcusedRate = clamp(0.01 + random.next() * 0.04, 0.005, 0.06);
  let absentCount = 0;
  let lateCount = 0;
  let excusedCount = 0;

  for (let recordIndex = 0; recordIndex < historicRecords; recordIndex += 1) {
    const statusRoll = random.next();

    if (statusRoll < historicAbsentRate) {
      absentCount += 1;
    } else if (statusRoll < historicAbsentRate + historicLateRate) {
      lateCount += 1;
    } else if (
      statusRoll <
      historicAbsentRate + historicLateRate + historicExcusedRate
    ) {
      excusedCount += 1;
    }
  }

  const presentCount = historicRecords - absentCount - lateCount - excusedCount;
  const assessmentCount = random.next() < 0.12 ? 0 : random.integer(1, 5);
  const assessmentAverage =
    assessmentCount === 0
      ? null
      : Math.round(
          clamp(
            91 -
              profile.riskPropensity * 43 -
              weeklyShock * 17 +
              random.normal() * 8,
            20,
            100
          ) * 100
        ) / 100;
  const assessmentMinimum =
    assessmentAverage === null
      ? null
      : Math.round(
          clamp(assessmentAverage - random.integer(0, 24), 0, 100) * 100
        ) / 100;
  const homeworkDueCount = random.next() < 0.08 ? 0 : random.integer(1, 7);
  const homeworkMissingCount = countEvents(
    homeworkDueCount,
    clamp(
      0.02 + profile.riskPropensity * 0.48 + weeklyShock * 0.12,
      0.005,
      0.82
    ),
    random
  );
  const homeworkLateCount = countEvents(
    homeworkDueCount - homeworkMissingCount,
    clamp(0.03 + profile.riskPropensity * 0.24, 0.01, 0.5),
    random
  );
  const homeworkCompletedCount = homeworkDueCount - homeworkMissingCount;
  const negativeBehaviorCount = countEvents(
    random.integer(0, 4),
    clamp(
      0.03 + profile.riskPropensity * 0.48 + Math.max(weeklyShock, 0),
      0.01,
      0.75
    ),
    random
  );
  const positiveBehaviorCount = countEvents(
    random.integer(0, 4),
    clamp(0.36 - profile.riskPropensity * 0.18, 0.05, 0.42),
    random
  );
  const negativeBehaviorSeveritySum = Array.from(
    { length: negativeBehaviorCount },
    () => random.integer(1, 4)
  ).reduce((total, severity) => total + severity, 0);
  const transportBoardedCount = profile.usesTransport
    ? countEvents(
        random.integer(5, 16),
        clamp(0.96 - profile.riskPropensity * 0.22, 0.55, 0.99),
        random
      )
    : 0;
  const transportAbsentCount = profile.usesTransport
    ? countEvents(
        random.integer(2, 8),
        clamp(0.015 + profile.riskPropensity * 0.18, 0.01, 0.35),
        random
      )
    : 0;
  const labelObservationCount = random.integer(8, 14);
  const futurePresentRate = clamp(
    0.99 -
      profile.riskPropensity * 0.28 -
      weeklyShock * 0.11 +
      random.normal() * 0.035,
    0.35,
    1
  );
  const futurePresentCount = clamp(
    Math.round(futurePresentRate * labelObservationCount),
    0,
    labelObservationCount
  );
  const labelFutureAttendancePercent =
    Math.round((100 * futurePresentCount * 100) / labelObservationCount) / 100;

  return {
    student_key: profile.studentKey,
    snapshot_date: toIsoDate(profile.cohortYear, snapshotIndex),
    grade_level_id: profile.gradeLevelId,
    days_since_enrollment:
      profile.daysSinceEnrollmentAtFirstSnapshot + snapshotIndex * 7,
    lookback_attendance_records: historicRecords,
    lookback_present_count: presentCount,
    lookback_absent_count: absentCount,
    lookback_late_count: lateCount,
    lookback_excused_count: excusedCount,
    lookback_attendance_percent:
      Math.round((100 * presentCount * 100) / historicRecords) / 100,
    lookback_assessment_count: assessmentCount,
    lookback_assessment_average_percent: assessmentAverage,
    lookback_assessment_minimum_percent: assessmentMinimum,
    lookback_homework_due_count: homeworkDueCount,
    lookback_homework_completed_count: homeworkCompletedCount,
    lookback_homework_late_count: homeworkLateCount,
    lookback_homework_missing_count: homeworkMissingCount,
    lookback_negative_behavior_count: negativeBehaviorCount,
    lookback_positive_behavior_count: positiveBehaviorCount,
    lookback_negative_behavior_severity_sum: negativeBehaviorSeveritySum,
    transport_assigned_at_snapshot: profile.usesTransport,
    lookback_transport_boarded_count: transportBoardedCount,
    lookback_transport_absent_count: transportAbsentCount,
    label_observation_count: labelObservationCount,
    label_future_attendance_percent: labelFutureAttendancePercent,
    label_at_risk_next_14d: labelFutureAttendancePercent < 85
  };
};

export function* generateSyntheticStudentRiskRows(
  options: SyntheticStudentRiskDatasetOptions
): Generator<StudentRiskDatasetRow> {
  const random = createSeededRandom(options.seed);

  for (
    let studentIndex = 1;
    studentIndex <= options.studentCount;
    studentIndex += 1
  ) {
    const profile = createStudentProfile(studentIndex, random);

    for (
      let snapshotIndex = 0;
      snapshotIndex < options.snapshotsPerStudent;
      snapshotIndex += 1
    ) {
      yield createDatasetRow(profile, snapshotIndex, random);
    }
  }
}

export const syntheticStudentRiskDatasetHeader =
  STUDENT_RISK_DATASET_COLUMNS.join(",");
