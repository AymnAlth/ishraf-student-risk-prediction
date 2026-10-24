# Student Risk Dataset Contract

## Objective

Each row represents one student at the end of a historical attendance date. The target is `label_at_risk_next_14d`: `true` when the student's recorded **present** attendance in the following 14 calendar days is below 85%, with at least three future attendance records. This mirrors the existing attendance summary convention, where only `present` contributes to the present count.

## Temporal Rules

- Features use the 28 calendar days ending on `snapshot_date` only.
- Assessment features require `graded_at` on or before `snapshot_date`.
- Homework features include work assigned and due on or before `snapshot_date`; a submission is visible only when its `updated_at` is not later than that date.
- Behavior and transport features use events no later than `snapshot_date`.
- The future 14-day window is label-only. Its counts and percentage must never be provided to a model as input.
- Any training split must group by `student_key` and be time-aware. Random row splitting would leak the same student across training and evaluation.

## Feature Sources

| Feature family | Source tables                                         | Exported information                                     |
| -------------- | ----------------------------------------------------- | -------------------------------------------------------- |
| Attendance     | `attendance`, `attendance_sessions`                   | Rolling counts and present percentage                    |
| Assessment     | `student_assessments`, `assessments`                  | Count, average, and minimum normalized score             |
| Homework       | `homework`, `homework_submissions`                    | Due, completed, late, and missing counts                 |
| Behavior       | `behavior_records`, `behavior_categories`             | Positive/negative event counts and negative severity sum |
| Transport      | `student_bus_assignments`, `trip_student_events`      | Assignment flag and aggregate boarded/absent events      |
| Context        | `students`, `student_academic_enrollments`, `classes` | Grade level and days since enrollment                    |

## Exclusions

The dataset must not contain direct identifiers or unreviewed sensitive fields: student database IDs, academic numbers, names, dates of birth, gender, parent data, contact data, addresses, home coordinates, bus stops, GPS, messages, notes, descriptions, remarks, tokens, or device data.

`student_key` is a pseudonymous HMAC key, not a feature. It exists solely to keep records of one student in the same evaluation partition.

## Readiness Gate

An exported file is a course dataset, not proof of model readiness. Before an offline experiment advances to a production candidate, a data owner must approve:

1. Enough distinct students and academic periods to evaluate generalization.
2. Sufficient examples of both labels without synthetic records mixed into real evaluation data.
3. A time-based, student-grouped holdout set.
4. Performance, calibration, fairness, and failure-mode review.
5. A human review workflow; no automated discipline, academic, transport, or access decision may be made from the prediction.

## Synthetic Companion Dataset

`student-risk-synthetic-training.csv` uses the same columns and temporal contract but is generated from fictional student profiles. It is intentionally separate from `student-risk-training.csv`, the small pseudonymized operational export. Do not concatenate the files for real-world validation and do not load either file back into the operational database.
